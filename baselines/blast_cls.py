import sys
sys.path.append(".")
import os
import subprocess
import pandas as pd
import argparse
import json
from tqdm import tqdm
import tempfile
from collections import defaultdict

def load_venusx_dataset(dataset_name, split):
    """Load VenusX dataset from JSON file"""
    dataset_path = f"data/VenusX_{dataset_name}/{split}.json"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    return data

def extract_category_fragments(data):
    """Extract fragments organized by category for training set"""
    category_fragments = defaultdict(list)
    
    for protein in data:
        uid = protein['uid']
        for fragment_group in protein['fragments']:
            category = fragment_group['category']
            interpro_id = fragment_group['interpro_id']
            
            for i, frag in enumerate(fragment_group['frags']):
                fragment_info = {
                    'uid': uid,
                    'category': category,
                    'interpro_id': interpro_id,
                    'sequence': frag['sequence'],
                    'start_pos': frag['start_position'],
                    'end_pos': frag['end_position'],
                    'frag_id': f"{uid}_{interpro_id}_{i}"
                }
                category_fragments[category].append(fragment_info)
    
    return category_fragments

def create_blast_database(sequences, db_path):
    """Create BLAST database from sequences"""
    # Create temporary fasta file
    temp_fasta = f"{db_path}.fasta"
    with open(temp_fasta, 'w') as f:
        for seq_id, sequence in sequences:
            f.write(f">{seq_id}\n{sequence}\n")
    
    # Create BLAST database
    cmd = ["makeblastdb", "-in", temp_fasta, "-dbtype", "prot", "-out", db_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error creating BLAST database: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    
    # Clean up temporary fasta
    os.remove(temp_fasta)

def run_blastp_alignment(query_sequences, db_path, evalue_threshold=10):
    """Run BLASTP alignment with query sequences against database"""
    # Create temporary query file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as query_file:
        for seq_id, sequence in query_sequences:
            query_file.write(f">{seq_id}\n{sequence}\n")
        query_path = query_file.name
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as output_file:
        output_path = output_file.name
    
    # Run BLASTP
    outfmt_fields = "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore"
    cmd = [
        "blastp",
        "-query", query_path,
        "-db", db_path,
        "-evalue", str(evalue_threshold),
        "-word_size", "2",
        "-max_target_seqs", "10000",  # Allow multiple hits
        "-seg", "no",
        "-out", output_path,
        "-outfmt", outfmt_fields
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running BLASTP: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    
    # Parse results
    results = []
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 12:
                        results.append({
                            'query_id': parts[0],
                            'subject_id': parts[1], 
                            'pident': float(parts[2]),
                            'length': int(parts[3]),
                            'mismatch': int(parts[4]),
                            'gapopen': int(parts[5]),
                            'qstart': int(parts[6]),
                            'qend': int(parts[7]),
                            'sstart': int(parts[8]),
                            'send': int(parts[9]),
                            'evalue': float(parts[10]),
                            'bitscore': float(parts[11])
                        })
    
    # Clean up temporary files
    os.unlink(query_path)
    os.unlink(output_path)
    
    return results

def merge_overlapping_ranges(ranges, max_gap=5):
    """Merge overlapping or nearby ranges"""
    if not ranges:
        return []
    
    # Sort ranges by start position
    ranges = sorted(ranges, key=lambda x: x[0])
    merged = [ranges[0]]
    
    for current in ranges[1:]:
        last = merged[-1]
        # If ranges overlap or are close (within max_gap), merge them
        if current[0] <= last[1] + max_gap:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged

def predict_fragment_locations(test_protein_sequence, fragment_sequences, 
                             identity_threshold=30.0, evalue_threshold=1e-3,
                             coverage_threshold=0.5):
    """Predict fragment locations in target protein using BLAST alignment"""
    
    if not fragment_sequences:
        return []
    
    # Create temporary database for the target protein
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "target_protein")
        
        # Create database with target protein
        create_blast_database([("target_protein", test_protein_sequence)], db_path)
        
        # Run BLAST with fragment sequences as queries
        query_sequences = [(f"frag_{i}", seq) for i, seq in enumerate(fragment_sequences)]
        blast_results = run_blastp_alignment(query_sequences, db_path, evalue_threshold=evalue_threshold*100)
    
    # Filter and process results
    predicted_ranges = []
    for result in blast_results:
        # Apply thresholds
        if (result['pident'] >= identity_threshold and 
            result['evalue'] <= evalue_threshold):
            
            # Check coverage threshold (Optional)
            # query_coverage = result['length'] / len(fragment_sequences[int(result['query_id'].split('_')[1])])
            # if query_coverage >= coverage_threshold:
            #     # Add predicted range (convert to 0-based indexing)
            #     predicted_ranges.append((result['sstart'] - 1, result['send']))

            predicted_ranges.append((result['sstart'] - 1, result['send']))

    # Merge overlapping predictions
    merged_ranges = merge_overlapping_ranges(predicted_ranges)
    
    return merged_ranges

def process_single_localization(test_data, train_category_fragments, 
                               identity_threshold=30.0, evalue_threshold=1e-3):
    """Process single localization task - find specific categories in test proteins"""
    results = []
    
    for protein in tqdm(test_data, desc="Processing proteins for single localization"):
        uid = protein['uid']
        sequence = protein['sequence']
        
        # Process each fragment type in the test protein
        for fragment_group in protein['fragments']:
            target_category = fragment_group['category']
            true_ranges = []
            
            # Get ground truth ranges
            for frag in fragment_group['frags']:
                true_ranges.append((frag['start_position'], frag['end_position'] + 1))
            
            # Get training fragments of the same category
            if target_category in train_category_fragments:
                train_fragments = train_category_fragments[target_category]
                fragment_sequences = [frag['sequence'] for frag in train_fragments]
                
                # Predict locations
                predicted_ranges = predict_fragment_locations(
                    sequence, fragment_sequences,
                    identity_threshold=identity_threshold,
                    evalue_threshold=evalue_threshold
                )
            else:
                predicted_ranges = []
            
            # Record result
            results.append({
                'uid': uid,
                'task_type': 'single',
                'target_category': target_category,
                'true_ranges': true_ranges,
                'predicted_ranges': predicted_ranges,
                'sequence_length': len(sequence)
            })
    
    return results

def process_multiple_localization(test_data, train_category_fragments,
                                identity_threshold=30.0, evalue_threshold=1e-3):
    """Process multiple localization task - find all fragments of each type"""
    results = []
    
    # Get all categories from training data
    all_categories = list(train_category_fragments.keys())
    
    for protein in tqdm(test_data, desc="Processing proteins for multiple localization"):
        uid = protein['uid']
        sequence = protein['sequence']
        
        # Build ground truth map
        true_category_ranges = defaultdict(list)
        for fragment_group in protein['fragments']:
            category = fragment_group['category']
            for frag in fragment_group['frags']:
                true_category_ranges[category].append((frag['start_position'], frag['end_position'] + 1))
        
        # For each category, predict locations
        for category in all_categories:
            if category in train_category_fragments:
                train_fragments = train_category_fragments[category]
                fragment_sequences = [frag['sequence'] for frag in train_fragments]
                
                # Predict locations
                predicted_ranges = predict_fragment_locations(
                    sequence, fragment_sequences,
                    identity_threshold=identity_threshold,
                    evalue_threshold=evalue_threshold
                )
            else:
                predicted_ranges = []
            
            # Record result
            results.append({
                'uid': uid,
                'task_type': 'multiple',
                'target_category': category,
                'true_ranges': true_category_ranges[category],
                'predicted_ranges': predicted_ranges,
                'sequence_length': len(sequence)
            })
    
    return results

def save_results_to_csv(results, output_path):
    """Save results to CSV file"""
    csv_data = []
    
    for result in results:
        # Convert ranges to string format for CSV
        true_ranges_str = ';'.join([f"{start}-{end}" for start, end in result['true_ranges']])
        pred_ranges_str = ';'.join([f"{start}-{end}" for start, end in result['predicted_ranges']])
        
        csv_data.append({
            'uid': result['uid'],
            'task_type': result['task_type'],
            'category': result['target_category'],
            'true_ranges': true_ranges_str,
            'predicted_ranges': pred_ranges_str,
            'sequence_length': result['sequence_length'],
            'num_true_fragments': len(result['true_ranges']),
            'num_predicted_fragments': len(result['predicted_ranges'])
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="BLAST-based protein fragment localization")
    parser.add_argument("--dataset", type=str, default="Act",
                       choices=['Act', 'BindI', 'Dom', 'Evo', 'Motif'],
                       help="Dataset name (Act/BindI/Dom/Evo/Motif)")
    parser.add_argument("--task", type=str, default='single',
                       choices=['single', 'multiple', 'both'],
                       help="Task type: single localization, multiple localization, or both")
    parser.add_argument("--output_dir", type=str, default="baselines/blast_cls_results",
                       help="Output directory for results")
    parser.add_argument("--identity_threshold", type=float, default=30.0,
                       help="Minimum identity percentage for BLAST hits")
    parser.add_argument("--evalue_threshold", type=float, default=1e-3,
                       help="Maximum E-value for BLAST hits")
    parser.add_argument("--coverage_threshold", type=float, default=0.5,
                       help="Minimum coverage threshold for fragment alignment")
    
    args = parser.parse_args()
    
    print(f"[*] Starting BLAST-based fragment localization for VenusX_{args.dataset}")
    print(f"[*] Task type: {args.task}")
    print(f"[*] Identity threshold: {args.identity_threshold}%")
    print(f"[*] E-value threshold: {args.evalue_threshold}")
    print(f"[*] Coverage threshold: {args.coverage_threshold}")
    
    # Create output directory
    dataset_out_dir = os.path.join(args.output_dir, f"VenusX_{args.dataset}")
    os.makedirs(dataset_out_dir, exist_ok=True)
    
    # Load datasets
    print("[1] Loading datasets...")
    train_data = load_venusx_dataset(args.dataset, "train")
    test_data = load_venusx_dataset(args.dataset, "test")
    
    # Extract training fragments by category
    print("[2] Organizing training fragments by category...")
    train_category_fragments = extract_category_fragments(train_data)
    print(f"Found {len(train_category_fragments)} categories in training data:")
    for category, fragments in train_category_fragments.items():
        print(f"  - {category}: {len(fragments)} fragments")
    
    # Process tasks
    all_results = []
    
    if args.task in ['single', 'both']:
        print("[3a] Processing single localization task...")
        single_results = process_single_localization(
            test_data, train_category_fragments,
            identity_threshold=args.identity_threshold,
            evalue_threshold=args.evalue_threshold
        )
        all_results.extend(single_results)
        
        # Save single localization results
        single_csv_path = os.path.join(dataset_out_dir, "single_localization_results.csv")
        save_results_to_csv(single_results, single_csv_path)
    
    if args.task in ['multiple', 'both']:
        print("[3b] Processing multiple localization task...")
        multiple_results = process_multiple_localization(
            test_data, train_category_fragments,
            identity_threshold=args.identity_threshold,
            evalue_threshold=args.evalue_threshold
        )
        all_results.extend(multiple_results)
        
        # Save multiple localization results
        multiple_csv_path = os.path.join(dataset_out_dir, "multiple_localization_results.csv")
        save_results_to_csv(multiple_results, multiple_csv_path)
    
    # # Save combined results
    # if args.task == 'both':
    #     combined_csv_path = os.path.join(dataset_out_dir, "combined_localization_results.csv")
    #     save_results_to_csv(all_results, combined_csv_path)
    
    print(f"Results saved in: {dataset_out_dir}")

if __name__ == "__main__":
    main()