"""Baseline BLAST implementation for reffering task."""

import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import subprocess
import pandas as pd
import argparse
import json
from tqdm import tqdm

def load_venusx_dataset(dataset_name, split):
    """Load VenusX dataset from JSON file"""
    dataset_path = os.path.join(project_root, "data", f"VenusX_{dataset_name}", f"{split}.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    return data

def extract_fragments_to_fasta(data, output_fasta, interpro_index_file):
    """Extract fragment sequences and create FASTA file with interpro_id indexing"""
    fragments = []
    interpro_index = {}
    
    for protein in data:
        uid = protein['uid']
        for fragment_group in protein['fragments']:
            interpro_id = fragment_group['interpro_id']
            for i, frag in enumerate(fragment_group['frags']):
                frag_id = f"{uid}_{interpro_id}_{i}"
                frag_seq = frag['sequence']
                fragments.append((frag_id, frag_seq))
                interpro_index[frag_id] = interpro_id
    
    # Write FASTA file
    with open(output_fasta, 'w') as f:
        for frag_id, seq in fragments:
            f.write(f">{frag_id}\n{seq}\n")
    
    # Save interpro index
    with open(interpro_index_file, 'w') as f:
        json.dump(interpro_index, f, indent=2)
    
    print(f"Extracted {len(fragments)} fragments to {output_fasta}")
    print(f"Saved interpro index to {interpro_index_file}")
    
    return fragments, interpro_index

def run_makeblastdb(fasta_file, db_name):
    """Create BLAST database from FASTA file"""
    cmd = ["makeblastdb", "-in", fasta_file, "-dbtype", "prot", "-out", db_name]
    subprocess.run(cmd, check=True)

def run_blastp_query(query_fasta, db_name, out_file, num_threads):
    """Run BLASTP search with query sequences against database"""
    outfmt_fields = "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore"
    cmd = [
        "blastp",
        "-query", query_fasta,
        "-db", db_name,
        "-evalue", "1000000",
        "-word_size", "2",
        "-max_target_seqs", "1",  # Only get top hit
        "-seg", "no",
        "-out", out_file,
        "-outfmt", outfmt_fields,
        "-num_threads", str(num_threads)
    ]
    subprocess.run(cmd, check=True)

def parse_blast_results(blast_output, train_interpro_index, test_interpro_index):
    """Parse BLAST results and create predictions"""
    predictions = []
    
    # First pass: collect best hits for each query (handle multiple hits per query)
    query_results = {}
    with open(blast_output, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 12:
                    query_id = parts[0]
                    target_id = parts[1]
                    evalue = float(parts[10])
                    bitscore = float(parts[11])
                    
                    # Keep only the best hit (lowest evalue, highest bitscore) for each query
                    if query_id not in query_results or evalue < query_results[query_id]['evalue']:
                        query_results[query_id] = {
                            'target_id': target_id,
                            'evalue': evalue,
                            'bitscore': bitscore
                        }
                    elif evalue == query_results[query_id]['evalue'] and bitscore > query_results[query_id]['bitscore']:
                        # If evalue is same, prefer higher bitscore
                        query_results[query_id] = {
                            'target_id': target_id,
                            'evalue': evalue,
                            'bitscore': bitscore
                        }
    
    # Second pass: create predictions for all test sequences (handle missing queries)
    for query_id in test_interpro_index.keys():
        true_interpro = test_interpro_index[query_id]
        
        if query_id in query_results:
            # Query has a BLAST hit
            result = query_results[query_id]
            target_id = result['target_id']
            evalue = result['evalue']
            pred_interpro = train_interpro_index.get(target_id, "Unknown")
        else:
            # Query has no BLAST hit - assign default values
            target_id = "No_Hit"
            evalue = float('inf')
            pred_interpro = "No_Prediction"
            
        
        predictions.append({
            'query_sequence': query_id,
            'true_interpro_id': true_interpro,
            'predicted_interpro_id': pred_interpro,
            'target_sequence': target_id,
            'evalue': evalue
        })
    
    return predictions

def main(dataset_name, num_threads, out_dir):
    """Main function for VenusX BLAST analysis"""
    print(f"[*] Processing VenusX_{dataset_name} dataset...")
    
    # Create output directory
    dataset_out_dir = os.path.join(out_dir, f"VenusX_{dataset_name}")
    os.makedirs(dataset_out_dir, exist_ok=True)
    
    # Step 1: Load datasets and extract fragments
    print("[1] Loading datasets and extracting fragments...")
    
    train_data = load_venusx_dataset(dataset_name, "train")
    test_data = load_venusx_dataset(dataset_name, "test")
    
    # Extract train fragments
    train_fasta = os.path.join(dataset_out_dir, "train_fragments.fasta")
    train_index_file = os.path.join(dataset_out_dir, "train_interpro_index.json")
    train_fragments, train_interpro_index = extract_fragments_to_fasta(
        train_data, train_fasta, train_index_file
    )
    
    # Extract test fragments
    test_fasta = os.path.join(dataset_out_dir, "test_fragments.fasta")
    test_index_file = os.path.join(dataset_out_dir, "test_interpro_index.json")
    test_fragments, test_interpro_index = extract_fragments_to_fasta(
        test_data, test_fasta, test_index_file
    )
    
    # Step 2: Build BLAST database from train fragments
    print("[2] Building BLAST database from train fragments...")
    db_name = os.path.join(dataset_out_dir, "train_db")
    run_makeblastdb(train_fasta, db_name)
    
    # Step 3: Search test fragments against train database
    print("[3] Searching test fragments against train database...")
    blast_results = os.path.join(dataset_out_dir, "blast_results.txt")
    
    if not os.path.exists(blast_results):
        run_blastp_query(test_fasta, db_name, blast_results, num_threads)
    else:
        print(f"[Info] BLAST results already exist at {blast_results}")
    
    # Step 4: Parse results and generate predictions
    print("[4] Parsing BLAST results and generating predictions...")
    predictions = parse_blast_results(blast_results, train_interpro_index, test_interpro_index)
    
    # Step 5: Save results to CSV
    print("[5] Saving results to CSV...")
    results_df = pd.DataFrame(predictions)
    csv_output = os.path.join(dataset_out_dir, "blast_predictions.csv")
    results_df.to_csv(csv_output, index=False)
    
    print(f"[✓] Results saved to {csv_output}")
    print(f"[✓] Total predictions: {len(predictions)}")
    
    # Calculate accuracy (simply)
    if len(predictions) > 0:
        correct = sum(1 for p in predictions if p['true_interpro_id'] == p['predicted_interpro_id'])
        accuracy = correct / len(predictions)
        print(f"[✓] Accuracy: {accuracy:.4f} ({correct}/{len(predictions)})")
    
    # Clean up database files
    os.system(f"rm -rf {db_name}*")
    
    return csv_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLAST analysis for VenusX protein fragments")
    parser.add_argument("--dataset", default="Act",choices=["Act", "BindI", "Dom", "Evo", "Motif"], required=True,
                       help="VenusX dataset to analyze")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for BLAST")
    parser.add_argument("--out_dir", type=str, default=os.path.join(project_root, "baselines", "blast_results"), help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    main(args.dataset, args.num_threads, args.out_dir)