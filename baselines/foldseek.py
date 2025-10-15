"""Baseline foldseek implementation for reffering task."""

import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import argparse
import subprocess
import pandas as pd
import torch
import numpy as np
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

def extract_fragments_info(data):
    """Extract fragment information and return fragment to interpro_id mapping"""
    fragment_info = {}
    
    for protein in data:
        uid = protein['uid']
        for fragment_group in protein['fragments']:
            interpro_id = fragment_group['interpro_id']
            for i, frag in enumerate(fragment_group['frags']):
                # Create fragment ID using start and end positions
                start_pos = frag['start_position'] + 1
                end_pos = frag['end_position'] + 1
                frag_id = f"{uid}_{interpro_id}_{i}"
                
                fragment_info[frag_id] = {
                    'interpro_id': interpro_id,
                    'sequence': frag['sequence'],
                    'start_position': start_pos,
                    'end_position': end_pos,
                    'uid': uid
                }
    
    return fragment_info

def load_name_corrections(correction_file):
    """Load PDB filename corrections from JSON file"""
    if os.path.exists(correction_file):
        with open(correction_file, 'r') as f:
            corrections = json.load(f)
        print(f"Loaded {len(corrections)} filename corrections")
        return corrections
    else:
        print(f"Correction file not found: {correction_file}")
        return {}

def get_pdb_filename(fragment_info, frag_id):
    """Generate PDB filename in format: {interpro_id}_{uid}_{start}-{end}.pdb"""
    info = fragment_info[frag_id]
    interpro_id = info['interpro_id']
    uid = info['uid']
    start_pos = info['start_position']
    end_pos = info['end_position']
    return f"{interpro_id}_{uid}_{start_pos}-{end_pos}.pdb"

def find_structure_file(fragment_info, frag_id, pdb_base_path, dataset_name, corrections):
    """Find structure file for a fragment, handling missing files with corrections"""
    # Generate expected PDB filename
    expected_filename = get_pdb_filename(fragment_info, frag_id)
    
    # Construct the expected path
    pdb_dir = os.path.join(pdb_base_path, f"VenusX_{dataset_name}_AlphaFold2_PDB", "alphafold2_pdb_fragment")
    expected_path = os.path.join(pdb_dir, expected_filename)
    
    # Check if the expected file exists
    if os.path.exists(expected_path):
        return expected_path
    
    # If not found, check corrections mapping
    if expected_filename in corrections:
        corrected_filename = corrections[expected_filename]
        corrected_path = os.path.join(pdb_dir, corrected_filename)
        if os.path.exists(corrected_path):
            return corrected_path
    
    return None

def create_pdb_directory_structure(fragment_info, pdb_base_path, dataset_name, split, corrections):
    """Create temporary directory structure for Foldseek with fragment PDB files"""
    temp_dir = f"temp_foldseek_{dataset_name}_{split}"
    os.makedirs(temp_dir, exist_ok=True)
    
    valid_fragments = []
    for frag_id in fragment_info.keys():
        # Find structure file using correction mapping
        pdb_source = find_structure_file(fragment_info, frag_id, pdb_base_path, dataset_name, corrections)

        assert pdb_source is not None, f"PDB file not found for fragment: {frag_id}"
        
        # Use the same fragment ID for the destination to maintain consistency
        pdb_dest = os.path.join(temp_dir, f"{frag_id}.pdb")
        if not os.path.exists(pdb_dest):
            os.symlink(os.path.abspath(pdb_source), pdb_dest)
        valid_fragments.append(frag_id)
    
    print(f"[Info] Created temp directory {temp_dir} with {len(valid_fragments)} PDB files")
    return temp_dir, valid_fragments

def run_foldseek_easysearch(query_dir, target_dir, out_prefix="aln", num_threads=8, alignment_type=2):
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        "foldseek", "easy-search",
        query_dir,
        target_dir,
        out_prefix,
        tmp_dir,
        "--exhaustive-search",
        "--max-seqs", "100000",
        "-e", "1000",
        "--min-seq-id", "0.0",
        "--alignment-type", str(alignment_type),
        "--threads", str(num_threads)
    ]

    print(f"[*] Running Foldseek easy-search with alignment_type={alignment_type}...")
    subprocess.run(cmd, check=True)
    print(f"[✓] Foldseek completed. Output: {out_prefix}")

    return out_prefix

def parse_foldseek_results(m8_file, train_fragment_info, test_fragment_info):
    """Parse Foldseek results and create predictions"""
    predictions = []
    
    # Create a dictionary for best matches per query
    best_matches = {}
    
    with open(m8_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 12:
                    query_id = parts[0]
                    target_id = parts[1]
                    evalue = float(parts[10])
                    bitscore = float(parts[11])
                    
                    # Skip self-alignments
                    if query_id == target_id:
                        continue
                    
                    # Only process if query is in test and target is in train
                    if query_id in test_fragment_info and target_id in train_fragment_info:
                        # Use bitscore for ranking (higher is better)
                        if query_id not in best_matches or bitscore > best_matches[query_id]['bitscore']:
                            best_matches[query_id] = {
                                'target_fragment_id': target_id,
                                'predicted_interpro_id': train_fragment_info[target_id]['interpro_id'],
                                'evalue': evalue,
                                'bitscore': bitscore
                            }
    
    # Convert best matches to predictions
    for query_id, test_info in test_fragment_info.items():
        if query_id in best_matches:
            match = best_matches[query_id]
            predictions.append({
                'query_fragment': query_id,
                'true_interpro_id': test_info['interpro_id'],
                'predicted_interpro_id': match['predicted_interpro_id'],
                'target_fragment': match['target_fragment_id'],
                'evalue': match['evalue'],
                'bitscore': match['bitscore']
            })
        else:
            # Query has no Foldseek match - assign default values
            predictions.append({
                'query_fragment': query_id,
                'true_interpro_id': test_info['interpro_id'],
                'predicted_interpro_id': 'No_Prediction',
                'target_fragment': 'No_Hit',
                'evalue': float('inf'),
                'bitscore': 0.0
            })
    
    return predictions

def cleanup_temp_directories(*temp_dirs):
    """Clean up temporary directories"""
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            print(f"[Info] Cleaned up temporary directory: {temp_dir}")

def main(dataset_name, pdb_base_path, out_dir, num_threads, alignment_type, correction_file):
    """Main function for VenusX Foldseek analysis"""
    print(f"[*] Processing VenusX_{dataset_name} dataset...")
    
    # Load filename corrections
    corrections = load_name_corrections(correction_file)
    
    # Create output directory
    dataset_out_dir = os.path.join(out_dir, f"VenusX_{dataset_name}")
    os.makedirs(dataset_out_dir, exist_ok=True)
    
    # Step 1: Load datasets and extract fragment information
    print("[1] Loading datasets and extracting fragment information...")
    
    train_data = load_venusx_dataset(dataset_name, "train")
    test_data = load_venusx_dataset(dataset_name, "test")
    
    train_fragment_info = extract_fragments_info(train_data)
    test_fragment_info = extract_fragments_info(test_data)
    
    print(f"[Info] Train fragments: {len(train_fragment_info)}")
    print(f"[Info] Test fragments: {len(test_fragment_info)}")
    
    # Step 2: Create temporary PDB directory structures
    print("[2] Creating temporary PDB directory structures...")
    
    train_temp_dir, valid_train_fragments = create_pdb_directory_structure(
        train_fragment_info, pdb_base_path, dataset_name, "train", corrections
    )
    test_temp_dir, valid_test_fragments = create_pdb_directory_structure(
        test_fragment_info, pdb_base_path, dataset_name, "test", corrections
    )
    
    # Filter fragment info to only include valid fragments
    train_fragment_info = {frag_id: train_fragment_info[frag_id] for frag_id in valid_train_fragments}
    test_fragment_info = {frag_id: test_fragment_info[frag_id] for frag_id in valid_test_fragments}
    
    try:
        # Step 3: Run Foldseek easy-search
        print("[3] Running Foldseek easy-search...")
        
        aln_file = os.path.join(dataset_out_dir, "foldseek_results.m8")
        
        if not os.path.exists(aln_file):
            run_foldseek_easysearch(
                query_dir=test_temp_dir,
                target_dir=train_temp_dir,
                out_prefix=aln_file,
                num_threads=num_threads,
                alignment_type=alignment_type
            )
        else:
            print(f"[Info] Foldseek results already exist at {aln_file}")
        
        # Step 4: Parse results and generate predictions
        print("[4] Parsing Foldseek results and generating predictions...")
        predictions = parse_foldseek_results(aln_file, train_fragment_info, test_fragment_info)
        
        # Step 5: Save results to CSV
        print("[5] Saving results to CSV...")
        results_df = pd.DataFrame(predictions)
        csv_output = os.path.join(dataset_out_dir, "foldseek_predictions.csv")
        results_df.to_csv(csv_output, index=False)
        
        print(f"[✓] Results saved to {csv_output}")
        print(f"[✓] Total predictions: {len(predictions)}")
        
        # Calculate accuracy
        if len(predictions) > 0:
            correct = sum(1 for p in predictions if p['true_interpro_id'] == p['predicted_interpro_id'])
            accuracy = correct / len(predictions)
            print(f"[✓] Accuracy: {accuracy:.4f} ({correct}/{len(predictions)})")
        
        return csv_output
    
    finally:
        # Clean up temporary directories
        cleanup_temp_directories(train_temp_dir, test_temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foldseek analysis for VenusX protein fragments")
    parser.add_argument("--dataset", default="Act", choices=["Act", "BindI", "Dom", "Evo", "Motif"],
                       help="VenusX dataset to analyze")
    parser.add_argument("--pdb_base_path", type=str, default="/home/lfj/database/VenusX_AFDB",
                       help="Base path to PDB files organized as: pdb_base_path/VenusX_{dataset}_AlphaFold2_PDB/alphafold2_pdb_fragment/{frag_id}.pdb")
    parser.add_argument("--out_dir", type=str, default=os.path.join(project_root, "baselines", "foldseek_results"), help="Output directory")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of threads")
    parser.add_argument("--alignment_type", type=int, choices=[0, 1, 2], default=0,
                        help="Alignment type: 0 (3Di), 1 (TMalign), 2 (3Di+AA, default)")
    parser.add_argument("--correction_file", type=str, 
                       default="/home/lfj/database/VenusX_AFDB/pdb_fragment_name_corrections.json",
                       help="Path to PDB filename correction mapping")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    main(args.dataset, args.pdb_base_path, args.out_dir, args.num_threads, args.alignment_type, args.correction_file)
