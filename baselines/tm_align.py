import sys
sys.path.append(".")
import os
import subprocess
import pandas as pd
import argparse
import json
from tqdm import tqdm

def load_venusx_dataset(dataset_name, split):
    """Load VenusX dataset from JSON file"""
    dataset_path = f"data/VenusX_{dataset_name}/{split}.json"
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
                end_pos = frag['end_position'] +  1
                frag_id = f"{uid}_{interpro_id}_{i}"
                
                fragment_info[frag_id] = {
                    'interpro_id': interpro_id,
                    'sequence': frag['sequence'],
                    'start_position': start_pos,
                    'end_position': end_pos,
                    'uid': uid
                }
    
    return fragment_info

def calculate_align_info(predicted_pdb_path, reference_pdb_path):
    """Calculate TM-align structural alignment between two PDB files"""
    cmd = f"TMalign {predicted_pdb_path} {reference_pdb_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stderr:
        print("Error in TMalign:", result.stderr)
        return None

    lines = result.stdout.split("\n")
    tm_score_1, tm_score_2, tm_score = None, None, None
    aligned_length, rmsd, seq_identity = None, None, None
    
    for line in lines:
        if "Aligned length" in line:
            aligned_length = int(line.split(",")[0].split("=")[1].strip())
            rmsd = float(line.split(",")[1].split("=")[1].strip())
            seq_identity = float(line.split(",")[2].split("=")[-1].strip())
        if "TM-score" in line and "Chain_1" in line:
            tm_score_1 = float(line.split(" ")[1].strip())
        if "TM-score" in line and "Chain_2" in line:
            tm_score_2 = float(line.split(" ")[1].strip())

    if tm_score_1 is not None and tm_score_2 is not None:
        tm_score = (tm_score_1 + tm_score_2) / 2
    
    align_info = {
        "aligned_length": aligned_length,
        "rmsd": rmsd,
        "seq_identity": seq_identity,
        "tm_score": tm_score,
        "tm_score_1": tm_score_1,
        "tm_score_2": tm_score_2
    }
    return align_info

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

def find_best_structural_match(test_frag_id, test_fragment_info, train_fragment_info, pdb_base_path, dataset_name, corrections):
    """Find the best structural match for a test fragment from train fragments"""
    best_match = None
    best_tm_score = -1
    
    # Get test PDB file path using correction mapping
    test_pdb_path = find_structure_file(test_fragment_info, test_frag_id, pdb_base_path, dataset_name, corrections)
    
    assert test_pdb_path is not None, f"Test PDB file not found for fragment: {test_frag_id}"
    
    for train_frag_id, train_info in train_fragment_info.items():
        # Get train PDB file path using correction mapping
        train_pdb_path = find_structure_file(train_fragment_info, train_frag_id, pdb_base_path, dataset_name, corrections)
        
        assert train_pdb_path is not None, f"Train PDB file not found for fragment: {train_frag_id}"
            
        align_info = calculate_align_info(test_pdb_path, train_pdb_path)
        
        if align_info and align_info['tm_score'] is not None:
            if align_info['tm_score'] > best_tm_score:
                best_tm_score = align_info['tm_score']
                best_match = {
                    'target_fragment_id': train_frag_id,
                    'predicted_interpro_id': train_info['interpro_id'],
                    'tm_score': align_info['tm_score'],
                    'tm_score_1': align_info['tm_score_1'],
                    'tm_score_2': align_info['tm_score_2'],
                    'aligned_length': align_info['aligned_length'],
                    'rmsd': align_info['rmsd'],
                    'seq_identity': align_info['seq_identity']
                }
    
    return best_match

def main(dataset_name, pdb_base_path, out_dir, correction_file):
    """Main function for VenusX TM-align analysis"""
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
    
    # Extract fragment information
    train_fragment_info = extract_fragments_info(train_data)
    test_fragment_info = extract_fragments_info(test_data)
    
    print(f"[Info] Train fragments: {len(train_fragment_info)}")
    print(f"[Info] Test fragments: {len(test_fragment_info)}")
    
    # Step 2: For each test fragment, find best structural match in train set
    print("[2] Finding best structural matches using TM-align...")
    predictions = []
    
    for test_frag_id, test_info in tqdm(test_fragment_info.items(), desc="Processing test fragments"):
        best_match = find_best_structural_match(
            test_frag_id, 
            test_fragment_info,
            train_fragment_info, 
            pdb_base_path, 
            dataset_name,
            corrections
        )
        
        if best_match:
            predictions.append({
                'query_fragment': test_frag_id,
                'true_interpro_id': test_info['interpro_id'],
                'predicted_interpro_id': best_match['predicted_interpro_id'],
                'target_fragment': best_match['target_fragment_id'],
                'tm_score': best_match['tm_score'],
                'tm_score_1': best_match['tm_score_1'],
                'tm_score_2': best_match['tm_score_2'],
                'aligned_length': best_match['aligned_length'],
                'rmsd': best_match['rmsd'],
                'seq_identity': best_match['seq_identity']
            })
        # else:
        #     raise NotImplementedError
    
    # Step 3: Save results to CSV
    print("[3] Saving results to CSV...")
    results_df = pd.DataFrame(predictions)
    csv_output = os.path.join(dataset_out_dir, "tmalign_predictions.csv")
    results_df.to_csv(csv_output, index=False)
    
    print(f"[✓] Results saved to {csv_output}")
    print(f"[✓] Total predictions: {len(predictions)}")
    
    # Calculate accuracy
    if len(predictions) > 0:
        correct = sum(1 for p in predictions if p['true_interpro_id'] == p['predicted_interpro_id'])
        accuracy = correct / len(predictions)
        print(f"[✓] Accuracy: {accuracy:.4f} ({correct}/{len(predictions)})")
    
    return csv_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TM-align analysis for VenusX protein fragments")
    parser.add_argument("--dataset", default='Act', choices=["Act", "BindI", "Dom", "Evo", "Motif"],
                       help="VenusX dataset to analyze")
    parser.add_argument("--pdb_base_path", type=str, default="/home/lfj/database/VenusX_AFDB",
                       help="Base path to PDB files organized as: pdb_base_path/VenusX_{dataset}_AlphaFold2_PDB/alphafold2_pdb_fragment/{frag_id}.pdb")
    parser.add_argument("--out_dir", type=str, default="baselines/tmalign_results", help="Output directory")
    parser.add_argument("--correction_file", type=str, 
                       default="/home/lfj/database/VenusX_AFDB/pdb_fragment_name_corrections.json",
                       help="Path to PDB filename correction mapping")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    main(args.dataset, args.pdb_base_path, args.out_dir, args.correction_file)
    