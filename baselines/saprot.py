import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import pandas as pd
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def load_venusx_dataset(dataset_name, split):
    """Load VenusX dataset from JSON file"""
    dataset_path = os.path.join(project_root, "data", f"VenusX_{dataset_name}", f"{split}.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    return data

def extract_fragments_with_labels(data):
    """Extract fragment sequences with their interpro_id labels"""
    fragments = []
    labels = []
    fragment_ids = []
    
    for protein in data:
        uid = protein['uid']
        for fragment_group in protein['fragments']:
            interpro_id = fragment_group['interpro_id']
            for i, frag in enumerate(fragment_group['frags']):
                frag_id = f"{uid}_{interpro_id}_{i}"
                frag_seq = frag['sequence']
                fragments.append(frag_seq)
                labels.append(interpro_id)
                fragment_ids.append(frag_id)
    
    return fragments, labels, fragment_ids

def get_pdb_filename(fragment_id, fragment_info):
    """Generate PDB filename based on fragment information"""
    info = fragment_info[fragment_id]
    interpro_id = info['interpro_id']
    uid = info['uid']
    start_pos = info['start_position']
    end_pos = info['end_position']
    return f"{interpro_id}_{uid}_{start_pos}-{end_pos}.pdb"

def extract_fragments_info(data):
    """Extract fragment information and return fragment to interpro_id mapping"""
    fragment_info = {}
    
    for protein in data:
        uid = protein['uid']
        for fragment_group in protein['fragments']:
            interpro_id = fragment_group['interpro_id']
            for i, frag in enumerate(fragment_group['frags']):
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

def find_structure_file(fragment_id, fragment_info, pdb_base_path, dataset_name, corrections):
    """Find structure file for a fragment, handling missing files with corrections"""
    info = fragment_info[fragment_id]
    interpro_id = info['interpro_id']
    uid = info['uid']
    start_pos = info['start_position']
    end_pos = info['end_position']
    
    # Generate expected PDB filename
    expected_filename = f"{interpro_id}_{uid}_{start_pos}-{end_pos}.pdb"
    
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
        else:
            raise FileNotFoundError(f"Corrected PDB file not found: {corrected_path}")
        
    return None

def load_struc_seq(path, foldseek_path, chains=["A"], process_id=0):
    """Extract structure sequence using Foldseek"""
    if not os.path.exists(foldseek_path):
        raise FileNotFoundError(f"Foldseek not found: {foldseek_path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDB file not found: {path}")

    tmp_save_path = f"get_struc_seq_{process_id}.tsv"
    cmd = f"{foldseek_path} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)

    seq_dict = {}
    name = os.path.basename(path)
    
    try:
        with open(tmp_save_path, "r") as r:
            for line in r:
                desc, seq, struc_seq = line.split("\t")[:3]
                name_chain = desc.split(" ")[0]
                chain = name_chain.replace(name, "").split("_")[-1]
                if chains is None or chain in chains:
                    if chain not in seq_dict:
                        # Combine amino acid sequence with structure tokens (lowercase)
                        combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                        seq_dict[chain] = [seq, struc_seq, combined_seq]
    except FileNotFoundError:
        print(f"Warning: Could not process structure file {path}")
        return {}

    # Clean up temporary files
    if os.path.exists(tmp_save_path):
        os.remove(tmp_save_path)
    if os.path.exists(tmp_save_path + ".dbtype"):
        os.remove(tmp_save_path + ".dbtype")
    
    return seq_dict

def load_saprot_model(model_path, device):
    """Load SaProt model and tokenizer"""
    print(f"Loading SaProt model from {model_path}...")
    
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmModel.from_pretrained(model_path)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    return model, tokenizer

def encode_sequences_batch(model, tokenizer, sequences, device, batch_size=16, foldseek_path=None):
    """Encode protein sequences using SaProt model in batches"""
    print(f"Encoding {len(sequences)} sequences in batches of {batch_size}...")
    
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_data = sequences[i:i + batch_size]
            batch_seqs = []
            
            for item in batch_data:
                if isinstance(item, dict) and 'combined_sequence' in item:
                    # Use combined sequence if available
                    batch_seqs.append(item['combined_sequence'])
                elif isinstance(item, dict) and 'sequence' in item:
                    # Use regular sequence if combined not available
                    batch_seqs.append(item['sequence'])
                else:
                    # Direct sequence string
                    batch_seqs.append(item)
            
            # Tokenize batch
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings
            outputs = model(**inputs)
            sequence_embeddings = outputs.last_hidden_state
            
            # Pool embeddings (mean pooling over sequence length, excluding special tokens)
            attention_mask = inputs['attention_mask']
            batch_embeddings = []
            
            for j in range(len(batch_seqs)):
                # Get valid token positions (excluding padding)
                valid_mask = attention_mask[j] == 1
                valid_embeddings = sequence_embeddings[j][valid_mask]
                
                # Skip first and last tokens (CLS and EOS)
                if len(valid_embeddings) > 2:
                    valid_embeddings = valid_embeddings[1:-1]
                
                # Mean pooling
                if len(valid_embeddings) > 0:
                    pooled_embedding = valid_embeddings.mean(dim=0)
                else:
                    raise ValueError("No valid tokens found for pooling.")
                
                batch_embeddings.append(pooled_embedding.cpu().numpy())
            
            embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

def prepare_sequences_with_structure(fragments, fragment_ids, fragment_info, pdb_base_path, dataset_name, corrections, foldseek_path):
    """Prepare sequences with structure information when available"""
    processed_sequences = []
    valid_fragment_ids = []
    
    print(f"Processing {len(fragments)} fragments with structure information...")
    
    for i, (frag_seq, frag_id) in enumerate(zip(fragments, fragment_ids)):
        # Try to find structure file
        structure_path = find_structure_file(frag_id, fragment_info, pdb_base_path, dataset_name, corrections)
        
        assert structure_path is not None, f"Structure file not found for fragment {frag_id}"

        # Extract structure sequence using Foldseek
        seq_dict = load_struc_seq(structure_path, foldseek_path, chains=["A"], process_id=i % 100)
        
        assert "A" in seq_dict and len(seq_dict["A"]) >= 3, f"Chain A not found in structure for fragment {frag_id}"

        # Use combined sequence (sequence + structure)
        combined_seq = seq_dict["A"][2]
        processed_sequences.append({
            'sequence': frag_seq,
            'combined_sequence': combined_seq,
            'has_structure': True
        })
        valid_fragment_ids.append(frag_id)
            
    structure_count = sum(1 for seq in processed_sequences if seq['has_structure'])
    print(f"Successfully loaded structure for {structure_count}/{len(processed_sequences)} fragments")
    
    return processed_sequences, valid_fragment_ids

def find_most_similar(query_embeddings, train_embeddings, train_labels, train_ids):
    """Find most similar training example for each query using cosine similarity"""
    print("Computing similarity matrix...")
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(query_embeddings, train_embeddings)
    
    predictions = []
    
    for i, query_similarities in enumerate(tqdm(similarity_matrix, desc="Finding best matches")):
        # Find index of most similar training example
        best_match_idx = np.argmax(query_similarities)
        best_similarity = query_similarities[best_match_idx]
        
        # Get prediction from most similar training example
        predicted_interpro = train_labels[best_match_idx]
        matched_train_id = train_ids[best_match_idx]
        
        predictions.append({
            'predicted_interpro_id': predicted_interpro,
            'matched_train_id': matched_train_id,
            'similarity_score': float(best_similarity)
        })
    
    return predictions

def save_embeddings_cache(embeddings, cache_path):
    """Save embeddings to cache file"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"Saved embeddings cache to {cache_path}")

def load_embeddings_cache(cache_path):
    """Load embeddings from cache file"""
    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        print(f"Loaded embeddings cache from {cache_path}")
        return embeddings
    return None

def get_model_name(model_path):
    """Extract model name from model path for output directory"""
    path_parts = model_path.rstrip('/').split('/')
    model_name = path_parts[-1]
    return model_name

def main(dataset_name, model_path, batch_size, out_dir, use_cuda, pdb_base_path, foldseek_path, correction_file):
    """Main function for SaProt-based VenusX analysis"""
    model_name = get_model_name(model_path)
    print(f"[*] Processing VenusX_{dataset_name} dataset with SaProt ({model_name})...")
    
    # Setup device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory with model-specific subdirectory
    dataset_out_dir = os.path.join(out_dir, model_name, f"VenusX_{dataset_name}")
    os.makedirs(dataset_out_dir, exist_ok=True)
    
    # Load filename corrections
    corrections = load_name_corrections(correction_file)
    
    # Step 1: Load datasets and extract fragments
    print("[1] Loading datasets and extracting fragments...")
    
    train_data = load_venusx_dataset(dataset_name, "train")
    test_data = load_venusx_dataset(dataset_name, "test")
    
    train_sequences, train_labels, train_ids = extract_fragments_with_labels(train_data)
    test_sequences, test_labels, test_ids = extract_fragments_with_labels(test_data)
    
    # Extract fragment info for structure file lookup
    train_fragment_info = extract_fragments_info(train_data)
    test_fragment_info = extract_fragments_info(test_data)
    
    print(f"Train fragments: {len(train_sequences)}")
    print(f"Test fragments: {len(test_sequences)}")
    
    # Step 2: Load SaProt model
    print("[2] Loading SaProt model...")
    model, tokenizer = load_saprot_model(model_path, device)
    
    # Step 3: Prepare training sequences with structure information
    print("[3] Preparing training sequences with structure information...")
    train_processed_sequences, valid_train_ids = prepare_sequences_with_structure(
        train_sequences, train_ids, train_fragment_info, pdb_base_path, dataset_name, corrections, foldseek_path
    )
    
    # Update labels to match valid sequences
    valid_train_labels = [train_labels[train_ids.index(frag_id)] for frag_id in valid_train_ids]
    
    # Step 4: Encode training sequences (with caching)
    print("[4] Encoding training sequences...")
    train_cache_path = os.path.join(dataset_out_dir, "train_embeddings.npy")
    train_embeddings = load_embeddings_cache(train_cache_path)
    
    if train_embeddings is None:
        train_embeddings = encode_sequences_batch(
            model, tokenizer, train_processed_sequences, device, batch_size, foldseek_path
        )
        save_embeddings_cache(train_embeddings, train_cache_path)
    
    # Step 5: Prepare test sequences with structure information
    print("[5] Preparing test sequences with structure information...")
    test_processed_sequences, valid_test_ids = prepare_sequences_with_structure(
        test_sequences, test_ids, test_fragment_info, pdb_base_path, dataset_name, corrections, foldseek_path
    )
    
    # Update labels to match valid sequences
    valid_test_labels = [test_labels[test_ids.index(frag_id)] for frag_id in valid_test_ids]
    
    # Step 6: Encode test sequences (with caching)
    print("[6] Encoding test sequences...")
    test_cache_path = os.path.join(dataset_out_dir, "test_embeddings.npy")
    test_embeddings = load_embeddings_cache(test_cache_path)
    
    if test_embeddings is None:
        test_embeddings = encode_sequences_batch(
            model, tokenizer, test_processed_sequences, device, batch_size, foldseek_path
        )
        save_embeddings_cache(test_embeddings, test_cache_path)
    
    # Step 7: Find most similar training examples for each test sequence
    print("[7] Finding most similar training examples...")
    predictions = find_most_similar(test_embeddings, train_embeddings, valid_train_labels, valid_train_ids)
    
    # Step 8: Create results dataframe
    print("[8] Creating results dataframe...")
    results = []
    for i, (test_id, true_label, pred_info) in enumerate(zip(valid_test_ids, valid_test_labels, predictions)):
        results.append({
            'query_sequence': test_id,
            'true_interpro_id': true_label,
            'predicted_interpro_id': pred_info['predicted_interpro_id'],
            'matched_train_sequence': pred_info['matched_train_id'],
            'similarity_score': pred_info['similarity_score'],
            'has_structure': test_processed_sequences[i]['has_structure']
        })
    
    # Step 9: Save results to CSV
    print("[9] Saving results to CSV...")
    results_df = pd.DataFrame(results)
    csv_output = os.path.join(dataset_out_dir, "saprot_predictions.csv")
    results_df.to_csv(csv_output, index=False)
    
    print(f"Results saved to {csv_output}")
    print(f"Total predictions: {len(results)}")
    
    # Calculate accuracy
    if len(results) > 0:
        correct = sum(1 for r in results if r['true_interpro_id'] == r['predicted_interpro_id'])
        accuracy = correct / len(results)
        print(f"Accuracy: {accuracy:.4f} ({correct}/{len(results)})")
        
        # Additional statistics
        avg_similarity = np.mean([r['similarity_score'] for r in results])
        structure_count = sum(1 for r in results if r['has_structure'])
        print(f"Average similarity score: {avg_similarity:.4f}")
        print(f"Fragments with structure: {structure_count}/{len(results)} ({structure_count/len(results)*100:.1f}%)")
    
    # Save additional metadata
    metadata = {
        'dataset_name': dataset_name,
        'model_path': model_path,
        'batch_size': batch_size,
        'device': str(device),
        'num_train_fragments': len(valid_train_ids),
        'num_test_fragments': len(valid_test_ids),
        'accuracy': accuracy if len(results) > 0 else 0.0,
        'avg_similarity': avg_similarity if len(results) > 0 else 0.0,
        'structure_coverage': structure_count / len(results) if len(results) > 0 else 0.0
    }
    
    metadata_file = os.path.join(dataset_out_dir, "saprot_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_file}")
    
    return csv_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SaProt analysis for VenusX protein fragments")
    parser.add_argument("--dataset", default='Act', choices=["Act", "BindI", "Dom", "Evo", "Motif"],
                       help="VenusX dataset to analyze")
    parser.add_argument("--model_path", type=str, 
                       default="/home/lfj/projects_dir/pretrained_model/SaProt_35M_AF2/",
                       help="Path to SaProt model")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for encoding sequences")
    parser.add_argument("--out_dir", type=str, default=os.path.join(project_root, "baselines", "saprot_results"), 
                       help="Output directory")
    parser.add_argument("--cpu", action="store_true", 
                       help="Force CPU usage (default: use CUDA if available)")
    parser.add_argument("--pdb_base_path", type=str, default="/home/lfj/database/VenusX_AFDB",
                       help="Base path to PDB structure files")
    parser.add_argument("--foldseek_path", type=str, default="/home/lfj/anaconda3/envs/fragllm/bin/foldseek",
                       help="Path to Foldseek executable")
    parser.add_argument("--correction_file", type=str, 
                       default="/home/lfj/database/VenusX_AFDB/pdb_fragment_name_corrections.json",
                       help="Path to PDB filename correction mapping")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    main(args.dataset, args.model_path, args.batch_size, args.out_dir, not args.cpu, 
         args.pdb_base_path, args.foldseek_path, args.correction_file)