import sys
sys.path.append(".")
import os
import pandas as pd
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def load_venusx_dataset(dataset_name, split):
    """Load VenusX dataset from JSON file"""
    dataset_path = f"data/VenusX_{dataset_name}/{split}.json"
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

def load_plm_model(model_path, device):
    """Load protein language model and tokenizer"""
    print(f"Loading PLM model from {model_path}...")
    
    # Special handling for T5-based models (like ProtT5, Ankh)
    if "t5" in model_path.lower() or "prot_t5" in model_path.lower():
        from transformers import T5EncoderModel, T5Tokenizer
        print("Detected ProtT5 model, using T5EncoderModel...")
        tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(model_path)
    elif "ankh" in model_path.lower():
        # Ankh models are T5-based but use AutoTokenizer
        from transformers import T5EncoderModel
        print("Detected Ankh model, using AutoTokenizer + T5EncoderModel...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5EncoderModel.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)    
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    return model, tokenizer

def encode_sequences_batch(model, tokenizer, sequences, device, batch_size=16, model_path=""):
    """Encode protein sequences using PLM model in batches"""
    print(f"Encoding {len(sequences)} sequences in batches of {batch_size}...")
    
    embeddings = []
    
    # Check if this is a T5 model
    is_t5_model = hasattr(model, 'encoder')
    is_ankh_model = "ankh" in model_path.lower()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i + batch_size]
            
            # For T5 models, add spaces between amino acids (except for Ankh which uses different format)
            if is_t5_model and not is_ankh_model:
                # ProtT5 needs spaces between amino acids
                batch_seqs = [' '.join(list(seq)) for seq in batch_seqs]
            
            # Tokenize batch
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings
            if is_t5_model:
                # T5EncoderModel returns encoder outputs directly
                outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                sequence_embeddings = outputs.last_hidden_state
            else:
                # Other models
                outputs = model(**inputs)
                sequence_embeddings = outputs.last_hidden_state
            
            # Pool embeddings (mean pooling over sequence length, excluding special tokens)
            attention_mask = inputs['attention_mask']
            batch_embeddings = []
            
            for j in range(len(batch_seqs)):
                # Get valid token positions (excluding padding and special tokens)
                valid_mask = attention_mask[j] == 1
                valid_embeddings = sequence_embeddings[j][valid_mask]
                
                # Skip first and last tokens (CLS and EOS) - except for T5 which uses different tokens
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
    # Extract the last directory name or model identifier
    path_parts = model_path.rstrip('/').split('/')
    model_name = path_parts[-1]
    
    return model_name

def main(dataset_name, model_path, batch_size, out_dir, use_cuda):
    """Main function for PLM-based VenusX analysis"""
    model_name = get_model_name(model_path)
    print(f"[*] Processing VenusX_{dataset_name} dataset with PLM ({model_name})...")
    
    # Setup device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory with model-specific subdirectory
    dataset_out_dir = os.path.join(out_dir, model_name, f"VenusX_{dataset_name}")
    os.makedirs(dataset_out_dir, exist_ok=True)
    
    # Step 1: Load datasets and extract fragments
    print("[1] Loading datasets and extracting fragments...")
    
    train_data = load_venusx_dataset(dataset_name, "train")
    test_data = load_venusx_dataset(dataset_name, "test")
    
    train_sequences, train_labels, train_ids = extract_fragments_with_labels(train_data)
    test_sequences, test_labels, test_ids = extract_fragments_with_labels(test_data)
    
    print(f"Train fragments: {len(train_sequences)}")
    print(f"Test fragments: {len(test_sequences)}")
    
    # Step 2: Load PLM model
    print("[2] Loading PLM model...")
    model, tokenizer = load_plm_model(model_path, device)
    
    # Step 3: Encode training sequences (with caching)
    print("[3] Encoding training sequences...")
    train_cache_path = os.path.join(dataset_out_dir, "train_embeddings.npy")
    train_embeddings = load_embeddings_cache(train_cache_path)
    
    if train_embeddings is None:
        train_embeddings = encode_sequences_batch(
            model, tokenizer, train_sequences, device, batch_size, model_path
        )
        save_embeddings_cache(train_embeddings, train_cache_path)
    
    # Step 4: Encode test sequences (with caching)
    print("[4] Encoding test sequences...")
    test_cache_path = os.path.join(dataset_out_dir, "test_embeddings.npy")
    test_embeddings = load_embeddings_cache(test_cache_path)
    
    if test_embeddings is None:
        test_embeddings = encode_sequences_batch(
            model, tokenizer, test_sequences, device, batch_size, model_path
        )
        save_embeddings_cache(test_embeddings, test_cache_path)
    
    # Step 5: Find most similar training examples for each test sequence
    print("[5] Finding most similar training examples...")
    predictions = find_most_similar(test_embeddings, train_embeddings, train_labels, train_ids)
    
    # Step 6: Create results dataframe
    print("[6] Creating results dataframe...")
    results = []
    for i, (test_id, true_label, pred_info) in enumerate(zip(test_ids, test_labels, predictions)):
        results.append({
            'query_sequence': test_id,
            'true_interpro_id': true_label,
            'predicted_interpro_id': pred_info['predicted_interpro_id'],
            'matched_train_sequence': pred_info['matched_train_id'],
            'similarity_score': pred_info['similarity_score']
        })
    
    # Step 7: Save results to CSV
    print("[7] Saving results to CSV...")
    results_df = pd.DataFrame(results)
    csv_output = os.path.join(dataset_out_dir, "plm_predictions.csv")
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
        print(f"Average similarity score: {avg_similarity:.4f}")
    
    # Save additional metadata
    metadata = {
        'dataset_name': dataset_name,
        'model_path': model_path,
        'batch_size': batch_size,
        'device': str(device),
        'num_train_fragments': len(train_sequences),
        'num_test_fragments': len(test_sequences),
        'accuracy': accuracy if len(results) > 0 else 0.0,
        'avg_similarity': avg_similarity if len(results) > 0 else 0.0
    }
    
    metadata_file = os.path.join(dataset_out_dir, "plm_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_file}")
    
    return csv_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein Language Model analysis for VenusX protein fragments")
    parser.add_argument("--dataset", choices=["Act", "BindI", "Dom", "Evo", "Motif"], required=True,
                       help="VenusX dataset to analyze")
    parser.add_argument("--model_path", type=str, 
                       default="/home/lfj/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D/",
                       help="Path to protein language model (ESM2, ProtBERT, etc.)")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for encoding sequences")
    parser.add_argument("--out_dir", type=str, default="baselines/plm_results", 
                       help="Output directory")
    parser.add_argument("--cpu", action="store_true", 
                       help="Force CPU usage (default: use CUDA if available)")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    main(args.dataset, args.model_path, args.batch_size, args.out_dir, not args.cpu)