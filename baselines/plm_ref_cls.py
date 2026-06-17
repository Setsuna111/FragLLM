import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

DATASET_NAMES = ["Act", "BindI", "Dom", "Evo", "Motif"]

def resolve_data_dir(data_dir):
    """Resolve dataset root relative to project root unless an absolute path is given."""
    data_dir = os.path.expanduser(data_dir)
    if os.path.isabs(data_dir):
        return data_dir
    return os.path.join(project_root, data_dir)

def get_data_dir_name(data_dir):
    """Get a stable name for separating output/cache directories."""
    return os.path.basename(os.path.normpath(data_dir))

def load_venusx_dataset(dataset_name, split, data_dir="data"):
    """Load VenusX dataset from JSON file"""
    data_root = resolve_data_dir(data_dir)
    dataset_path = os.path.join(data_root, f"VenusX_{dataset_name}", f"{split}.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    return data
    
def extract_fragments_with_labels(data, dataset_name=None):
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
                if dataset_name is not None:
                    frag_id = f"{dataset_name}:{frag_id}"
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

def load_embeddings_cache(cache_path, expected_count=None):
    """Load embeddings from cache file"""
    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        if expected_count is not None and len(embeddings) != expected_count:
            print(
                f"Cache size mismatch for {cache_path}: "
                f"{len(embeddings)} != {expected_count}. Recomputing..."
            )
            return None
        print(f"Loaded embeddings cache from {cache_path}")
        return embeddings
    return None

def get_model_name(model_path):
    """Extract model name from model path for output directory"""
    # Extract the last directory name or model identifier
    path_parts = model_path.rstrip('/').split('/')
    model_name = path_parts[-1]
    
    return model_name

def get_base_out_dir(out_dir, model_name, data_dir_name):
    """Get output directory for a model/data split."""
    if data_dir_name == "data":
        return os.path.join(out_dir, model_name)
    return os.path.join(out_dir, model_name, data_dir_name)

def compute_classification_metrics(results, label_to_idx):
    """Compute multiclass metrics over the global InterPro label space."""
    if not results:
        return {
            'acc': 0.0,
            'recall': 0.0,
            'precision': 0.0,
            'f1': 0.0,
            'mcc': 0.0,
            'total': 0,
            'correct': 0,
            'num_classes': len(label_to_idx),
        }

    true_idx = [label_to_idx[r['true_interpro_id']] for r in results]
    pred_idx = [label_to_idx[r['predicted_interpro_id']] for r in results]
    label_indices = list(range(len(label_to_idx)))

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_idx,
        pred_idx,
        labels=label_indices,
        average='macro',
        zero_division=0,
    )
    correct = sum(1 for true_label, pred_label in zip(true_idx, pred_idx) if true_label == pred_label)

    return {
        'acc': float(accuracy_score(true_idx, pred_idx)),
        'recall': float(recall),
        'precision': float(precision),
        'f1': float(f1),
        'mcc': float(matthews_corrcoef(true_idx, pred_idx)),
        'total': len(results),
        'correct': int(correct),
        'num_classes': len(label_to_idx),
    }

def main(model_path, batch_size, out_dir, use_cuda, data_dir="data", eval_datasets=None):
    """Main function for PLM-based VenusX analysis"""
    eval_datasets = eval_datasets or DATASET_NAMES
    model_name = get_model_name(model_path)
    data_root = resolve_data_dir(data_dir)
    data_dir_name = get_data_dir_name(data_root)
    print(f"[*] Processing VenusX datasets with PLM ({model_name})...")
    print(f"Using dataset root: {data_root}")
    print(f"Training datasets: {', '.join(DATASET_NAMES)}")
    print(f"Evaluation datasets: {', '.join(eval_datasets)}")
    
    # Setup device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_out_dir = get_base_out_dir(out_dir, model_name, data_dir_name)
    os.makedirs(base_out_dir, exist_ok=True)
    
    # Step 1: Load all datasets and extract fragments
    print("[1] Loading datasets and extracting fragments...")

    train_sequences = []
    train_labels = []
    train_ids = []
    test_sets = {}
    global_labels = set()

    for dataset_name in DATASET_NAMES:
        train_data = load_venusx_dataset(dataset_name, "train", data_root)
        test_data = load_venusx_dataset(dataset_name, "test", data_root)

        ds_train_sequences, ds_train_labels, ds_train_ids = extract_fragments_with_labels(
            train_data, dataset_name
        )
        ds_test_sequences, ds_test_labels, ds_test_ids = extract_fragments_with_labels(
            test_data, dataset_name
        )

        train_sequences.extend(ds_train_sequences)
        train_labels.extend(ds_train_labels)
        train_ids.extend(ds_train_ids)
        test_sets[dataset_name] = {
            'sequences': ds_test_sequences,
            'labels': ds_test_labels,
            'ids': ds_test_ids,
        }
        global_labels.update(ds_train_labels)
        global_labels.update(ds_test_labels)

        print(
            f"VenusX_{dataset_name}: "
            f"train fragments={len(ds_train_sequences)}, test fragments={len(ds_test_sequences)}"
        )

    label_to_idx = {label: idx for idx, label in enumerate(sorted(global_labels))}
    label_index_file = os.path.join(base_out_dir, "plm_label_to_idx.json")
    with open(label_index_file, 'w') as f:
        json.dump(label_to_idx, f, indent=2)

    print(f"Combined train fragments: {len(train_sequences)}")
    print(f"Global label count: {len(label_to_idx)}")
    print(f"Label index saved to {label_index_file}")
    
    # Step 2: Load PLM model
    print("[2] Loading PLM model...")
    model, tokenizer = load_plm_model(model_path, device)
    
    # Step 3: Encode combined training sequences (with caching)
    print("[3] Encoding combined training sequences...")
    train_cache_path = os.path.join(base_out_dir, "all_train_embeddings.npy")
    train_embeddings = load_embeddings_cache(train_cache_path, expected_count=len(train_sequences))
    
    if train_embeddings is None:
        train_embeddings = encode_sequences_batch(
            model, tokenizer, train_sequences, device, batch_size, model_path
        )
        save_embeddings_cache(train_embeddings, train_cache_path)

    all_metrics = {}
    csv_outputs = {}

    for dataset_name in eval_datasets:
        if dataset_name not in test_sets:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset_out_dir = os.path.join(base_out_dir, f"VenusX_{dataset_name}")
        os.makedirs(dataset_out_dir, exist_ok=True)

        test_sequences = test_sets[dataset_name]['sequences']
        test_labels = test_sets[dataset_name]['labels']
        test_ids = test_sets[dataset_name]['ids']

        # Step 4: Encode test sequences (with caching)
        print(f"[4:{dataset_name}] Encoding test sequences...")
        test_cache_path = os.path.join(dataset_out_dir, "test_embeddings.npy")
        test_embeddings = load_embeddings_cache(test_cache_path, expected_count=len(test_sequences))

        if test_embeddings is None:
            test_embeddings = encode_sequences_batch(
                model, tokenizer, test_sequences, device, batch_size, model_path
            )
            save_embeddings_cache(test_embeddings, test_cache_path)

        # Step 5: Find most similar training examples for each test sequence
        print(f"[5:{dataset_name}] Finding most similar training examples...")
        predictions = find_most_similar(test_embeddings, train_embeddings, train_labels, train_ids)

        # Step 6: Create results dataframe
        print(f"[6:{dataset_name}] Creating results dataframe...")
        results = []
        for test_id, true_label, pred_info in zip(test_ids, test_labels, predictions):
            results.append({
                'query_sequence': test_id,
                'true_interpro_id': true_label,
                'true_label_idx': label_to_idx[true_label],
                'predicted_interpro_id': pred_info['predicted_interpro_id'],
                'predicted_label_idx': label_to_idx[pred_info['predicted_interpro_id']],
                'matched_train_sequence': pred_info['matched_train_id'],
                'similarity_score': pred_info['similarity_score']
            })

        # Step 7: Save results to CSV
        print(f"[7:{dataset_name}] Saving results and metrics...")
        results_df = pd.DataFrame(results)
        csv_output = os.path.join(dataset_out_dir, "plm_predictions.csv")
        results_df.to_csv(csv_output, index=False)
        csv_outputs[dataset_name] = csv_output

        metrics = compute_classification_metrics(results, label_to_idx)
        metrics['avg_similarity'] = (
            float(np.mean([r['similarity_score'] for r in results])) if results else 0.0
        )
        all_metrics[dataset_name] = metrics

        metrics_file = os.path.join(dataset_out_dir, "plm_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        metadata = {
            'dataset_name': dataset_name,
            'train_datasets': DATASET_NAMES,
            'data_dir': data_dir,
            'data_root': data_root,
            'model_path': model_path,
            'batch_size': batch_size,
            'device': str(device),
            'num_train_fragments': len(train_sequences),
            'num_test_fragments': len(test_sequences),
            'num_global_labels': len(label_to_idx),
            'metrics': metrics,
        }

        metadata_file = os.path.join(dataset_out_dir, "plm_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Results saved to {csv_output}")
        print(f"Metrics saved to {metrics_file}")
        print(f"Total predictions: {len(results)}")
        print(f"Accuracy: {metrics['acc']:.4f} ({metrics['correct']}/{metrics['total']})")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        print(f"Average similarity score: {metrics['avg_similarity']:.4f}")

    all_metrics_file = os.path.join(base_out_dir, "plm_all_metrics.json")
    with open(all_metrics_file, 'w') as f:
        json.dump({
            'train_datasets': DATASET_NAMES,
            'eval_datasets': eval_datasets,
            'num_train_fragments': len(train_sequences),
            'num_global_labels': len(label_to_idx),
            'metrics': all_metrics,
        }, f, indent=2)

    print(f"All metrics saved to {all_metrics_file}")

    return csv_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein Language Model analysis for VenusX protein fragments")
    parser.add_argument("--dataset", choices=DATASET_NAMES, default=None,
                       help="Deprecated alias for evaluating one test dataset. Training always uses all datasets.")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=DATASET_NAMES,
                       help="VenusX test datasets to evaluate. Training always uses all datasets.")
    parser.add_argument("--model_path", type=str, 
                       default="/home/dataset-local/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D/",
                       help="Path to protein language model (ESM2, ProtBERT, etc.)")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for encoding sequences")
    parser.add_argument("--out_dir", type=str, default=os.path.join(project_root, "baselines", "plm_results"), 
                       help="Output directory")
    parser.add_argument("--data_dir", type=str, default="data_70",
                       help="Dataset root directory, e.g. data, data_70, data_30, or an absolute path")
    parser.add_argument("--cpu", default=False, action="store_true", 
                       help="Force CPU usage (default: use CUDA if available)")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    eval_datasets = [args.dataset] if args.dataset is not None else args.datasets
    main(args.model_path, args.batch_size, args.out_dir, not args.cpu, args.data_dir, eval_datasets)
