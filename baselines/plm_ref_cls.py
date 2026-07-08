import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import math
from collections import Counter
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

DATASET_NAMES = ["Act", "BindI", "Dom", "Evo", "Motif"]
ESMC_MODEL_PATH = "/home/dataset-local/projects_dir/pretrained_model/ESMC-600M/"
INTERPROT_REPO_ROOT = "/home/dataset-local/projects_dir/CAPSUL/interprot"
INTERPROT_ESM_MODEL_DIR = "/home/dataset-local/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D"
INTERPROT_SAE_CHECKPOINT = "/home/dataset-local/projects_dir/pretrained_model/InterProt-ESM2-SAEs/esm2_plm1280_l24_sae4096.safetensors"
INTERPROT_PLM_LAYER = 24
INTERPROT_ESM_DIM = 1280
INTERPROT_SAE_DIM = 4096
INTERPROT_POOLING = "max"

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

def _mean_pool_valid_tokens(sequence_embeddings, attention_mask):
    """Mean pool valid residue/token embeddings while skipping boundary tokens."""
    batch_embeddings = []
    for j in range(sequence_embeddings.shape[0]):
        valid_mask = attention_mask[j] == 1
        valid_embeddings = sequence_embeddings[j][valid_mask]

        if len(valid_embeddings) > 2:
            valid_embeddings = valid_embeddings[1:-1]

        if len(valid_embeddings) > 0:
            pooled_embedding = valid_embeddings.mean(dim=0)
        else:
            raise ValueError("No valid tokens found for pooling.")

        batch_embeddings.append(pooled_embedding.float().cpu().numpy())

    return batch_embeddings

def _is_esmc_model(model_path):
    model_path_lower = model_path.lower().rstrip("/")
    return model_path_lower == "esmc" or "esmc" in os.path.basename(model_path_lower)

def _is_interprot_model(model_path):
    model_path_lower = model_path.lower().rstrip("/")
    model_basename = os.path.basename(model_path_lower)
    return (
        model_path_lower in {"interprot", "interprot_sae", "interprot-esm2-sae"}
        or "interprot" in model_path_lower
        or (model_basename.startswith("esm2_plm") and model_basename.endswith(".safetensors"))
    )

class ESMCSequenceEncoder:
    """ESMC masked-LM wrapper exposing the same batch encoder interface."""

    def __init__(self, model_path, device):
        from transformers import AutoModelForMaskedLM

        load_kwargs = {}
        if device.type == "cuda":
            load_kwargs["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, **load_kwargs).eval()
        if "device_map" not in load_kwargs:
            self.model = self.model.to(device)

        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def encode_sequences(self, sequences, batch_size=16, max_length=1024):
        embeddings = []
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i + batch_size]
            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            sequence_embeddings = outputs.hidden_states[-1]
            embeddings.extend(
                _mean_pool_valid_tokens(sequence_embeddings, inputs["attention_mask"])
            )

        return np.array(embeddings)

class InterProtSequenceEncoder:
    """InterProt ESM2+SAE feature extractor adapted from the reference script."""

    def __init__(
        self,
        esm_model_dir,
        sae_checkpoint,
        plm_layer,
        esm_dim,
        sae_dim,
        pooling,
        device,
    ):
        try:
            from safetensors.torch import load_file
            from transformers import EsmModel
        except ImportError as exc:
            raise ImportError(
                "InterProt encoding requires `transformers` and `safetensors` "
                "in the active environment."
            ) from exc

        if INTERPROT_REPO_ROOT not in sys.path:
            sys.path.insert(0, INTERPROT_REPO_ROOT)
        from interprot.sae_model import SparseAutoencoder

        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_dir)
        self.esm_model = EsmModel.from_pretrained(esm_model_dir).to(device).eval()
        self.sae_model = SparseAutoencoder(esm_dim, sae_dim)
        self.sae_model.load_state_dict(load_file(sae_checkpoint))
        self.sae_model = self.sae_model.to(device).eval()

        for param in self.esm_model.parameters():
            param.requires_grad = False
        for param in self.sae_model.parameters():
            param.requires_grad = False

        self.plm_layer = plm_layer
        self.pooling = pooling
        self.device = device

    @torch.no_grad()
    def encode_sequences(self, sequences, batch_size=16, max_length=1024):
        embeddings = []
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i + batch_size]
            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.esm_model(**inputs, output_hidden_states=True)
            esm_layer_acts = outputs.hidden_states[self.plm_layer]
            sae_acts = self.sae_model.get_acts(esm_layer_acts)
            lengths = inputs["attention_mask"].sum(dim=1)

            for idx, length in enumerate(lengths.tolist()):
                start = 1 if length > 2 else 0
                end = length - 1 if length > 2 else length
                residue_acts = sae_acts[idx, start:end]
                if residue_acts.numel() == 0:
                    residue_acts = sae_acts[idx, :1]

                if self.pooling == "mean":
                    feature = residue_acts.mean(dim=0)
                elif self.pooling == "max":
                    feature = residue_acts.max(dim=0).values
                elif self.pooling == "mean_max":
                    feature = torch.cat(
                        [residue_acts.mean(dim=0), residue_acts.max(dim=0).values],
                        dim=0,
                    )
                else:
                    raise ValueError(f"Unsupported pooling: {self.pooling}")

                embeddings.append(feature.float().cpu().numpy())

        return np.array(embeddings)

def load_plm_model(model_path, device):
    """Load protein language model and tokenizer"""
    print(f"Loading PLM model from {model_path}...")

    if _is_esmc_model(model_path):
        resolved_model_path = ESMC_MODEL_PATH if model_path.lower().rstrip("/") == "esmc" else model_path
        print("Detected ESMC model, using AutoModelForMaskedLM...")
        encoder = ESMCSequenceEncoder(resolved_model_path, device)
        print(f"Model loaded on device: {encoder.device}")
        return encoder, None

    if _is_interprot_model(model_path):
        sae_checkpoint = (
            model_path
            if model_path.lower().rstrip("/").endswith(".safetensors")
            else INTERPROT_SAE_CHECKPOINT
        )
        print("Detected InterProt model, using ESM2 hidden states + SAE activations...")
        encoder = InterProtSequenceEncoder(
            esm_model_dir=INTERPROT_ESM_MODEL_DIR,
            sae_checkpoint=sae_checkpoint,
            plm_layer=INTERPROT_PLM_LAYER,
            esm_dim=INTERPROT_ESM_DIM,
            sae_dim=INTERPROT_SAE_DIM,
            pooling=INTERPROT_POOLING,
            device=device,
        )
        print(f"Model loaded on device: {encoder.device}")
        return encoder, None
    
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

    if hasattr(model, "encode_sequences"):
        return model.encode_sequences(sequences, batch_size=batch_size, max_length=1024)
    
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
            
            batch_embeddings = _mean_pool_valid_tokens(
                sequence_embeddings, inputs['attention_mask']
            )
            
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

def compute_classification_metrics(results, label_to_idx=None):
    """Compute multiclass metrics without a dense confusion matrix.

    The main macro precision/recall/f1 fields are averaged over labels that
    appear as true labels in the current test set. Stricter observed/global
    macro metrics are also saved for interpretation.
    """
    if not results:
        global_num_classes = len(label_to_idx) if label_to_idx is not None else 0
        return {
            'acc': 0.0,
            'recall': 0.0,
            'precision': 0.0,
            'f1': 0.0,
            'mcc': 0.0,
            'total': 0,
            'correct': 0,
            'num_classes': 0,
            'macro_average': 'test_true_labels',
            'observed_precision': 0.0,
            'observed_recall': 0.0,
            'observed_f1': 0.0,
            'observed_num_classes': 0,
            'global_precision': 0.0,
            'global_recall': 0.0,
            'global_f1': 0.0,
            'global_num_classes': global_num_classes,
        }

    true_labels = [r['true_interpro_id'] for r in results]
    pred_labels = [r['predicted_interpro_id'] for r in results]
    true_count = Counter(true_labels)
    pred_count = Counter(pred_labels)
    tp_count = Counter(
        true_label
        for true_label, pred_label in zip(true_labels, pred_labels)
        if true_label == pred_label
    )

    total = len(results)
    correct = sum(tp_count.values())
    accuracy = correct / total

    def macro_scores(labels):
        labels = list(labels)
        if not labels:
            return 0.0, 0.0, 0.0, 0

        precisions = []
        recalls = []
        f1_scores = []
        for label in labels:
            tp = tp_count[label]
            precision = tp / pred_count[label] if pred_count[label] > 0 else 0.0
            recall = tp / true_count[label] if true_count[label] > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        return (
            sum(precisions) / len(labels),
            sum(recalls) / len(labels),
            sum(f1_scores) / len(labels),
            len(labels),
        )

    true_label_set = sorted(true_count)
    observed_label_set = sorted(set(true_count) | set(pred_count))
    if label_to_idx is None:
        global_label_set = observed_label_set
    else:
        global_label_set = sorted(set(label_to_idx) | set(true_count) | set(pred_count))

    precision, recall, f1, num_classes = macro_scores(true_label_set)
    observed_precision, observed_recall, observed_f1, observed_num_classes = macro_scores(
        observed_label_set
    )
    global_precision, global_recall, global_f1, global_num_classes = macro_scores(
        global_label_set
    )

    active_labels = set(true_count) | set(pred_count)
    sum_row_col = sum(true_count[label] * pred_count[label] for label in active_labels)
    numerator = correct * total - sum_row_col
    denominator_left = total * total - sum(value * value for value in true_count.values())
    denominator_right = total * total - sum(value * value for value in pred_count.values())
    denominator = math.sqrt(denominator_left * denominator_right)
    mcc = numerator / denominator if denominator > 0 else 0.0

    return {
        'acc': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'mcc': mcc,
        'total': total,
        'correct': correct,
        'num_classes': num_classes,
        'macro_average': 'test_true_labels',
        'observed_precision': observed_precision,
        'observed_recall': observed_recall,
        'observed_f1': observed_f1,
        'observed_num_classes': observed_num_classes,
        'global_precision': global_precision,
        'global_recall': global_recall,
        'global_f1': global_f1,
        'global_num_classes': global_num_classes,
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
    
    # parser.add_argument("--model_path", type=str, default="/home/dataset-local/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D/")
    # parser.add_argument("--model_path", type=str, default="esmc")
    parser.add_argument("--model_path", type=str, default="interprot")

    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for encoding sequences")
    parser.add_argument("--out_dir", type=str, default=os.path.join(project_root, "baselines", "plm_ref_cls_results"), 
                       help="Output directory")
    # parser.add_argument("--data_dir", type=str, default="data_70", help="Dataset root directory")
    # parser.add_argument("--data_dir", type=str, default="data_30", help="Dataset root directory")
    parser.add_argument("--data_dir", type=str, default="data_frag_50", help="Dataset root directory")
    parser.add_argument("--cpu", default=False, action="store_true", 
                       help="Force CPU usage (default: use CUDA if available)")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    eval_datasets = [args.dataset] if args.dataset is not None else args.datasets
    main(args.model_path, args.batch_size, args.out_dir, not args.cpu, args.data_dir, eval_datasets)
