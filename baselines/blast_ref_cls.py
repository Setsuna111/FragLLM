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
import math
from collections import Counter
from tqdm import tqdm

DATASET_NAMES = ["Act", "BindI", "Dom", "Evo", "Motif"]

def get_data_dir_name(data_dir):
    """Get a stable name for separating output/cache directories."""
    return os.path.basename(os.path.normpath(data_dir))

def load_venusx_dataset(dataset_name, split, data_dir):
    """Load VenusX dataset from JSON file"""
    dataset_path = os.path.join(project_root, data_dir, f"VenusX_{dataset_name}", f"{split}.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    return data

def extract_fragments_to_fasta(data, output_fasta, interpro_index_file, dataset_name=None):
    """Extract fragment sequences and create FASTA file with interpro_id indexing"""
    fragments = []
    interpro_index = {}
    
    for protein in data:
        uid = protein['uid']
        for fragment_group in protein['fragments']:
            interpro_id = fragment_group['interpro_id']
            for i, frag in enumerate(fragment_group['frags']):
                frag_id = f"{uid}_{interpro_id}_{i}"
                if dataset_name is not None:
                    frag_id = f"{dataset_name}:{frag_id}"
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

def load_combined_train_fragments(dataset_names, data_dir):
    """Load and concatenate train fragments from all VenusX datasets."""
    all_fragments = []
    all_interpro_index = {}
    global_labels = set()

    for dataset_name in dataset_names:
        train_data = load_venusx_dataset(dataset_name, "train", data_dir)
        fragments = []

        for protein in train_data:
            uid = protein['uid']
            for fragment_group in protein['fragments']:
                interpro_id = fragment_group['interpro_id']
                global_labels.add(interpro_id)
                for i, frag in enumerate(fragment_group['frags']):
                    frag_id = f"{dataset_name}:{uid}_{interpro_id}_{i}"
                    fragments.append((frag_id, frag['sequence']))
                    all_interpro_index[frag_id] = interpro_id

        all_fragments.extend(fragments)
        print(f"VenusX_{dataset_name}: train fragments={len(fragments)}")

    return all_fragments, all_interpro_index, global_labels

def collect_interpro_labels(dataset_names, split, data_dir):
    """Collect InterPro labels from a split across VenusX datasets."""
    labels = set()
    for dataset_name in dataset_names:
        data = load_venusx_dataset(dataset_name, split, data_dir)
        for protein in data:
            for fragment_group in protein['fragments']:
                labels.add(fragment_group['interpro_id'])
    return labels

def write_fragments_to_fasta(fragments, output_fasta, interpro_index, interpro_index_file):
    """Write pre-extracted fragments and their InterPro index."""
    with open(output_fasta, 'w') as f:
        for frag_id, seq in fragments:
            f.write(f">{frag_id}\n{seq}\n")

    with open(interpro_index_file, 'w') as f:
        json.dump(interpro_index, f, indent=2)

    print(f"Extracted {len(fragments)} fragments to {output_fasta}")
    print(f"Saved interpro index to {interpro_index_file}")

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

def compute_classification_metrics(predictions, label_to_idx=None):
    """Compute multiclass metrics without a dense confusion matrix.

    The main macro precision/recall/f1 fields are averaged over labels that
    appear as true labels in the current test set. Stricter observed/global
    macro metrics are also saved for interpretation.
    """
    if not predictions:
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

    true_labels = [p['true_interpro_id'] for p in predictions]
    pred_labels = [p['predicted_interpro_id'] for p in predictions]
    true_count = Counter(true_labels)
    pred_count = Counter(pred_labels)
    tp_count = Counter(
        true_label
        for true_label, pred_label in zip(true_labels, pred_labels)
        if true_label == pred_label
    )

    total = len(predictions)
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

def main(dataset_name, num_threads, out_dir, data_dir):
    """Main function for VenusX BLAST analysis"""
    print(f"[*] Processing VenusX_{dataset_name} dataset...")
    print(f"Training datasets: {', '.join(DATASET_NAMES)}")
    
    # Create output directory
    data_dir_name = get_data_dir_name(data_dir)
    dataset_out_dir = os.path.join(out_dir, f"VenusX_{dataset_name}_{data_dir_name}_all_train")
    os.makedirs(dataset_out_dir, exist_ok=True)
    
    # Step 1: Load datasets and extract fragments
    print("[1] Loading datasets and extracting fragments...")
    
    train_fragments, train_interpro_index, global_labels = load_combined_train_fragments(
        DATASET_NAMES, data_dir
    )
    test_data = load_venusx_dataset(dataset_name, "test", data_dir)
    
    # Extract train fragments from the union of all training datasets
    train_fasta = os.path.join(dataset_out_dir, "train_fragments.fasta")
    train_index_file = os.path.join(dataset_out_dir, "train_interpro_index.json")
    write_fragments_to_fasta(
        train_fragments, train_fasta, train_interpro_index, train_index_file
    )
    
    # Extract test fragments
    test_fasta = os.path.join(dataset_out_dir, "test_fragments.fasta")
    test_index_file = os.path.join(dataset_out_dir, "test_interpro_index.json")
    test_fragments, test_interpro_index = extract_fragments_to_fasta(
        test_data, test_fasta, test_index_file, dataset_name
    )

    global_labels.update(collect_interpro_labels(DATASET_NAMES, "test", data_dir))
    label_to_idx = {label: idx for idx, label in enumerate(sorted(global_labels))}
    label_index_file = os.path.join(dataset_out_dir, "blast_label_to_idx.json")
    with open(label_index_file, 'w') as f:
        json.dump(label_to_idx, f, indent=2)

    print(f"Combined train fragments: {len(train_fragments)}")
    print(f"Global label count: {len(label_to_idx)}")
    print(f"Label index saved to {label_index_file}")
    
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
    results_df['true_label_idx'] = results_df['true_interpro_id'].map(label_to_idx)
    results_df['predicted_label_idx'] = results_df['predicted_interpro_id'].map(label_to_idx)
    csv_output = os.path.join(dataset_out_dir, "blast_predictions.csv")
    results_df.to_csv(csv_output, index=False)
    
    print(f"[✓] Results saved to {csv_output}")
    print(f"[✓] Total predictions: {len(predictions)}")
    
    # Calculate classification metrics
    metrics = compute_classification_metrics(predictions, label_to_idx)
    metrics_output = os.path.join(dataset_out_dir, "blast_metrics.json")
    with open(metrics_output, 'w') as f:
        json.dump(metrics, f, indent=2)

    metadata = {
        'dataset_name': dataset_name,
        'train_datasets': DATASET_NAMES,
        'data_dir': data_dir,
        'num_train_fragments': len(train_fragments),
        'num_test_fragments': len(test_fragments),
        'num_global_labels': len(label_to_idx),
        'metrics': metrics,
    }
    metadata_output = os.path.join(dataset_out_dir, "blast_metadata.json")
    with open(metadata_output, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[✓] Accuracy: {metrics['acc']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"[✓] Macro average: {metrics['macro_average']} ({metrics['num_classes']} classes)")
    print(f"[✓] Recall: {metrics['recall']:.4f}")
    print(f"[✓] Precision: {metrics['precision']:.4f}")
    print(f"[✓] F1: {metrics['f1']:.4f}")
    print(f"[✓] Observed-label F1: {metrics['observed_f1']:.4f} ({metrics['observed_num_classes']} classes)")
    print(f"[✓] Global-label F1: {metrics['global_f1']:.4f} ({metrics['global_num_classes']} classes)")
    print(f"[✓] MCC: {metrics['mcc']:.4f}")
    print(f"[✓] Metrics saved to {metrics_output}")
    print(f"[✓] Metadata saved to {metadata_output}")
    
    # Clean up database files
    os.system(f"rm -rf {db_name}*")
    
    return csv_output

def run_evaluations(eval_datasets, num_threads, out_dir, data_dir):
    """Run BLAST analysis for one or more VenusX test datasets."""
    csv_outputs = {}
    print(f"Evaluation datasets: {', '.join(eval_datasets)}")

    for dataset_name in eval_datasets:
        csv_outputs[dataset_name] = main(dataset_name, num_threads, out_dir, data_dir)

    return csv_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLAST analysis for VenusX protein fragments")
    parser.add_argument("--data_dir", default="data_70", help="Dataset root directory, e.g. data, data_70, data_30, or an absolute path")
    # parser.add_argument("--data_dir", default="data_30", help="VenusX dataset to analyze")
    parser.add_argument("--dataset", default=None, choices=DATASET_NAMES, help="Evaluate one VenusX test dataset. Training always uses all datasets.")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=DATASET_NAMES, help="VenusX test datasets to evaluate. Training always uses all datasets.")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for BLAST")
    parser.add_argument("--out_dir", type=str, default=os.path.join(project_root, "baselines", "blast_ref_cls_results"), help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    eval_datasets = [args.dataset] if args.dataset is not None else args.datasets
    run_evaluations(eval_datasets, args.num_threads, args.out_dir, args.data_dir)
