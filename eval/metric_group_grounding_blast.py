import pandas as pd
import argparse
import numpy as np
from typing import Any, Dict
import json
from collections import defaultdict

argParser = argparse.ArgumentParser()
argParser.add_argument("--results_path", type=str, help="path to BLAST classification results CSV file")
argParser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for position matching")
args = argParser.parse_args()

def parse_ranges(range_str):
    """Parse range string like '249-265;300-350' into list of tuples"""
    if not range_str or pd.isna(range_str):
        return []
    ranges = []
    for r in range_str.split(';'):
        if r.strip():
            start, end = map(int, r.split('-'))
            ranges.append((start, end))
    return ranges

def compute_position_iou(pos1, pos2):
    """Compute IoU between two position intervals"""
    start1, end1 = pos1
    start2, end2 = pos2
    
    # Compute intersection
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    intersection = max(0, inter_end - inter_start + 1)
    
    # Compute union
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = union_end - union_start + 1
    
    return intersection / union if union > 0 else 0

def match_fragments_by_iou(pred_ranges, true_ranges, iou_threshold=0.1):
    """Match predicted and true fragments based on IoU"""
    if not pred_ranges or not true_ranges:
        return [], []
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_ranges), len(true_ranges)))
    for i, pred_range in enumerate(pred_ranges):
        for j, true_range in enumerate(true_ranges):
            iou_matrix[i, j] = compute_position_iou(pred_range, true_range)
    
    # Find best matches using greedy approach
    matched_pred_idx = []
    matched_true_idx = []
    
    while iou_matrix.size > 0 and np.max(iou_matrix) >= iou_threshold:
        # Find best match
        best_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        i, j = best_idx
        
        matched_pred_idx.append(i)
        matched_true_idx.append(j)
        
        # Remove matched row and column
        iou_matrix = np.delete(iou_matrix, i, axis=0)
        iou_matrix = np.delete(iou_matrix, j, axis=1)
        
        # Update indices for remaining items
        for k in range(len(matched_pred_idx)-1):
            if matched_pred_idx[k] > i:
                matched_pred_idx[k] -= 1
        for k in range(len(matched_true_idx)-1):
            if matched_true_idx[k] > j:
                matched_true_idx[k] -= 1
    
    return matched_pred_idx, matched_true_idx

def evaluate_blast_classification(args: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate BLAST classification results for multiple localization task"""
    df = pd.read_csv(args.results_path)
    
    # Group by protein (uid) and aggregate results
    protein_results = defaultdict(lambda: {'true_fragments': [], 'pred_fragments': []})
    
    for _, row in df.iterrows():
        uid = row['uid']
        true_ranges = parse_ranges(row['true_ranges'])
        pred_ranges = parse_ranges(row['predicted_ranges'])
        
        protein_results[uid]['true_fragments'].extend(true_ranges)
        protein_results[uid]['pred_fragments'].extend(pred_ranges)
    
    # Calculate metrics
    total_true_fragments = 0
    total_pred_fragments = 0
    total_matched_fragments = 0
    total_iou = 0
    total_proteins = len(protein_results)
    
    per_protein_metrics = []
    
    for uid, data in protein_results.items():
        true_ranges = data['true_fragments']
        pred_ranges = data['pred_fragments']
        
        num_true = len(true_ranges)
        num_pred = len(pred_ranges)
        
        # Match fragments based on IoU
        matched_pred_idx, matched_true_idx = match_fragments_by_iou(
            pred_ranges, true_ranges, args.iou_threshold
        )
        
        num_matched = len(matched_pred_idx)
        
        # Calculate IoU for matched fragments
        protein_iou = 0
        if num_matched > 0:
            for i, j in zip(matched_pred_idx, matched_true_idx):
                protein_iou += compute_position_iou(pred_ranges[i], true_ranges[j])
            protein_iou /= num_matched
        
        # Per-protein metrics
        precision = num_matched / num_pred if num_pred > 0 else 0
        recall = num_matched / num_true if num_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_protein_metrics.append({
            'uid': uid,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': protein_iou,
            'num_true': num_true,
            'num_pred': num_pred,
            'num_matched': num_matched
        })
        
        total_true_fragments += num_true
        total_pred_fragments += num_pred
        total_matched_fragments += num_matched
        total_iou += protein_iou
    
    # Global metrics
    global_precision = total_matched_fragments / total_pred_fragments if total_pred_fragments > 0 else 0
    global_recall = total_matched_fragments / total_true_fragments if total_true_fragments > 0 else 0
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0
    avg_iou = total_iou / total_proteins if total_proteins > 0 else 0
    
    # Average per-protein metrics
    avg_precision = np.mean([m['precision'] for m in per_protein_metrics])
    avg_recall = np.mean([m['recall'] for m in per_protein_metrics])
    avg_f1 = np.mean([m['f1'] for m in per_protein_metrics])
    
    results = {
        'global_precision': global_precision,
        'global_recall': global_recall,
        'global_f1': global_f1,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'avg_iou': avg_iou,
        'total_proteins': total_proteins,
        'total_true_fragments': total_true_fragments,
        'total_pred_fragments': total_pred_fragments,
        'total_matched_fragments': total_matched_fragments
    }
    
    # Save detailed results
    output_file = args.results_path.replace('.csv', '_metrics.json')
    with open(output_file, 'w') as f:
        json.dump({
            'summary_metrics': results,
            'per_protein_metrics': per_protein_metrics
        }, f, indent=2)
    
    # Print results
    print("BLAST Classification Evaluation Results:")
    print("="*50)
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    
    return results

if __name__ == "__main__":
    args = argParser.parse_args()
    evaluate_blast_classification(args)