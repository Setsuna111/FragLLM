import pandas as pd
import argparse
import re
import numpy as np
from typing import Any, Dict, List
import json
argParser = argparse.ArgumentParser()
argParser.add_argument("--results_path",default="/home/lfj/projects_dir/FragLLM/baselines/blast_cls_results/VenusX_Act/single_localization_results.csv", type=str, help="path to BLAST classification results CSV file")
argParser.add_argument("--iou_threshold", default=0.5, type=float, help="iou threshold")

args = argParser.parse_args()


# Parse range strings from BLAST classification results
def parse_range_string(range_str):
    """Parse range string like '249-265;300-320' into list of tuples"""
    ranges = []
    if not range_str or pd.isna(range_str) or range_str.strip() == '':
        return ranges
    
    # Split by semicolon for multiple ranges
    range_parts = range_str.split(';')
    for part in range_parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = part.split('-')
                ranges.append((int(start), int(end)))
            except ValueError:
                print(f"Invalid range format: {part}")
                continue
    return ranges


def match_positions(prediction_positions, reference_positions):
    # Match the positions with position interval distance
    matched_pred_positions = []
    matched_ref_positions = []
    # NOTE: 两种匹配方式，一种是计算区间中心点距离，另一种是计算iou
    # # 计算区间中心点
    # pred_centers = [(a[0] + a[1]) / 2.0 for a in prediction_positions]
    # ref_centers = [(b[0] + b[1]) / 2.0 for b in reference_positions]
    # dist_matrix = np.zeros((len(pred_centers), len(ref_centers)))
    # for i, ac in enumerate(pred_centers):
    #     for j, bc in enumerate(ref_centers):
    #         dist_matrix[i, j] = abs(ac - bc)
    # pred_centers = [(a[0] + a[1]) / 2.0 for a in prediction_positions]
    # ref_centers = [(b[0] + b[1]) / 2.0 for b in reference_positions]
    # dist_matrix = np.zeros((len(pred_centers), len(ref_centers)))
    # for i, ac in enumerate(pred_centers):
    #     for j, bc in enumerate(ref_centers):
    #         dist_matrix[i, j] = abs(ac - bc)
    # # Find one-to-one matches
    # matched_pred_positions = []
    # matched_ref_positions = []
    # while (dist_matrix!=1024).any():
    #     min_dist_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    #     matched_pred_positions.append(prediction_positions[min_dist_idx[0]])
    #     matched_ref_positions.append(reference_positions[min_dist_idx[1]])
    #     dist_matrix[min_dist_idx[0], :] = 1024
    #     dist_matrix[:, min_dist_idx[1]] = 1024

    # iou矩阵
    dist_matrix = np.zeros((len(prediction_positions), len(reference_positions)))
    for i, pred_pos in enumerate(prediction_positions):
        for j, ref_pos in enumerate(reference_positions):
            dist_matrix[i, j] = compute_iou_single(pred_pos, ref_pos)
    # Find one-to-one matches
    matched_pred_positions = []
    matched_ref_positions = []
    while (dist_matrix!=-1).any():
        max_dist_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        matched_pred_positions.append(prediction_positions[max_dist_idx[0]])
        matched_ref_positions.append(reference_positions[max_dist_idx[1]])
        dist_matrix[max_dist_idx[0], :] = -1
        dist_matrix[:, max_dist_idx[1]] = -1
    return matched_pred_positions, matched_ref_positions
    
        

# Compute the distance between the positions
def compute_distance(positions_pre, positions_ref):
    distance = 0
    for pos_pre, pos_ref in zip(positions_pre, positions_ref):
        distance += abs(pos_pre[0] - pos_ref[0]) + abs(pos_pre[1] - pos_ref[1])
    distance /= len(positions_pre)
    return distance

# Compute the iou between the positions
def compute_iou(positions_pre, positions_ref):
    iou = 0
    unions = 0
    intersections = 0
    for pos_pre, pos_ref in zip(positions_pre, positions_ref):
        inter_left = max(pos_pre[0], pos_ref[0])
        inter_right = min(pos_pre[1], pos_ref[1])
        intersection = max(0, inter_right - inter_left + 1)
        union_left = min(pos_pre[0], pos_ref[0])
        union_right = max(pos_pre[1], pos_ref[1])
        union = union_right - union_left + 1
        unions += union
        intersections += intersection
        if union == 0:
            iou += 0
        else:
            iou += intersection / union
    iou /= len(positions_pre)
    return iou, unions, intersections
# Compute the iou between the positions
def compute_iou(positions_pre, positions_ref):
    iou = 0
    unions = 0
    intersections = 0
    for pos_pre, pos_ref in zip(positions_pre, positions_ref):
        inter_left = max(pos_pre[0], pos_ref[0])
        inter_right = min(pos_pre[1], pos_ref[1])
        intersection = max(0, inter_right - inter_left + 1)
        union_left = min(pos_pre[0], pos_ref[0])
        union_right = max(pos_pre[1], pos_ref[1])
        union = union_right - union_left + 1
        unions += union
        intersections += intersection
        if union == 0:
            iou += 0
        else:
            iou += intersection / union
    iou /= len(positions_pre)
    return iou, unions, intersections

# Compute the recall between the positions
def compute_TP(positions_pre, positions_ref, iou_threshold):
    recall = 0
    for pos_pre, pos_ref in zip(positions_pre, positions_ref):
        inter_left = max(pos_pre[0], pos_ref[0])
        inter_right = min(pos_pre[1], pos_ref[1])
        intersection = max(0, inter_right - inter_left + 1)
        union_left = min(pos_pre[0], pos_ref[0])
        union_right = max(pos_pre[1], pos_ref[1])
        union = union_right - union_left + 1
        if union == 0:
            iou = 0
        else:
            iou = intersection / union
        if iou >= iou_threshold:
            recall += 1
    # recall /= len(positions_ref)
    return recall



def evaluate_single_grounding(args: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate BLAST classification results from blast_cls.py output"""
    res = pd.read_csv(args.results_path)
    
    # Filter single localization results only
    if 'task_type' in res.columns:
        res = res[res['task_type'] == 'single']
    
    ious = []
    distances = []
    TP_nums = []
    unions_list = []
    intersections_list = []
    pred_nums = []
    ref_nums = []
    results = {}
    
    for _, row in res.iterrows():
        # Parse true and predicted ranges
        true_ranges = parse_range_string(row['true_ranges'])
        pred_ranges = parse_range_string(row['predicted_ranges'])

        # print(f"True ranges: {true_ranges}, Predicted ranges: {pred_ranges}")  # debug
        
        # Record number of predictions and references
        pred_nums.append(len(pred_ranges))  # TP + FP
        ref_nums.append(len(true_ranges))   # TP + FN
        
        # Handle cases where predictions < references by padding with (0,0)
        if len(pred_ranges) < len(true_ranges):
            pred_ranges = pred_ranges + [(0,0)] * (len(true_ranges) - len(pred_ranges))
        
        # Match positions
        matched_pred_positions, matched_ref_positions = match_positions(pred_ranges, true_ranges)
        
        # Calculate metrics for current sample
        if matched_pred_positions and matched_ref_positions:
            iou, unions, intersections = compute_iou(matched_pred_positions, matched_ref_positions)
            distance = compute_distance(matched_pred_positions, matched_ref_positions)
            tp_num = compute_TP(matched_pred_positions, matched_ref_positions, args.iou_threshold)
        else:
            iou, unions, intersections = 0, 0, 0
            distance = 0
            tp_num = 0
        
        ious.append(iou)
        unions_list.append(unions)
        intersections_list.append(intersections)
        distances.append(distance)
        TP_nums.append(tp_num)
    
    # Calculate aggregate metrics
    avg_iou = sum(ious) / len(ious) if ious else 0
    global_iou = sum(intersections_list) / sum(unions_list) if sum(unions_list) > 0 else 0
    avg_distance = sum(distances) / len(distances) if distances else 0
    global_recall = sum(TP_nums) / sum(ref_nums) if sum(ref_nums) > 0 else 0
    global_precision = sum(TP_nums) / sum(pred_nums) if sum(pred_nums) > 0 else 0
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if global_precision + global_recall != 0 else 0
    
    # Store results
    results['avg_iou'] = avg_iou
    results['global_iou'] = global_iou
    results['avg_distance'] = avg_distance
    results['global_recall'] = global_recall
    results['global_precision'] = global_precision
    results['global_f1'] = global_f1
    results['total_samples'] = len(res)
    results['total_predictions'] = sum(pred_nums)
    results['total_references'] = sum(ref_nums)
    results['total_tp'] = sum(TP_nums)
    
    # Save metrics to JSON
    output_file = args.results_path.replace('.csv', '_metrics.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"BLAST Classification Evaluation Results:")
    print(f"Total samples: {results['total_samples']}")
    print(f"Total predictions: {results['total_predictions']}")
    print(f"Total references: {results['total_references']}")
    print(f"Total true positives: {results['total_tp']}")
    print("=" * 50)
    
    # Print results
    for key, value in results.items():
        if key not in ['total_samples', 'total_predictions', 'total_references', 'total_tp']:
            print(f"{key}: {value:.4f}")
    
    print(f"Metrics saved to: {output_file}")
    return results

if __name__ == "__main__":
    args = argParser.parse_args()
    evaluate_single_grounding(args)

