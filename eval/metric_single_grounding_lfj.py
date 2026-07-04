import pandas as pd
import argparse
import re
import numpy as np
from typing import Any, Dict, List
import json
argParser = argparse.ArgumentParser()
argParser.add_argument("--results_path", type=str, help="path to save the generated description")
argParser.add_argument("--iou_threshold", type=float, help="iou threshold")

args = argParser.parse_args()


# extract the position from the responses
def extract_position_single(response):
    positions = []
    # pattern = re.compile(r'<reserved_special_token_3>(.*?)<reserved_special_token_4>')  # <reserved_special_token_3>(0,107)<reserved_special_token_4>
    # pattern = re.compile(r'\((.*?)\)')  # <reserved_special_token_3>(0,107)<reserved_special_token_4>
    pattern = re.compile(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)')
    matches = pattern.findall(response)

    for match in matches:
        try:
            # # 移除字符串两端的括号和空格
            # cleaned_match = match.strip().strip('()')
            # # 按逗号分割字符串
            # lon_str, lat_str = cleaned_match.split(',')
            lon_str, lat_str = match[0], match[1]
            # 将字符串转换为浮点数并存为元组
            positions.append((int(lon_str), int(lat_str)))
        except ValueError:
            # 如果转换失败（例如，格式不正确），则跳过此匹配
            print(f"无法将 '{match}' 转换为坐标，已跳过。")
            continue
    return positions # list of strings

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
    if len(positions_pre) != 0:
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
    if len(positions_pre) != 0:
        iou /= len(positions_pre)
    return iou, unions, intersections

def compute_iou_single(pos_pre, pos_ref):
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
    return iou

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


def positions_to_residue_set(positions):
    residues = set()
    for start, end in positions:
        if end < start:
            start, end = end, start
        residues.update(range(start, end + 1))
    return residues


def compute_residue_level_iou(prediction_positions, reference_positions):
    pred_residues = positions_to_residue_set(prediction_positions)
    ref_residues = positions_to_residue_set(reference_positions)
    intersection = len(pred_residues & ref_residues)
    union = len(pred_residues | ref_residues)
    if union == 0:
        return 0, union, intersection
    return intersection / union, union, intersection


def evaluate_single_grounding(args: Dict[str, Any]) -> Dict[str, Any]:
    res = pd.read_csv(args.results_path)
    res = res.drop_duplicates(subset=['dataset_idx'])
    predictions = res['generated'].tolist()
    references = res['reference'].tolist()
    ious = []
    distances = []
    TP_nums = []
    precisions = []
    unions_list = []
    intersections_list = []
    residue_ious = []
    residue_unions_list = []
    residue_intersections_list = []
    pred_nums = []
    ref_nums = []
    results = {}
    for (idx, (prediction, reference)) in enumerate(zip(predictions, references)):
        prediction_positions = extract_position_single(prediction)
        reference_positions = extract_position_single(reference)
        residue_iou, residue_unions, residue_intersections = compute_residue_level_iou(prediction_positions, reference_positions)
        residue_ious.append(residue_iou)
        residue_unions_list.append(residue_unions)
        residue_intersections_list.append(residue_intersections)
        # 记录预测和参考位置的数量
        pred_nums.append(len(prediction_positions)) # TP + FP
        ref_nums.append(len(reference_positions)) # TP + FN
        # 对于预测位置数量小于参考位置数量的情况，用(0,0)填充缺少的预测位置，其他情况直接按照one-t-one匹配
        if len(prediction_positions) < len(reference_positions):
            # 用(0,0)填充缺少的预测位置
            prediction_positions = prediction_positions + [(0,0)] * (len(reference_positions) - len(prediction_positions))
        matched_pred_positions, matched_ref_positions = match_positions(prediction_positions, reference_positions)
        # print(f"------------{idx}------------")
        # print("matched_pred_positions", matched_pred_positions)
        # print("matched_ref_positions", matched_ref_positions)
        # 计算当前样本的miou, unions, intersections
        iou, unions, intersections = compute_iou(matched_pred_positions, matched_ref_positions)
        ious.append(iou)
        unions_list.append(unions)
        intersections_list.append(intersections)
        # 计算当前样本的起止位置距离
        distances.append(compute_distance(matched_pred_positions, matched_ref_positions))
        # 计算当前样本TP数
        TP_nums.append(compute_TP(matched_pred_positions, matched_ref_positions, args.iou_threshold))
    # 计算平均miou, unions, intersections
    # import pdb; pdb.set_trace()
    avg_iou = sum(ious) / len(ious)
    global_iou = sum(intersections_list) / sum(unions_list)
    residue_level_avg_iou = sum(residue_ious) / len(residue_ious)
    residue_level_global_iou = sum(residue_intersections_list) / sum(residue_unions_list) if sum(residue_unions_list) != 0 else 0
    # 计算平均起止位置距离
    avg_distance = sum(distances) / len(distances)
    # 计算总召回率
    global_recall = sum(TP_nums) / sum(ref_nums)
    # 计算总精确率
    global_precision = sum(TP_nums) / sum(pred_nums)
    # 计算总f1值
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if global_precision + global_recall != 0 else 0
    # 记录结果
    results['avg_iou'] = avg_iou
    results['global_iou'] = global_iou
    results['avg_distance'] = avg_distance
    results['global_recall'] = global_recall
    results['global_precision'] = global_precision
    results['global_f1'] = global_f1
    results['interval_level_avg_iou'] = avg_iou
    results['interval_level_global_iou'] = global_iou
    results['interval_level_avg_distance'] = avg_distance
    results['interval_level_global_recall'] = global_recall
    results['interval_level_global_precision'] = global_precision
    results['interval_level_global_f1'] = global_f1
    results['residue_level_avg_iou'] = residue_level_avg_iou
    results['residue_level_global_iou'] = residue_level_global_iou
    with open(args.results_path.replace('.csv', '_metrics.json'), 'w') as f:
        json.dump(results, f)
    print_keys = [
        'interval_level_avg_iou',
        'interval_level_global_iou',
        'interval_level_avg_distance',
        'interval_level_global_recall',
        'interval_level_global_precision',
        'interval_level_global_f1',
        'residue_level_avg_iou',
    ]
    for key in print_keys:
        value = results[key]
        print(f"{key}: {value}")
    return results

if __name__ == "__main__":
    args = argParser.parse_args()
    evaluate_single_grounding(args)
