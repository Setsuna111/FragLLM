import pandas as pd
import argparse
import re
import numpy as np
from typing import Any, Dict, List
import json
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
argParser = argparse.ArgumentParser()
argParser.add_argument("--results_path", type=str, help="path to save the generated description")
argParser.add_argument("--iou_threshold", type=float, help="iou threshold")
argParser.add_argument("--bert_threshold", type=float, help="bert threshold")
argParser.add_argument("--model_path", type=str, default="/home/djy/projects/Data/HF_models/biobert-large-cased-v1.1", help="path to the prot2text model")
argParser.add_argument("--embedding_path", type=str, default="/home/djy/projects/Documents/FragLLM_git/eval_local/vectors_val_test/VenusX_Act.npz", help="path to the embedding file")
args = argParser.parse_args()

# Load pre-trained model tokenizer and model for evaluation
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModel.from_pretrained(args.model_path)


def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    # Use the mean of the last hidden states as sentence embedding
    sentence_embedding = torch.mean(outputs.last_hidden_state[0], dim=0).detach().numpy()

    return sentence_embedding

def text_similarity_bert(str1, str2):
    emb1 = get_bert_embedding(str1)
    emb2 = get_bert_embedding(str2)
    return cosine_similarity([emb1], [emb2])[0, 0]


# 从保存的embedding和category_names中检索category_name
def get_embedding_from_category_name(pred_label, data):
    feature_vectors = data["feature_vectors"]
    category_names = data["category_names"]
    emb_pred = get_bert_embedding(pred_label)
    # 选择相似性最大的category_name
    similarity_matrix = cosine_similarity([emb_pred], feature_vectors)
    max_similarity_idx = similarity_matrix.argmax()
    max_category_name = category_names[max_similarity_idx]
    return max_category_name

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

# extract the class name from the responses
def extract_class_name(response):
    if " at " in response:
        class_name = response.split(" at ")[0]
    else:
        # 找到re.compile(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)')所在的第一个位置
        pattern = re.compile(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)')
        match = pattern.search(response)
        if match:
            start_idx = match.start()   # 匹配到的起始位置
            if start_idx == 1:
                import pdb; pdb.set_trace()
            print("起始位置:", start_idx)
            class_name = response[:start_idx]
        else:
            # import pdb; pdb.set_trace()
            print("未找到匹配")
            class_name = response
    return class_name

# extract the target from the responses
def extract_target(response):
    # 找到response中第一个“:”的位置
    colon_idx = response.find(":")
    target = response[colon_idx+1:].strip() 
    targets = target.split(";")
    targets_clean = []
    for target in targets:
        if " and " in target and "and (" not in target:
            targets_clean.extend(target.split(" and "))
        else:
            targets_clean.append(target.strip())
    return targets_clean

def match_positions(prediction_positions, reference_positions):
    # Match the positions with position interval distance
    matched_pred_positions = []
    matched_ref_positions = []
    # 计算区间中心点
    pred_centers = [(a[0] + a[1]) / 2.0 for a in prediction_positions]
    ref_centers = [(b[0] + b[1]) / 2.0 for b in reference_positions]
    dist_matrix = np.zeros((len(pred_centers), len(ref_centers)))
    for i, ac in enumerate(pred_centers):
        for j, bc in enumerate(ref_centers):
            dist_matrix[i, j] = abs(ac - bc)
    # Find one-to-one matches
    matched_pred_positions = []
    matched_ref_positions = []
    while (dist_matrix!=1024).any():
        min_dist_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        matched_pred_positions.append(prediction_positions[min_dist_idx[0]])
        matched_ref_positions.append(reference_positions[min_dist_idx[1]])
        dist_matrix[min_dist_idx[0], :] = 1024
        dist_matrix[:, min_dist_idx[1]] = 1024
    return matched_pred_positions, matched_ref_positions

def match_labels_idx(pred_labels, ref_labels, bert_threshold=None):
    matched_pred_labels_idx = []
    matched_ref_labels_idx = []
    matched_berts = []
    bert_matrix = np.zeros((len(pred_labels), len(ref_labels)))
    for i, pred_label in enumerate(pred_labels):
        for j, ref_label in enumerate(ref_labels):
            bert_matrix[i, j] = text_similarity_bert(pred_label, ref_label)
    while (bert_matrix!=0).any():
        min_bert_idx = np.unravel_index(np.argmax(bert_matrix), bert_matrix.shape)
        if (bert_threshold is not None) and (bert_matrix[min_bert_idx[0], min_bert_idx[1]] < bert_threshold):
            break
        matched_pred_labels_idx.append(min_bert_idx[0])
        matched_ref_labels_idx.append(min_bert_idx[1])
        matched_berts.append(bert_matrix[min_bert_idx[0], min_bert_idx[1]])
        bert_matrix[min_bert_idx[0], :] = 0
        bert_matrix[:, min_bert_idx[1]] = 0
    return matched_pred_labels_idx, matched_ref_labels_idx, matched_berts
        

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

# Compute the recall between the positions
def compute_recall(positions_pre, positions_ref, iou_threshold):
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







def evaluate_group_grounding(args: Dict[str, Any]) -> Dict[str, Any]:
    data_npz = np.load(args.embedding_path)
    res = pd.read_csv(args.results_path)
    res = res.drop_duplicates(subset=['dataset_idx'])
    predictions = res['generated'].tolist()
    references = res['reference'].tolist()
    results = {}
    pred_targets_num = []  # TP + FP for class name
    ref_targets_num = [] # TP + FN for class name
    matched_targets_num = [] # TP for class name

    # for positions
    pred_positions_num = []
    ref_positions_num = []
    matched_positions_num = []

    # for iou
    iou_samples = []
    unions_samples = []
    intersections_samples = []
    distances_samples = []
    bert_samples = []
    for (idx, (prediction, reference)) in enumerate(zip(predictions, references)):
        # prediction = re.sub(r'[^\x20-\x7E]', '', prediction)
        # reference.replace("<|reserved_special_token_0|>", "")
        # 提取targets
        pred_targets = extract_target(prediction) # list of strings
        ref_targets = extract_target(reference) # list of strings
        pred_targets_num.append(len(pred_targets))
        ref_targets_num.append(len(ref_targets))
        # 所有目标的预测位置数量和参考位置数量
        pred_positions_num.append(sum([len(extract_position_single(pred_target)) for pred_target in pred_targets]))
        ref_positions_num.append(sum([len(extract_position_single(ref_target)) for ref_target in ref_targets]))
      
        pred_labels = [extract_class_name(pred_target) for pred_target in pred_targets] # list of pred labels
        # import pdb; pdb.set_trace()
        pred_labels = [get_embedding_from_category_name(pred_label, data_npz) for pred_label in pred_labels]
        ref_labels = [extract_class_name(ref_target) for ref_target in ref_targets]
        # 匹配labels, 可能存在预测标签数量小于参考标签数量的情况
        matched_pred_labels_idx, matched_ref_labels_idx, matched_berts = match_labels_idx(pred_labels, ref_labels)
        matched_pred_targets = [pred_targets[i] for i in matched_pred_labels_idx]
        matched_ref_targets = [ref_targets[i] for i in matched_ref_labels_idx]
        matched_targets_num.append(len([i for i in matched_berts if i >= args.bert_threshold]))
        # matched_pred_labels = [pred_labels[i] for i in matched_pred_labels_idx]
        # matched_ref_labels = [ref_labels[i] for i in matched_ref_labels_idx]
        if len(pred_targets) < len(ref_targets): # 少预测的使用None填充
            matched_pred_targets = matched_pred_targets + ["None"] * (len(ref_targets) - len(matched_pred_targets))
            matched_ref_targets = matched_ref_targets + [x for x in ref_targets if x not in matched_ref_targets]
            matched_berts = matched_berts + [0] * (len(ref_targets) - len(matched_berts))

            # matched_pred_labels = matched_pred_labels + ["None"] * (len(ref_targets) - len(matched_pred_labels))
            # matched_ref_labels = matched_ref_labels + [x for x in ref_labels if x not in matched_ref_labels]
            # import pdb; pdb.set_trace()
        # assert matched_targets_num[-1] <= 
        bert_samples.append(sum(matched_berts))
        # 计算每个匹配target的指标
        TP_nums = 0
        unions_target = 0
        intersections_target = 0
        distances_target = 0.0
        for (target_idx, (pred_target, ref_target)) in enumerate(zip(matched_pred_targets, matched_ref_targets)):
            # 提取当前目标的预测位置和参考位置
            prediction_positions = extract_position_single(pred_target)
            reference_positions = extract_position_single(ref_target)
            if len(prediction_positions) < len(reference_positions):
                prediction_positions = prediction_positions + [(0,0)] * (len(reference_positions) - len(prediction_positions))
            matched_pred_positions, matched_ref_positions = match_positions(prediction_positions, reference_positions)
            # 统计iou>0.5的区间数量(ubder bert_score threshold)
            if matched_berts[target_idx] >= args.bert_threshold:
                TP_nums += compute_TP(matched_pred_positions, matched_ref_positions, args.iou_threshold)
            iou, unions, intersections = compute_iou(matched_pred_positions, matched_ref_positions)
            distances_target += compute_distance(matched_pred_positions, matched_ref_positions)
            unions_target += unions
            intersections_target += intersections
        iou_samples.append(intersections_target/unions_target)
        unions_samples.append(unions_target)
        intersections_samples.append(intersections_target)
        matched_positions_num.append(TP_nums)
        distances_samples.append(distances_target/len(matched_pred_positions))
    # 计算指标
    # iou指标
    avg_iou = sum(iou_samples) / len(iou_samples)
    global_iou = sum(intersections_samples) / sum(unions_samples)
    # 类别P，R，F1指标
    global_class_recall = sum(matched_targets_num) / sum(ref_targets_num)
    global_class_precision = sum(matched_targets_num) / sum(pred_targets_num)
    global_class_f1 = 2 * global_class_precision * global_class_recall / (global_class_precision + global_class_recall) if global_class_precision + global_class_recall != 0 else 0
    # 位置距离指标
    avg_distance = sum(distances_samples) / len(distances_samples)
    # bert指标
    avg_bert = sum(bert_samples) / len(bert_samples)
    # 位置P，R，F1指标
    global_position_recall = sum(matched_positions_num) / sum(ref_positions_num)
    global_position_precision = sum(matched_positions_num) / sum(pred_positions_num)
    global_position_f1 = 2 * global_position_precision * global_position_recall / (global_position_precision + global_position_recall) if global_position_precision + global_position_recall != 0 else 0
 
    # 记录结果
    results['avg_iou'] = avg_iou
    results['global_iou'] = global_iou
    results['avg_distance'] = avg_distance
    results['avg_bert'] = avg_bert
    results['global_class_recall'] = global_class_recall
    results['global_class_precision'] = global_class_precision
    results['global_class_f1'] = global_class_f1
    results['global_position_recall'] = global_position_recall
    results['global_position_precision'] = global_position_precision
    results['global_position_f1'] = global_position_f1
    with open(args.results_path.replace('.csv', '_metrics.json'), 'w') as f:
        json.dump(results, f)
    # 逐行打印结果
    for key, value in results.items():
        print(f"{key}: {value}")
    return results

if __name__ == "__main__":
    args = argParser.parse_args()
    evaluate_group_grounding(args)

