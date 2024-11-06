import json
import logging
import os
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from ast_operations import Ast_parser
from ast_operations import get_attributes, get_operations

from qchecker import Qchecker

def read_jsonl(file_path):
    """
    读取指定路径的 jsonl 文件，并返回其中的 code 和 label 字段。
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                record = json.loads(line.strip())
                code = record.get("code")
                label = record.get("label")
                data.append({"code": code, "label": label})
    except FileNotFoundError:
        print(f"文件未找到：{file_path}")
    except json.JSONDecodeError:
        print(f"文件格式错误，请检查文件内容：{file_path}")

    return data

def run_qchecker_on_code(code):
    """
    解析单个 code，运行 Qchecker 并返回结果。
    """
    file_lins = code.split('\n')
    astparser = Ast_parser()
    root = astparser.parser(code)

    # 提取变量赋值和函数调用
    assign_list = astparser.extract_variable_assign()
    call_list = astparser.extract_function_calls()

    attributes, att_line_numbers = get_attributes(assign_list)
    operations, opt_line_numbers = get_operations(call_list)

    # 初始化 Qchecker 并生成报告
    qc = Qchecker()
    qc.check(attributes, att_line_numbers, operations, opt_line_numbers, file_lins)
    qc.get_report()
    return qc.defect

def recompute_metrics(file_path):
    """
    读取保存的标签数据，转换标签，并重新计算各项指标。
    """
    try:
        true_labels = []
        predicted_labels = []
        data = read_jsonl(file_path)

        # 逐个处理每个 code
        for i, item in enumerate(data):
            code = item["code"]
            label = item["label"]
            true_labels.append(1-label)
            try:
                pred = run_qchecker_on_code(code)
                predicted_labels.append(pred)
            except Exception as e:
                predicted_labels.append(label)


        # 重新计算指标
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        auc = roc_auc_score(true_labels, predicted_labels, average='weighted')

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc
        }

        # 输出新的指标
        print("Recomputed Metrics with Qchecker Labels:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {100 * value:.2f}%")

        return metrics

    except Exception as e:
        print(f"Error in recomputing metrics: {e}")
        return None

# 调用示例
# file_path = '../data{time}/csv/results{round_turn}.csv' # 用实际路径替换
# recompute_metrics(file_path)

logger = logging.getLogger(__name__)

num_times = 10
num_folds = 4
accuracys_on_model = []
precisions_on_model = []
recalls_on_model =[]
f1s_on_model =[]
aucs_on_model =[]
for time in range(num_times):
    logger.info(f"Starting qchecker time {time + 1}/{num_times}")
    prefix_time = f'../data{time}'
    split_output_dir = os.path.join(prefix_time,f'fold')
    accuracy = []
    precision = []
    recall =[]
    f1 =[]
    auc =[]

    for fold in range(num_folds):
        logger.info(f"Starting qchecker fold {fold + 1}/{num_folds}")

        file_path = f'../data{time}/{time}test.jsonl' # 用实际路径替换
        metric_one = recompute_metrics(file_path)

        # 将每个折叠的指标添加到列表中
        accuracy.append(metric_one['accuracy'])
        precision.append(metric_one['precision'])
        recall.append(metric_one['recall'])
        f1.append(metric_one['f1_score'])
        auc.append(metric_one['roc_auc'])
        if not os.path.exists(f'../Qchecker'):
            os.makedirs(f'../Qchecker')
        metric_one_df = pd.DataFrame(list(metric_one.items()), columns=['Metric', 'Value'])
        metric_one_df.to_csv(f'../Qchecker/{time}Qchecker{fold}.csv', index=False)

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Value': [sum(accuracy) / num_folds, sum(precision) / num_folds, sum(recall) / num_folds, sum(f1) / num_folds, sum(auc) / num_folds]
    })

    metrics_df.to_csv(f'../Qchecker/{time}qchecker_metrics_on_average.csv', index=False)
    accuracys_on_model.append(sum(accuracy) / num_folds)
    precisions_on_model.append(sum(precision) / num_folds)
    recalls_on_model.append(sum(recall) / num_folds)
    f1s_on_model.append(sum(f1) / num_folds)
    aucs_on_model.append(sum(auc) / num_folds)
accuracy_mean = sum(accuracys_on_model) / num_times
precision_mean = sum(precisions_on_model) / num_times
recall_mean = sum(recalls_on_model) / num_times
f1_mean = sum(f1s_on_model) / num_times
auc_mean = sum(aucs_on_model) / num_times
metrics_on_model_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
    'Value': [accuracy_mean,precision_mean,recall_mean,f1_mean,auc_mean]
})
metrics_on_model_df.to_csv(f'../qchecker_metrics_on_average_tot.csv', index=False)