import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.stats import norm
import os
import warnings
import argparse
warnings.filterwarnings('ignore')

# ================== 配置（通过命令行参数传入） ==================
def parse_args():
    parser = argparse.ArgumentParser(description='分析模型预测结果并绘制ROC曲线')
    parser.add_argument('--sample_info_file', type=str, 
                        default='/HOME/szfy_whlxy/szfy_whlxy_1/AI/Transformer/all_samples_gw_sampling.txt',
                        help='样本信息文件路径')
    parser.add_argument('--results_dir', type=str,
                        default='/HOME/szfy_whlxy/szfy_whlxy_1/AI/Transformer/results/ptpe',
                        help='预测结果目录路径')
    parser.add_argument('--disease', type=str, default='Preterm_PE',
                        choices=['Preterm_PE', 'Term_PE'],
                        help='疾病类型: Preterm_PE 或 Term_PE')
    parser.add_argument('--target_specificity', type=float, default=0.92,
                        help='目标特异度阈值，例如 0.90, 0.92, 0.95 等')
    return parser.parse_args()

args = parse_args()

base_dir = "/HOME/szfy_whlxy/szfy_whlxy_1/AI/Transformer"
sample_info_file = args.sample_info_file
results_dir = args.results_dir
disease = args.disease
target_spec = args.target_specificity

print(f"样本信息文件: {sample_info_file}")
print(f"结果目录: {results_dir}")
print(f"疾病类型: {disease}")
print(f"目标特异度: {target_spec*100:.0f}%")

models = ["Clinical-Factors", "Metabolomics", "Proteomics", "Multi-Modal-Fusion"]
datasets = ["Training", "Validation", "External"]

# ================== 1. 读取样本信息，筛选 GA < 16 ==================
sample_df = pd.read_csv(sample_info_file, sep='\t')
sample_df = sample_df[sample_df['GA_sampling'] < 16].copy()
print(f"筛选后样本总数: {len(sample_df)}")
print("各数据集样本数：")
print(sample_df['DATASET'].value_counts())

# ================== 2. 读取预测结果（适配 External 文件名） ==================
def load_predictions(model, dataset, disease):
    if dataset == "Training":
        file_name = f"{model}_Training_{disease}_Results.txt"
    elif dataset == "Validation":
        file_name = f"{model}_Validation_{disease}_Results.txt"
    elif dataset == "External":
        file_name = f"External_{model}_{disease}_Results.txt"
    else:
        raise ValueError(f"未知数据集: {dataset}")
    file_path = os.path.join(results_dir, file_name)
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在，跳过")
        return None
    df = pd.read_csv(file_path, sep='\t')
    df.rename(columns={'Sample_ID': 'ID'}, inplace=True)
    # 删除预测文件中的Group列，避免与sample_df的Group列冲突
    if 'Group' in df.columns:
        df = df.drop(columns=['Group'])
    return df

pred_dfs = {}
for model in models:
    for dataset in datasets:
        key = (model, dataset)
        df = load_predictions(model, dataset, disease)
        if df is not None:
            pred_dfs[key] = df

# ================== 3. 合并样本信息与预测结果 ==================
merged = {}
for model in models:
    for dataset in datasets:
        key = (model, dataset)
        if key not in pred_dfs:
            continue
        pred_df = pred_dfs[key]
        # 此时pred_df已经没有Group列，合并时不会产生冲突
        merged_df = pd.merge(sample_df, pred_df, on='ID', how='inner')
        merged[key] = merged_df

# ================== 4. 绘制 ROC 曲线（仅保留验证集+外部验证集合集） ==================
fig, ax2 = plt.subplots(1, 1, figsize=(8, 8))

for model in models:
    valid_dfs = []
    val_key = (model, "Validation")
    ext_key = (model, "External")
    if val_key in merged:
        valid_dfs.append(merged[val_key])
    if ext_key in merged:
        valid_dfs.append(merged[ext_key])
    if not valid_dfs:
        continue
    df_combined = pd.concat(valid_dfs, ignore_index=True).drop_duplicates(subset='ID')
    y_true = (df_combined['Group'] == disease).astype(int)
    y_score = df_combined['Prediction_Score']
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, label=f'{model} (AUC={roc_auc:.3f})', linewidth=2.5)
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=16)
ax2.set_ylabel('True Positive Rate (Sensitivity)', fontsize=16)
ax2.set_title('Validation+External ROC Curves', fontsize=16, fontweight='bold')
ax2.legend(loc='lower right', fontsize=14)
ax2.grid(alpha=0.3)
ax2.tick_params(labelsize=14)

plt.tight_layout()
roc_curves_file = f'ROC_curves_{disease}.png'
plt.savefig(roc_curves_file, dpi=300, bbox_inches='tight')
plt.show()

# ================== 阈值选择策略 ==================
threshold_method = 'spec_target'  # 改为使用目标特异度
manual_threshold = 0.5

model_for_threshold = "Multi-Modal-Fusion"
key_train = (model_for_threshold, "Training")
if key_train not in merged:
    raise ValueError(f"训练集 {model_for_threshold} 结果缺失，无法确定阈值")
df_train = merged[key_train]
y_true_train = (df_train['Group'] == disease).astype(int)
y_score_train = df_train['Prediction_Score']

fpr_train, tpr_train, thresholds = roc_curve(y_true_train, y_score_train)

# 计算目标特异度对应的阈值（目标特异度 = 1 - FPR）
if threshold_method == 'spec_target':
    target_fpr = 1 - target_spec  # 例如 target_spec=0.92, target_fpr=0.08
    idx = np.argmin(np.abs(fpr_train - target_fpr))
    threshold = thresholds[idx]
    print(f"基于训练集特异度{target_spec*100:.0f}%的阈值: {threshold:.6f}")
elif threshold_method == 'youden':
    youden = tpr_train - fpr_train
    idx = np.argmax(youden)
    threshold = thresholds[idx]
    print(f"基于训练集约登指数的阈值: {threshold:.6f}")
elif threshold_method == 'manual':
    threshold = manual_threshold
    print(f"手动指定阈值: {threshold:.6f}")
else:
    raise ValueError("threshold_method 必须是 'spec_target', 'youden' 或 'manual'")

# ================== 6. 在合并验证集上评估性能 ==================
valid_dfs = []
val_key = (model_for_threshold, "Validation")
ext_key = (model_for_threshold, "External")
if val_key in merged:
    valid_dfs.append(merged[val_key])
if ext_key in merged:
    valid_dfs.append(merged[ext_key])
if not valid_dfs:
    raise ValueError("没有验证集数据，无法评估")
df_valid = pd.concat(valid_dfs, ignore_index=True).drop_duplicates(subset='ID')

y_true_valid = (df_valid['Group'] == disease).astype(int)
y_score_valid = df_valid['Prediction_Score']
y_pred_valid = (y_score_valid >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_true_valid, y_pred_valid).ravel()
sensitivity = tp / (tp + fn) if (tp+fn)>0 else 0
specificity = tn / (tn + fp) if (tn+fp)>0 else 0
ppv = tp / (tp + fp) if (tp+fp)>0 else 0
npv = tn / (tn + fn) if (tn+fn)>0 else 0

def wilson_ci(count, n, alpha=0.05):
    if n == 0:
        return (0, 0)
    p = count / n
    z = norm.ppf(1 - alpha/2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    half_width = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (center - half_width, center + half_width)

ci_sens = wilson_ci(tp, tp+fn)
ci_spec = wilson_ci(tn, tn+fp)
ci_ppv = wilson_ci(tp, tp+fp)
ci_npv = wilson_ci(tn, tn+fn)

print(f"\n===== 合并验证集性能 (阈值={threshold:.6f}, 目标特异度={target_spec*100:.0f}%) =====")
print(f"灵敏度: {sensitivity:.3f} (95%CI: {ci_sens[0]:.3f}-{ci_sens[1]:.3f})")
print(f"特异度: {specificity:.3f} (95%CI: {ci_spec[0]:.3f}-{ci_spec[1]:.3f})")
print(f"阳性预测值: {ppv:.3f} (95%CI: {ci_ppv[0]:.3f}-{ci_ppv[1]:.3f})")
print(f"阴性预测值: {npv:.3f} (95%CI: {ci_npv[0]:.3f}-{ci_npv[1]:.3f})")

# ================== 定义评估函数（直接接收数组） ==================
def evaluate_threshold(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    ppv = tp/(tp+fp) if (tp+fp)>0 else 0
    npv = tn/(tn+fn) if (tn+fn)>0 else 0
    return sens, spec, ppv, npv

# ================== 7. 绘制四格表（目标特异度阈值） ==================
conf_matrix = np.array([[tn, fp], [fn, tp]])
fig_cm, ax_cm = plt.subplots(figsize=(8, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Control', disease],
            yticklabels=['Control', disease],
            annot_kws={'size': 15},
            ax=ax_cm)
ax_cm.set_xlabel('Predicted', fontsize=15)
ax_cm.set_ylabel('True', fontsize=15)
ax_cm.set_title(f'Confusion Matrix (Threshold={threshold:.3f}, Specificity={target_spec*100:.0f}%)\nValidation+External', 
                fontsize=17, fontweight='bold')
ax_cm.tick_params(labelsize=14)
ax_cm.set_aspect('equal')
plt.tight_layout()
conf_matrix_file = f'Confusion_Matrix_Spec{int(target_spec*100)}_{disease}.png'
plt.savefig(conf_matrix_file, dpi=300, bbox_inches='tight')
plt.show()

# ================== 7b. 绘制四格表（带百分比） ==================
# 计算混淆矩阵各元素
n_control = np.sum(y_true_valid == 0)
n_case = np.sum(y_true_valid == 1)

# 创建带标签的混淆矩阵
conf_matrix = np.array([[tn, fp], [fn, tp]])

# 绘制热力图
fig_cm, ax_cm = plt.subplots(figsize=(8, 8))

# 自定义annotations，显示数量和百分比
annot_array = np.empty_like(conf_matrix, dtype=object)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        if i == 0 and j == 0:  # TN
            total = n_control
            pct = conf_matrix[i, j] / total * 100 if total > 0 else 0
            annot_array[i, j] = f'{conf_matrix[i, j]}\n({pct:.1f}%)'
        elif i == 0 and j == 1:  # FP
            total = n_control
            pct = conf_matrix[i, j] / total * 100 if total > 0 else 0
            annot_array[i, j] = f'{conf_matrix[i, j]}\n({pct:.1f}%)'
        elif i == 1 and j == 0:  # FN
            total = n_case
            pct = conf_matrix[i, j] / total * 100 if total > 0 else 0
            annot_array[i, j] = f'{conf_matrix[i, j]}\n({pct:.1f}%)'
        else:  # TP
            total = n_case
            pct = conf_matrix[i, j] / total * 100 if total > 0 else 0
            annot_array[i, j] = f'{conf_matrix[i, j]}\n({pct:.1f}%)'

# 创建热力图
sns.heatmap(conf_matrix, annot=annot_array, fmt='', cmap='Blues', cbar=False,
            xticklabels=['Control', disease],
            yticklabels=['Control', disease],
            annot_kws={'size': 13, 'weight': 'bold'},
            ax=ax_cm,
            linewidths=2,
            linecolor='white')

# 添加标题和标签
ax_cm.set_xlabel('Predicted', fontsize=16, fontweight='bold')
ax_cm.set_ylabel('True', fontsize=16, fontweight='bold')
ax_cm.set_title(f'Confusion Matrix (Training Specificity={target_spec*100:.0f}%, Threshold={threshold:.4f})\nValidation+External (N={len(df_valid)})',
                fontsize=15, fontweight='bold', pad=20)
ax_cm.tick_params(labelsize=14)

# 添加性能指标文本
metrics_text = f'Sensitivity: {sensitivity:.3f} (95%CI: {ci_sens[0]:.3f}-{ci_sens[1]:.3f})\n' \
               f'Specificity: {specificity:.3f} (95%CI: {ci_spec[0]:.3f}-{ci_spec[1]:.3f})\n' \
               f'PPV: {ppv:.3f} (95%CI: {ci_ppv[0]:.3f}-{ci_ppv[1]:.3f})\n' \
               f'NPV: {npv:.3f} (95%CI: {ci_npv[0]:.3f}-{ci_npv[1]:.3f})'

plt.text(1.25, 0.5, metrics_text, transform=ax_cm.transAxes,
         fontsize=12, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
conf_matrix_pct_file = f'Confusion_Matrix_Spec{int(target_spec*100)}_withPct_{disease}.png'
plt.savefig(conf_matrix_pct_file, dpi=300, bbox_inches='tight')
plt.show()

# ================== 8. 保存性能指标 ==================
performance_df = pd.DataFrame({
    'Metric': ['Sensitivity', 'Specificity', 'PPV', 'NPV'],
    'Estimate': [sensitivity, specificity, ppv, npv],
    'Lower_CI': [ci_sens[0], ci_spec[0], ci_ppv[0], ci_npv[0]],
    'Upper_CI': [ci_sens[1], ci_spec[1], ci_ppv[1], ci_npv[1]]
})
performance_file = f'Validation_Performance_Spec{int(target_spec*100)}_{disease}.csv'
performance_df.to_csv(performance_file, index=False)
print(f"\n性能指标已保存至: {performance_file}")

# ================== 9. 输出多模态模型所有数据集的预测结果（GA<16，左连接） ==================
model_out = "Multi-Modal-Fusion"
all_pred = []
for dataset in ["Training", "Validation", "External"]:
    key = (model_out, dataset)
    if key in pred_dfs:
        df_pred = pred_dfs[key].copy()
        all_pred.append(df_pred[['ID', 'Prediction_Score']])
if all_pred:
    pred_all = pd.concat(all_pred, ignore_index=True).drop_duplicates(subset='ID')
else:
    pred_all = pd.DataFrame(columns=['ID', 'Prediction_Score'])

out_df = sample_df.merge(pred_all, on='ID', how='left')
out_df.rename(columns={'DATASET': 'Dataset'}, inplace=True)
out_df.sort_values(['Dataset', 'ID'], inplace=True)
output_file = f'Multi-Modal-Fusion_All_Datasets_GA_lt16_{disease}.txt'
out_df[['ID', 'Dataset', 'Group', 'Prediction_Score']].to_csv(output_file, sep='\t', index=False)
print(f"\n多模态模型在三个数据集（GA<16）的全部样本预测结果已保存至: {output_file}")
print(f"总行数: {len(out_df)}，各数据集计数：\n{out_df['Dataset'].value_counts()}")

# ================== 10. 额外输出：阈值信息 ==================
threshold_info = pd.DataFrame({
    'Parameter': ['Target_Specificity', 'Threshold', 'Actual_Specificity', 'Sensitivity', 'PPV', 'NPV'],
    'Value': [f'{target_spec*100:.0f}%', f'{threshold:.6f}', f'{specificity:.3f}', 
              f'{sensitivity:.3f}', f'{ppv:.3f}', f'{npv:.3f}']
})
threshold_file = f'Threshold_Info_Spec{int(target_spec*100)}_{disease}.txt'
threshold_info.to_csv(threshold_file, sep='\t', index=False)
print(f"阈值信息已保存至: {threshold_file}")
