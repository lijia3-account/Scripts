#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习模型比较脚本（含 AUC 统计学差异检验）
自动检测疾病类型（Preterm_PE 或 Term_PE），并适配阳性标签。
支持通过命令行参数指定输入输出路径。
修改内容：输出每个模型在训练集、内部验证集、外部验证集上的预测结果。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import os
import warnings
import argparse
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
N_BOOTSTRAP = 2000  # Bootstrap 重采样次数
CLINICAL_FEATURES = ['BMI', 'Age', 'Birth_history', 'IVF', 'RPL', 'PMH', 'MAP']

# ============================================================================
# 数据加载和预处理（自动检测疾病类型）
# ============================================================================
def load_and_preprocess(train_path, external_path):
    """
    加载训练集（含 Training 和 EV）和外部验证集。
    自动检测疾病类型（Preterm_PE 或 Term_PE）。
    返回：
        X_train, y_train, X_val, y_val, X_test, y_test,
        train_ids, val_ids, test_ids, feature_names, disease_type, positive_label
    """
    train_df = pd.read_csv(train_path, sep='\t', engine='python')
    if len(train_df.columns) < 10:
        train_df = pd.read_csv(train_path, sep='\s+', engine='python')
    
    all_cols = train_df.columns.tolist()
    id_col = all_cols[0]
    dataset_col = all_cols[1]
    group_col = all_cols[2]
    feature_cols = [c for c in all_cols if c not in [id_col, dataset_col, group_col]]
    
    # 检测疾病类型
    unique_groups = train_df[group_col].unique()
    disease_types = [g for g in unique_groups if g != 'control' and g != 'Group']
    if len(disease_types) == 0:
        raise ValueError("未检测到疾病类型（非 control 的 Group 值）")
    positive_label = disease_types[0]  # 取第一个非 control 值
    print(f"检测到疾病类型: {positive_label}")
    
    missing_clinical = [f for f in CLINICAL_FEATURES if f not in feature_cols]
    if missing_clinical:
        raise ValueError(f"临床特征缺失: {missing_clinical}")
    
    train_mask = train_df[dataset_col] == 'Training'
    val_mask = train_df[dataset_col] == 'Test'
    
    X_train_raw = train_df.loc[train_mask, feature_cols].copy()
    y_train = (train_df.loc[train_mask, group_col] == positive_label).astype(int).values
    train_ids = train_df.loc[train_mask, id_col].values.tolist()
    
    X_val_raw = train_df.loc[val_mask, feature_cols].copy()
    y_val = (train_df.loc[val_mask, group_col] == positive_label).astype(int).values
    val_ids = train_df.loc[val_mask, id_col].values.tolist()
    
    # 外部验证集
    ext_df = pd.read_csv(external_path, sep='\t', engine='python')
    if len(ext_df.columns) < 10:
        ext_df = pd.read_csv(external_path, sep='\s+', engine='python')
    # 查找 Group 列（大小写不敏感）
    group_col_ext = None
    for c in ext_df.columns:
        if c.lower() == 'group':
            group_col_ext = c
            break
    if group_col_ext is None:
        raise ValueError("外部验证集缺少 'Group' 列")
    
    ext_feature_cols = [c for c in feature_cols if c in ext_df.columns]
    X_test_raw = ext_df[ext_feature_cols].copy()
    y_test = (ext_df[group_col_ext] == positive_label).astype(int).values
    test_ids = ext_df.iloc[:, 0].values.tolist()
    
    # 缺失值填充与标准化
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_val_imp = imputer.transform(X_val_raw)
    X_test_imp = imputer.transform(X_test_raw)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imp)
    X_val = scaler.transform(X_val_imp)
    X_test = scaler.transform(X_test_imp)
    
    return (X_train, y_train, X_val, y_val, X_test, y_test,
            train_ids, val_ids, test_ids, feature_cols, positive_label)

# ============================================================================
# Bootstrap AUC 差异检验
# ============================================================================
def bootstrap_auc_diff(y_true, prob1, prob2, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_SEED):
    np.random.seed(random_state)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        y_boot = y_true[idx]
        p1_boot = prob1[idx]
        p2_boot = prob2[idx]
        auc1 = roc_auc_score(y_boot, p1_boot)
        auc2 = roc_auc_score(y_boot, p2_boot)
        diffs.append(auc1 - auc2)
    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    p_value = 2 * min(np.mean(diffs > 0), np.mean(diffs < 0))
    p_value = min(p_value, 1.0)
    return {'mean_diff': mean_diff, 'std_diff': std_diff, 'ci_lower': ci_lower, 'ci_upper': ci_upper, 'p_value': p_value}

# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='机器学习模型比较（含AUC差异检验）')
    parser.add_argument('--data_path', type=str, required=True,
                        help='训练数据文件路径（含Training和Test）')
    parser.add_argument('--external_path', type=str, required=True,
                        help='外部验证集文件路径')
    parser.add_argument('--transformer_result_path', type=str, required=True,
                        help='Transformer外部预测结果文件路径')
    parser.add_argument('--output_dir', type=str, default='./ml_comparison_results',
                        help='输出目录')
    args = parser.parse_args()
    
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("加载数据...")
    (X_train, y_train, X_val, y_val, X_test, y_test,
     train_ids, val_ids, test_ids, feature_names, positive_label) = load_and_preprocess(
        args.data_path, args.external_path
    )
    disease_type = positive_label  # 例如 'Preterm_PE' 或 'Term_PE'
    print(f"训练集: {X_train.shape}, 内部验证集: {X_val.shape}, 外部验证集: {X_test.shape}")
    print(f"外部验证集阳性比例: {y_test.mean():.2%}")
    
    # 定义模型
    models = {
        'LogisticRegression': LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=RANDOM_SEED),
        'ElasticNet': LogisticRegression(penalty='elasticnet', C=1.0, l1_ratio=0.5, solver='saga', max_iter=1000, random_state=RANDOM_SEED),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                     eval_metric='logloss', use_label_encoder=False, random_state=RANDOM_SEED),
        'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                       random_state=RANDOM_SEED),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=RANDOM_SEED),
        'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=0.001,
                             batch_size=32, max_iter=200, random_state=RANDOM_SEED, early_stopping=True,
                             validation_fraction=0.1, n_iter_no_change=10)
    }
    
    # TabNet
    tabnet_params = {'n_d': 8, 'n_a': 8, 'n_steps': 3, 'gamma': 1.3,
                     'lambda_sparse': 1e-4, 'optimizer_fn': torch.optim.Adam,
                     'optimizer_params': dict(lr=2e-2), 'mask_type': 'sparsemax',
                     'scheduler_params': {"step_size":10, "gamma":0.9},
                     'scheduler_fn': torch.optim.lr_scheduler.StepLR, 'epsilon': 1e-15}
    tabnet = TabNetClassifier(**tabnet_params)
    tabnet.fit(X_train.astype(np.float32), y_train,
               eval_set=[(X_val.astype(np.float32), y_val)],
               max_epochs=50, patience=10, batch_size=32, virtual_batch_size=32, drop_last=False)
    
    # 存储所有模型的测试集预测概率（用于ROC和AUC比较）
    predictions = {}
    
    # 训练其他模型并预测
    for name, model in models.items():
        print(f"训练 {name}...")
        if name == 'XGBoost':
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif name == 'LightGBM':
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
        else:
            model.fit(X_train, y_train)
        
        # 预测三个数据集
        proba_train = model.predict_proba(X_train)[:, 1]
        proba_val = model.predict_proba(X_val)[:, 1]
        proba_test = model.predict_proba(X_test)[:, 1]
        
        # 保存训练集结果
        df_train = pd.DataFrame({
            'Sample_ID': train_ids,
            'Group': [positive_label if l == 1 else 'control' for l in y_train],
            'Prediction_Score': proba_train,
            'Dataset': 'Training',
            'Model': name,
            'Disease_Type': disease_type
        })
        train_file = os.path.join(OUTPUT_DIR, f'Training_{name}_{disease_type}_Results.txt')
        df_train.to_csv(train_file, sep='\t', index=False)
        print(f"保存 {name} 训练集预测结果到 {train_file}")
        
        # 保存内部验证集结果
        df_val = pd.DataFrame({
            'Sample_ID': val_ids,
            'Group': [positive_label if l == 1 else 'control' for l in y_val],
            'Prediction_Score': proba_val,
            'Dataset': 'EV',
            'Model': name,
            'Disease_Type': disease_type
        })
        val_file = os.path.join(OUTPUT_DIR, f'Test_{name}_{disease_type}_Results.txt')
        df_val.to_csv(val_file, sep='\t', index=False)
        print(f"保存 {name} 内部验证集预测结果到 {val_file}")
        
        # 保存外部验证集结果
        df_test = pd.DataFrame({
            'Sample_ID': test_ids,
            'Group': [positive_label if l == 1 else 'control' for l in y_test],
            'Prediction_Score': proba_test,
            'Dataset': 'External',
            'Model': name,
            'Disease_Type': disease_type
        })
        test_file = os.path.join(OUTPUT_DIR, f'External_{name}_{disease_type}_Results.txt')
        df_test.to_csv(test_file, sep='\t', index=False)
        print(f"保存 {name} 外部验证集预测结果到 {test_file}")
        
        # 存储测试概率用于后续分析
        predictions[name] = proba_test
        print(f"{name} 所有预测完成")
    
    # TabNet 预测
    tabnet_proba_train = tabnet.predict_proba(X_train.astype(np.float32))[:, 1]
    tabnet_proba_val = tabnet.predict_proba(X_val.astype(np.float32))[:, 1]
    tabnet_proba_test = tabnet.predict_proba(X_test.astype(np.float32))[:, 1]
    name_tab = 'TabNet'
    
    # 保存训练集
    df_train = pd.DataFrame({
        'Sample_ID': train_ids,
        'Group': [positive_label if l == 1 else 'control' for l in y_train],
        'Prediction_Score': tabnet_proba_train,
        'Dataset': 'Training',
        'Model': name_tab,
        'Disease_Type': disease_type
    })
    train_file = os.path.join(OUTPUT_DIR, f'Training_{name_tab}_{disease_type}_Results.txt')
    df_train.to_csv(train_file, sep='\t', index=False)
    print(f"保存 {name_tab} 训练集预测结果到 {train_file}")
    
    # 保存内部验证集
    df_val = pd.DataFrame({
        'Sample_ID': val_ids,
        'Group': [positive_label if l == 1 else 'control' for l in y_val],
        'Prediction_Score': tabnet_proba_val,
        'Dataset': 'EV',
        'Model': name_tab,
        'Disease_Type': disease_type
    })
    val_file = os.path.join(OUTPUT_DIR, f'Test_{name_tab}_{disease_type}_Results.txt')
    df_val.to_csv(val_file, sep='\t', index=False)
    print(f"保存 {name_tab} 内部验证集预测结果到 {val_file}")
    
    # 保存外部验证集
    df_test = pd.DataFrame({
        'Sample_ID': test_ids,
        'Group': [positive_label if l == 1 else 'control' for l in y_test],
        'Prediction_Score': tabnet_proba_test,
        'Dataset': 'External',
        'Model': name_tab,
        'Disease_Type': disease_type
    })
    test_file = os.path.join(OUTPUT_DIR, f'External_{name_tab}_{disease_type}_Results.txt')
    df_test.to_csv(test_file, sep='\t', index=False)
    print(f"保存 {name_tab} 外部验证集预测结果到 {test_file}")
    
    predictions[name_tab] = tabnet_proba_test
    print("TabNet 所有预测完成")
    
    # 读取 Transformer 外部预测结果
    trans_df = pd.read_csv(args.transformer_result_path, sep='\t')
    trans_preds = []
    for sid in test_ids:
        val = trans_df[trans_df['Sample_ID'] == sid]['Prediction_Score'].values
        trans_preds.append(val[0] if len(val) > 0 else 0.5)
    trans_preds = np.array(trans_preds)
    # 保存 Transformer 外部结果（原逻辑）
    df_trans = pd.DataFrame({
        'Sample_ID': test_ids,
        'Group': [positive_label if l == 1 else 'control' for l in y_test],
        'Prediction_Score': trans_preds,
        'Dataset': 'External',
        'Model': 'Transformer',
        'Disease_Type': disease_type
    })
    trans_file = os.path.join(OUTPUT_DIR, f'External_Transformer_{disease_type}_Results.txt')
    df_trans.to_csv(trans_file, sep='\t', index=False)
    print(f"保存 Transformer 外部预测结果到 {trans_file}")
    predictions['Transformer'] = trans_preds
    
    # ROC 曲线（仅外部验证集）
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
    for (name, probs), color in zip(predictions.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=color, label=f'{name} (AUC = {roc_auc_val:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC=0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    plt.title(f'ROC Curves - External Validation Set ({disease_type})', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    roc_file = os.path.join(OUTPUT_DIR, f'ROC_Curves_All_Models_External_{disease_type}.png')
    plt.savefig(roc_file, dpi=300, bbox_inches='tight')
    print(f"ROC曲线保存至 {roc_file}")
    plt.close()
    
    # AUC 差异检验（每个模型 vs Transformer）
    print("\n进行 AUC 差异 Bootstrap 检验（每个模型 vs Transformer）...")
    trans_probs = predictions['Transformer']
    comparison_results = []
    for model_name, probs in predictions.items():
        if model_name == 'Transformer':
            continue
        auc_model = roc_auc_score(y_test, probs)
        auc_trans = roc_auc_score(y_test, trans_probs)
        boot_res = bootstrap_auc_diff(y_test, probs, trans_probs, n_bootstrap=N_BOOTSTRAP)
        comparison_results.append({
            'Model': model_name,
            'AUC_Model': auc_model,
            'AUC_Transformer': auc_trans,
            'AUC_Difference': boot_res['mean_diff'],
            'Std_Error': boot_res['std_diff'],
            'CI_Lower': boot_res['ci_lower'],
            'CI_Upper': boot_res['ci_upper'],
            'P_value': boot_res['p_value']
        })
    comp_df = pd.DataFrame(comparison_results)
    comp_file = os.path.join(OUTPUT_DIR, f'AUC_Comparison_Results_{disease_type}.txt')
    comp_df.to_csv(comp_file, sep='\t', index=False)
    print(f"AUC 比较结果保存至 {comp_file}")
    
    print("\nAUC 比较结果（模型 vs Transformer）:")
    print(comp_df[['Model', 'AUC_Difference', 'CI_Lower', 'CI_Upper', 'P_value']].to_string(index=False))
    print("\n所有分析完成。")

if __name__ == '__main__':
    main()
