#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTPE Transformer 模型 - 增强版（完整回应审稿人意见）
整合类别不平衡处理、校准评估、DCA、SHAP 稳定性分析及外部验证集评估
基于 pe_transformer_final.py 修改，保持训练过程主体不变
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
import matplotlib.pyplot as plt
import warnings
import argparse
import os
warnings.filterwarnings('ignore')

# 尝试导入 shap
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP 库未安装，将使用 Gradient-based 重要性分析作为替代")

# ============================================================================
# 配置参数（新增类别不平衡处理开关）
# ============================================================================
class Config:
    DATA_PATH = '/HOME/szfy_whlxy/szfy_whlxy_1/AI/data/PTPE_control_all_data.txt'
    EXTERNAL_PATH = None
    OUTPUT_DIR = './output'
    ID_COL = 0
    DATASET_COL = 1
    GROUP_COL = 2
    
    # 组学特征文件路径（通过 add_argument 传入）
    HIGH_RISK_FILE = None
    METABOLOMICS_FILE = None
    PROTEOMICS_FILE = None
    
    # 特征列名列表（从文件读取）
    HIGH_RISK_COLS = None
    METABOLOMICS_COLS = None
    PROTEOMICS_COLS = None

    H_HIGH_RISK = 32
    H_METAB = 64
    H_PROT = 32
    HIDDEN_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 2
    DROPOUT = 0.4
    MAX_LEN = 150

    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 20

    # 类别不平衡处理选项（参考 ptpe_transformer_final_revised.py）
    USE_CLASS_WEIGHT = True       # 使用 BCEWithLogitsLoss 的 pos_weight
    USE_FOCAL_LOSS = False        # 备选 Focal Loss（暂不启用，保持简单）
    USE_RESAMPLING = False        # 备选重采样（暂不启用）
    FOCAL_LOSS_GAMMA = 2.0

    SHAP_BACKGROUND_SAMPLES = 100
    SHAP_EVAL_SAMPLES = 50
    SHAP_NSAMPLES = 200

    CALIBRATION_THRESHOLD = 0.9
    CALIBRATION_N_BINS = 10
    DCA_THRESHOLD_RANGE = np.arange(0.01, 0.5, 0.01)
    SHAP_N_BACKGROUND = 50
    SHAP_N_BOOTSTRAP = 50
    SHAP_STABILITY_THRESHOLD = 0.7

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RANDOM_SEED = 42

def wilson_ci(n, successes, alpha=0.05):
    """
    计算二项比例的 Wilson 置信区间 (95% CI)
    参数:
        n: 总样本数
        successes: 成功次数
        alpha: 显著性水平 (默认0.05)
    返回:
        (lower, upper) 置信区间下界和上界
    """
    if n == 0:
        return (0.0, 0.0)
    p_hat = successes / n
    z = 1.96  # 95% 正态分位数
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    half_width = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / denom
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return lower, upper

# ============================================================================
# 加载特征列名文件
# ============================================================================
def load_feature_columns(file_path):
    """
    从文件中读取特征列名，文件每行一个特征名
    返回: 特征名列表
    """
    if file_path is None or not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        feature_names = [line.strip() for line in f if line.strip()]
    
    print(f"  从 {file_path} 加载了 {len(feature_names)} 个特征")
    return feature_names

def get_column_indices(df, feature_names, feature_type_name):
    """
    根据特征名列表，在DataFrame的列名中查找对应的列索引
    返回: 列索引列表
    """
    all_columns = list(df.columns)
    indices = []
    missing = []
    
    for name in feature_names:
        if name in all_columns:
            indices.append(all_columns.index(name))
        else:
            missing.append(name)
    
    if missing:
        print(f"  ⚠️ 警告: {feature_type_name} 中以下特征未在数据文件中找到:")
        for m in missing[:10]:  # 只显示前10个
            print(f"    - {m}")
        if len(missing) > 10:
            print(f"    ... 等共 {len(missing)} 个特征未找到")
    
    if not indices:
        raise ValueError(f"{feature_type_name}: 没有匹配到任何特征列，请检查特征名文件是否正确")
    
    return indices

# ============================================================================
# Focal Loss 实现（参考 revised 版本）
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha[1], self.alpha[0])
            focal_weight = alpha_t * focal_weight
        loss = focal_weight * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ============================================================================
# 数据集类（训练/验证）- 增加类别权重计算
# ============================================================================
class PTPEDataset(Dataset):
    def __init__(self, data_path, dataset_type='Training', config=Config()):
        self.df = pd.read_csv(data_path, sep='\t', engine='python')
        if len(self.df.columns) < 10:
            self.df = pd.read_csv(data_path, sep='\s+', engine='python')

        self.all_columns = list(self.df.columns)
        self.all_sample_ids = self.df.iloc[:, config.ID_COL].values
        mask = self.df.iloc[:, config.DATASET_COL] == dataset_type
        self.sample_ids = self.all_sample_ids[mask]
        self.df = self.df[mask].reset_index(drop=True)

        group_col = self.df.iloc[:, config.GROUP_COL]
        if 'Preterm_PE' in group_col.values:
            self.disease_type = 'Preterm_PE'
            self.positive_label = 'Preterm_PE'
        elif 'Term_PE' in group_col.values:
            self.disease_type = 'Term_PE'
            self.positive_label = 'Term_PE'
        else:
            raise ValueError("Unknown disease type.")

        self.labels = (self.df.iloc[:, config.GROUP_COL] == self.positive_label).astype(int).values
        self.config = config
        self.dataset_type = dataset_type

        # 根据特征文件加载对应的列索引
        self.high_risk, self.high_risk_mean, self.high_risk_std = self._standardize(
            self.df.iloc[:, config.HIGH_RISK_COLS].values
        )
        self.metabolomics, self.metab_mean, self.metab_std = self._standardize(
            self.df.iloc[:, config.METABOLOMICS_COLS].values
        )
        self.proteomics, self.prot_mean, self.prot_std = self._standardize(
            self.df.iloc[:, config.PROTEOMICS_COLS].values
        )

        self.high_risk_names = [self.all_columns[i] for i in config.HIGH_RISK_COLS]
        self.metabolomics_names = [self.all_columns[i] for i in config.METABOLOMICS_COLS]
        self.proteomics_names = [self.all_columns[i] for i in config.PROTEOMICS_COLS]

        # 计算类别权重（用于损失函数和采样）
        self.class_weights = self._compute_class_weights()
        self.sample_weights = self._compute_sample_weights()

        print(f"{dataset_type} 数据集大小：{len(self)} 样本")
        print(f"  - 疾病类型：{self.disease_type}")
        print(f"  - 阳性样本数：{self.labels.sum()}，阴性样本数：{len(self) - self.labels.sum()}")
        print(f"  - 阳性样本比例：{self.labels.mean():.2%}")
        print(f"  - 类别权重（负:正）：{self.class_weights[0]:.3f}:{self.class_weights[1]:.3f}")
        print(f"  - 高危因素特征维度：{self.high_risk.shape[1]}")
        print(f"  - 代谢组特征维度：{self.metabolomics.shape[1]}")
        print(f"  - 蛋白组特征维度：{self.proteomics.shape[1]}")

    def get_sample_ids(self):
        return self.sample_ids.tolist()

    def _standardize(self, X):
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True)
        std[std == 0] = 1
        return (X - mean) / std, mean.flatten(), std.flatten()

    def _compute_class_weights(self):
        n_total = len(self.labels)
        n_pos = self.labels.sum()
        n_neg = n_total - n_pos
        if n_pos == 0 or n_neg == 0:
            return torch.FloatTensor([1.0, 1.0])
        weight_neg = n_total / (2.0 * n_neg)
        weight_pos = n_total / (2.0 * n_pos)
        return torch.FloatTensor([weight_neg, weight_pos])

    def _compute_sample_weights(self):
        class_weights = self.class_weights.numpy()
        sample_weights = np.where(self.labels == 1, class_weights[1], class_weights[0])
        return torch.FloatTensor(sample_weights)

    def get_sampler(self):
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self),
            replacement=True
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.high_risk[idx]),
            torch.FloatTensor(self.metabolomics[idx]),
            torch.FloatTensor(self.proteomics[idx]),
            torch.FloatTensor([self.labels[idx]])
        )

    def get_combined_features(self):
        return np.hstack([self.high_risk, self.metabolomics, self.proteomics])

    def get_all_feature_names(self):
        return self.high_risk_names + self.metabolomics_names + self.proteomics_names

# ============================================================================
# 外部验证集数据集（使用中位数填充 NA）
# ============================================================================
class ExternalValidationDataset(Dataset):
    def __init__(self, data_path, train_stats, disease_type, positive_label, config=Config()):
        self.df = pd.read_csv(data_path, sep='\t', engine='python')
        if len(self.df.columns) < 10:
            self.df = pd.read_csv(data_path, sep='\s+', engine='python')

        # 确定 group 列（外部验证集列名已统一为 'Group'）
        group_col = None
        for col in self.df.columns:
            if col.lower() == 'group':
                group_col = col
                break
        if group_col is None:
            raise ValueError("External file missing 'group' column")
        group_vals = self.df[group_col].values
        self.labels = (group_vals == positive_label).astype(int)

        # 高危因素列名（从训练统计量获取）
        high_risk_cols = train_stats['high_risk_names']
        # 检查列是否存在
        missing = [c for c in high_risk_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"外部验证集缺少高危因素列: {missing}")
        high_risk_df = self.df[high_risk_cols].copy()
        # 用各列中位数填充 NA
        high_risk_df = high_risk_df.fillna(high_risk_df.median())
        high_risk_raw = high_risk_df.values.astype(float)

        # 代谢物和蛋白列名（从训练统计量获取）
        metab_cols = train_stats['metab_names']
        prot_cols = train_stats['prot_names']
        # 检查列是否存在
        missing_metab = [c for c in metab_cols if c not in self.df.columns]
        if missing_metab:
            raise ValueError(f"外部验证集缺少代谢物列: {missing_metab[:5]}...")
        missing_prot = [c for c in prot_cols if c not in self.df.columns]
        if missing_prot:
            raise ValueError(f"外部验证集缺少蛋白列: {missing_prot[:5]}...")

        metab_df = self.df[metab_cols].copy()
        metab_df = metab_df.fillna(metab_df.median())
        metab_raw = metab_df.values.astype(float)

        prot_df = self.df[prot_cols].copy()
        prot_df = prot_df.fillna(prot_df.median())
        prot_raw = prot_df.values.astype(float)

        # 应用训练集的标准化参数（均值、标准差）
        self.high_risk = (high_risk_raw - train_stats['high_risk_mean']) / train_stats['high_risk_std']
        self.metabolomics = (metab_raw - train_stats['metab_mean']) / train_stats['metab_std']
        self.proteomics = (prot_raw - train_stats['prot_mean']) / train_stats['prot_std']

        # 处理潜在的 NaN（例如训练集标准差为0或外部验证集特征全为缺失）
        self.high_risk = np.nan_to_num(self.high_risk, nan=0.0)
        self.metabolomics = np.nan_to_num(self.metabolomics, nan=0.0)
        self.proteomics = np.nan_to_num(self.proteomics, nan=0.0)

        self.config = config
        self.dataset_type = 'External'
        self.disease_type = disease_type
        self.positive_label = positive_label
        self.feature_names = train_stats['feature_names']

        print(f"外部验证集加载完成，样本数：{len(self)}")
        print(f"  - 阳性样本比例：{self.labels.mean():.2%}")
        print(f"  - 已使用各特征中位数填充缺失值")

    def get_sample_ids(self):
        # 第一列为 ID
        return self.df.iloc[:, 0].values.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.high_risk[idx]),
            torch.FloatTensor(self.metabolomics[idx]),
            torch.FloatTensor(self.proteomics[idx]),
            torch.FloatTensor([self.labels[idx]])
        )

# ============================================================================
# 模型组件（原样保留）
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 4, dropout=dropout, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.pos_encoder(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        cls_output = x[:, 0]
        return self.norm(cls_output)

class MultiModalTransformer(nn.Module):
    def __init__(self, config=Config()):
        super(MultiModalTransformer, self).__init__()
        self.config = config
        self.high_risk_encoder = TransformerEncoder(
            input_dim=len(config.HIGH_RISK_COLS), d_model=config.H_HIGH_RISK,
            num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS, dropout=config.DROPOUT
        )
        self.metabolomics_encoder = TransformerEncoder(
            input_dim=len(config.METABOLOMICS_COLS), d_model=config.H_METAB,
            num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS, dropout=config.DROPOUT
        )
        self.proteomics_encoder = TransformerEncoder(
            input_dim=len(config.PROTEOMICS_COLS), d_model=config.H_PROT,
            num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS, dropout=config.DROPOUT
        )
        self.fusion_dim = config.H_HIGH_RISK + config.H_METAB + config.H_PROT
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(32, 1)
        )

    def forward(self, x_high_risk, x_metab, x_prot, mode='full'):
        if mode == 'high_risk':
            feat_h = self.high_risk_encoder(x_high_risk)
            feat_m = torch.zeros(x_high_risk.size(0), self.config.H_METAB, device=x_high_risk.device)
            feat_p = torch.zeros(x_high_risk.size(0), self.config.H_PROT, device=x_high_risk.device)
        elif mode == 'metab':
            feat_h = torch.zeros(x_high_risk.size(0), self.config.H_HIGH_RISK, device=x_high_risk.device)
            feat_m = self.metabolomics_encoder(x_metab)
            feat_p = torch.zeros(x_high_risk.size(0), self.config.H_PROT, device=x_high_risk.device)
        elif mode == 'prot':
            feat_h = torch.zeros(x_high_risk.size(0), self.config.H_HIGH_RISK, device=x_high_risk.device)
            feat_m = torch.zeros(x_high_risk.size(0), self.config.H_METAB, device=x_high_risk.device)
            feat_p = self.proteomics_encoder(x_prot)
        else:
            feat_h = self.high_risk_encoder(x_high_risk)
            feat_m = self.metabolomics_encoder(x_metab)
            feat_p = self.proteomics_encoder(x_prot)
        fused = torch.cat([feat_h, feat_m, feat_p], dim=1)
        logits = self.classifier(fused)
        return logits

    def predict_proba(self, x_high_risk, x_metab, x_prot):
        logits = self.forward(x_high_risk, x_metab, x_prot, mode='full')
        return torch.sigmoid(logits)

# ============================================================================
# 训练和验证函数（原样保留）
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device, mode='full'):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for batch_idx, (x_h, x_m, x_p, y) in enumerate(dataloader):
        x_h = x_h.to(device)
        x_m = x_m.to(device)
        x_p = x_p.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(x_h, x_m, x_p, mode=mode)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten())
        all_labels.extend(y.detach().cpu().numpy().flatten())
    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_labels

def validate(model, dataloader, criterion, device, mode='full'):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_h, x_m, x_p, y in dataloader:
            x_h = x_h.to(device)
            x_m = x_m.to(device)
            x_p = x_p.to(device)
            y = y.to(device)
            outputs = model(x_h, x_m, x_p, mode=mode)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten())
            all_labels.extend(y.detach().cpu().numpy().flatten())
    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_labels

def calculate_metrics(preds, labels, threshold=0.5):
    preds_binary = (np.array(preds) >= threshold).astype(int)
    labels = np.array(labels)
    return {
        'accuracy': accuracy_score(labels, preds_binary),
        'precision': precision_score(labels, preds_binary, zero_division=0),
        'recall': recall_score(labels, preds_binary, zero_division=0),
        'f1': f1_score(labels, preds_binary, zero_division=0)
    }

# ============================================================================
# 校准评估、DCA、阈值校准（来自 revised 版本）
# ============================================================================
def evaluate_calibration(y_true, y_prob, model_name, n_bins=10, save_path=None):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    brier = brier_score_loss(y_true, y_prob)
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='quantile'
        )
    except ValueError:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=min(n_bins, len(y_true)//10), strategy='uniform'
        )
    if len(fraction_of_positives) > 1:
        X = np.vstack([mean_predicted_value, np.ones(len(mean_predicted_value))]).T
        try:
            slope, intercept = np.linalg.lstsq(X, fraction_of_positives, rcond=None)[0]
        except np.linalg.LinAlgError:
            slope, intercept = 1.0, 0.0
    else:
        slope, intercept = 1.0, 0.0

    # ===== 字体增大 =====
    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({'axes.labelsize': 20})
    plt.rcParams.update({'xtick.labelsize': 18})
    plt.rcParams.update({'ytick.labelsize': 18})
    plt.rcParams.update({'legend.fontsize': 16})
    plt.rcParams.update({'axes.titlesize': 20})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', linewidth=2)
    ax1.plot(mean_predicted_value, fraction_of_positives, 's-',
             color='blue', linewidth=2, markersize=8, label=f'{model_name}')
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(f'Reliability Diagram\nBrier Score = {brier:.4f}')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    ax2.hist(y_prob[y_true == 0], bins=20, alpha=0.5, label='Control', color='green', density=True)
    ax2.hist(y_prob[y_true == 1], bins=20, alpha=0.5, label='Case', color='red', density=True)
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('Predicted Probability Distribution')
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  校准评估图已保存：{save_path}")
    plt.close()

    # 简化的 HL 统计量
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    hl_statistic = 0
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        n_in_bin = np.sum(in_bin)
        if n_in_bin > 0:
            observed_pos = np.sum(y_true[in_bin])
            expected_pos = np.sum(y_prob[in_bin])
            if expected_pos > 0 and (n_in_bin - expected_pos) > 0:
                hl_statistic += (observed_pos - expected_pos) ** 2 / expected_pos
                hl_statistic += ((n_in_bin - observed_pos) - (n_in_bin - expected_pos)) ** 2 / (n_in_bin - expected_pos)

    results = {
        'brier_score': brier,
        'calibration_slope': slope,
        'calibration_intercept': intercept,
        'hosmer_lemeshow_stat': hl_statistic,
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value': mean_predicted_value
    }
    print(f"    Brier Score: {brier:.4f}")
    print(f"    Calibration Slope: {slope:.3f} (ideal = 1.0)")
    print(f"    Calibration Intercept: {intercept:.3f} (ideal = 0.0)")
    print(f"    Hosmer-Lemeshow Statistic: {hl_statistic:.2f}")
    return results

def decision_curve_analysis(y_true, y_prob, model_name, threshold_range=None,
                           prevalence=None, save_path=None):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    if threshold_range is None:
        threshold_range = Config.DCA_THRESHOLD_RANGE
    if prevalence is None:
        prevalence = np.mean(y_true)

    net_benefit_model = []
    net_benefit_all = []
    net_benefit_none = []
    for threshold in threshold_range:
        pred_pos = (y_prob >= threshold).astype(int)
        tp = np.sum((pred_pos == 1) & (y_true == 1))
        fp = np.sum((pred_pos == 1) & (y_true == 0))
        n = len(y_true)
        if n > 0:
            nb_model = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        else:
            nb_model = 0
        nb_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        nb_none = 0
        net_benefit_model.append(nb_model)
        net_benefit_all.append(nb_all)
        net_benefit_none.append(nb_none)

    # ===== 字体增大 =====
    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({'axes.labelsize': 20})
    plt.rcParams.update({'xtick.labelsize': 18})
    plt.rcParams.update({'ytick.labelsize': 18})
    plt.rcParams.update({'legend.fontsize': 18})
    plt.rcParams.update({'axes.titlesize': 20})

    plt.figure(figsize=(10, 8))
    plt.plot(threshold_range, net_benefit_model, 'b-', linewidth=2.5,
             label=f'{model_name} (Model)')
    plt.plot(threshold_range, net_benefit_all, 'r--', linewidth=2,
             label='Treat All')
    plt.plot(threshold_range, net_benefit_none, 'k:', linewidth=2,
             label='Treat None')
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title(f'Decision Curve Analysis - {model_name}\n(Prevalence = {prevalence:.1%})',
              fontsize=18)
    plt.legend(loc='lower left',fontsize=16)
    plt.grid(alpha=0.3)
    plt.xlim([0, threshold_range[-1]])
    y_min = min(min(net_benefit_model), min(net_benefit_all), -0.05)
    y_max = max(max(net_benefit_model), max(net_benefit_all), 0.1)
    plt.ylim([y_min, y_max])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  DCA曲线已保存：{save_path}")
    plt.close()

    max_nb_idx = np.argmax(net_benefit_model)
    max_nb_threshold = threshold_range[max_nb_idx]
    max_nb_value = net_benefit_model[max_nb_idx]
    print(f"    最大净获益: {max_nb_value:.4f} @ 阈值 = {max_nb_threshold:.2f}")
    print(f"    在10%阈值下的净获益: {net_benefit_model[9]:.4f}")
    return {
        'thresholds': threshold_range,
        'net_benefit_model': np.array(net_benefit_model),
        'net_benefit_all': np.array(net_benefit_all),
        'net_benefit_none': np.array(net_benefit_none),
        'max_net_benefit': max_nb_value,
        'optimal_threshold': max_nb_threshold
    }

def calibrate_threshold(labels, probs, target_specificity=0.9):
    """
    基于目标特异度校准分类阈值，并输出指标及 95% 置信区间
    """
    labels = np.array(labels)
    probs = np.array(probs)
    neg_probs = probs[labels == 0]
    
    if len(neg_probs) == 0:
        print("  警告: 没有阴性样本，无法校准特异度")
        return 0.5, {}
    
    sorted_neg_probs = np.sort(neg_probs)
    
    # 【修正】取高分位数，确保达到目标特异度
    target_index = int(np.floor(target_specificity * len(sorted_neg_probs)))
    # 边界裁剪
    target_index = max(0, min(target_index, len(sorted_neg_probs) - 1))
    calibrated_threshold = sorted_neg_probs[target_index]
    
    # 防止阈值极端
    if calibrated_threshold < 0.01:
        calibrated_threshold = 0.01
    elif calibrated_threshold > 0.99:
        calibrated_threshold = 0.99

    preds_binary = (probs >= calibrated_threshold).astype(int)
    tp = np.sum((preds_binary == 1) & (labels == 1))
    tn = np.sum((preds_binary == 0) & (labels == 0))
    fp = np.sum((preds_binary == 1) & (labels == 0))
    fn = np.sum((preds_binary == 0) & (labels == 1))
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    n_pred_pos = np.sum(preds_binary == 1)
    n_pred_neg = np.sum(preds_binary == 0)
    n_total = len(labels)

    # 计算点估计
    sensitivity = tp / n_pos if n_pos > 0 else 0.0
    specificity = tn / n_neg if n_neg > 0 else 0.0
    ppv = tp / n_pred_pos if n_pred_pos > 0 else 0.0
    npv = tn / n_pred_neg if n_pred_neg > 0 else 0.0
    accuracy = (tp + tn) / n_total if n_total > 0 else 0.0

    # 计算 95% CI
    sens_ci = wilson_ci(n_pos, tp)
    spec_ci = wilson_ci(n_neg, tn)
    ppv_ci = wilson_ci(n_pred_pos, tp)
    npv_ci = wilson_ci(n_pred_neg, tn)
    acc_ci = wilson_ci(n_total, tp + tn)

    metrics = {
        'threshold': float(calibrated_threshold),
        'specificity': specificity,
        'sensitivity': sensitivity,
        'ppv': ppv,
        'npv': npv,
        'accuracy': accuracy,
        'specificity_ci': spec_ci,
        'sensitivity_ci': sens_ci,
        'ppv_ci': ppv_ci,
        'npv_ci': npv_ci,
        'accuracy_ci': acc_ci,
    }
    
    print(f"  校准阈值: {calibrated_threshold:.4f} (目标特异度: {target_specificity:.1%})")
    print(f"  实际特异度: {specificity:.3f} (95% CI: [{spec_ci[0]:.3f}, {spec_ci[1]:.3f}])")
    print(f"  灵敏度: {sensitivity:.3f} (95% CI: [{sens_ci[0]:.3f}, {sens_ci[1]:.3f}])")
    print(f"  PPV: {ppv:.3f} (95% CI: [{ppv_ci[0]:.3f}, {ppv_ci[1]:.3f}])")
    print(f"  NPV: {npv:.3f} (95% CI: [{npv_ci[0]:.3f}, {npv_ci[1]:.3f}])")
    print(f"  准确率: {accuracy:.3f} (95% CI: [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}])")
    
    return calibrated_threshold, metrics

def plot_modality_importance(ranking_df, save_dir, disease_type):
    """
    绘制各组学对多模态模型的贡献分析（三子图：总贡献、平均贡献、贡献百分比）
    ranking_df: 包含 Feature_Name, SHAP_Value, Modality 列的 DataFrame
    """
    # 汇总各组学总 SHAP 值
    modality_summary = ranking_df.groupby('Modality')['SHAP_Value'].agg(['sum', 'mean', 'count']).reset_index()
    modality_summary.columns = ['Modality', 'Total_SHAP', 'Mean_SHAP', 'Feature_Count']
    
    # 计算每个特征的平均贡献 = 总贡献 / 特征数
    modality_summary['Mean_per_Feature'] = modality_summary['Total_SHAP'] / modality_summary['Feature_Count']
    
    # 计算百分比贡献
    total_shap = modality_summary['Total_SHAP'].sum()
    modality_summary['Contribution_%'] = (modality_summary['Total_SHAP'] / total_shap * 100).round(2)
    
    # 按总贡献排序（保持统一顺序）
    modality_summary = modality_summary.sort_values('Total_SHAP', ascending=False)
    
    # 定义颜色映射（与截图一致：Clinical=红色, Metabolomics=青色, Proteomics=蓝色）
    modality_colors = {
        'Clinical Factors': '#FF6B6B',
        'Metabolomics': '#4ECDC4',
        'Proteomics': '#45B7D1'
    }
    
    # 确保颜色顺序与数据顺序一致
    colors = [modality_colors.get(m, '#999999') for m in modality_summary['Modality']]

    # 打印表格
    print("\n===== Modality Importance Summary =====")
    print(modality_summary.to_string(index=False))

    # 保存表格
    table_path = os.path.join(save_dir, f'Modality_Importance_{disease_type}.txt')
    modality_summary.to_csv(table_path, sep='\t', index=False)
    print(f"组学重要性表格已保存至：{table_path}")

    # ===== 修改1：创建 1行3列的子图布局 =====
    #plt.rcParams.update({'font.size': 14})
    #plt.rcParams.update({'axes.labelsize': 14})
    #plt.rcParams.update({'xtick.labelsize': 12})
    #plt.rcParams.update({'ytick.labelsize': 12})
    #plt.rcParams.update({'axes.titlesize': 14})

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1行3列，总宽度18英寸

    # ===== 子图 C: Total Contribution by Modality =====
    ax_c = axes[0]
    bars_c = ax_c.bar(modality_summary['Modality'], modality_summary['Total_SHAP'],
                      color=colors, edgecolor='black', linewidth=0.5)
    ax_c.set_ylabel('Total SHAP Value', fontsize=16)
    ax_c.set_title('Total Contribution by Modality', fontsize=16)
    ax_c.set_xticklabels(modality_summary['Modality'], rotation=45, ha='right', fontsize=16)
    ax_c.set_ylim(0, modality_summary['Total_SHAP'].max() * 1.15)
    ax_c.grid(axis='y', alpha=0.3)
    
    # 在柱顶标注绝对数值（不是百分比！）
    for bar, val in zip(bars_c, modality_summary['Total_SHAP']):
        height = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                  f'{val:.4f}', ha='center', va='bottom', fontsize=13)

    # ===== 子图 D: Mean Contribution per Feature (新增) =====
    ax_d = axes[1]
    bars_d = ax_d.bar(modality_summary['Modality'], modality_summary['Mean_per_Feature'],
                      color=colors, edgecolor='black', linewidth=0.5)
    ax_d.set_ylabel('Mean SHAP Value per Feature', fontsize=16)
    ax_d.set_title('Mean Contribution per Feature', fontsize=16)
    ax_d.set_xticklabels(modality_summary['Modality'], rotation=45, ha='right', fontsize=16)
    ax_d.set_ylim(0, modality_summary['Mean_per_Feature'].max() * 1.15)
    ax_d.grid(axis='y', alpha=0.3)
    
    # 在柱顶标注每个特征的平均贡献值
    for bar, val in zip(bars_d, modality_summary['Mean_per_Feature']):
        height = bar.get_height()
        ax_d.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                  f'{val:.4f}', ha='center', va='bottom', fontsize=16)

    # ===== 子图 E: Contribution Percentage by Modality (新增) =====
    ax_e = axes[2]
    bars_e = ax_e.bar(modality_summary['Modality'], modality_summary['Contribution_%'],
                      color=colors, edgecolor='black', linewidth=0.5)
    ax_e.set_ylabel('Contribution (%)', fontsize=14)
    ax_e.set_title('Contribution Percentage by Modality', fontsize=16)
    ax_e.set_xticklabels(modality_summary['Modality'], rotation=45, ha='right', fontsize=16)
    ax_e.set_ylim(0, 100)  # 百分比固定0-100范围
    ax_e.grid(axis='y', alpha=0.3)
    
    # 在柱顶标注百分比
    for bar, val in zip(bars_e, modality_summary['Contribution_%']):
        height = bar.get_height()
        ax_e.text(bar.get_x() + bar.get_width()/2., height + 2,
                  f'{val:.1f}%', ha='center', va='bottom', fontsize=13)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f'Modality_Importance_{disease_type}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"组学重要性三子图已保存至：{fig_path}")
    plt.close()

# ============================================================================
# SHAP 分析器（包含稳定性分析）
# ============================================================================
class SHAPStabilityAnalyzer:
    def __init__(self, model, config=Config()):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        if not SHAP_AVAILABLE:
            raise ImportError("shap库未安装。请运行: pip install shap")

    def _get_model_predictor(self, mode='full'):
        def predictor(X):
            self.model.eval()
            n_h = len(self.config.HIGH_RISK_COLS)
            n_m = len(self.config.METABOLOMICS_COLS)
            with torch.no_grad():
                x_h = torch.FloatTensor(X[:, :n_h]).to(self.device)
                x_m = torch.FloatTensor(X[:, n_h:n_h + n_m]).to(self.device)
                x_p = torch.FloatTensor(X[:, n_h + n_m:]).to(self.device)
                logits = self.model(x_h, x_m, x_p, mode='full')
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
            return probs
        return predictor

    def compute_shap_values(self, background_data, test_data, feature_names):
        predictor = self._get_model_predictor()
        explainer = shap.KernelExplainer(predictor, background_data)
        shap_values = explainer.shap_values(test_data, nsamples=100)
        return shap_values, explainer.expected_value

    def bootstrap_shap_stability(self, dataset, n_bootstrap=None, feature_names=None):
        if n_bootstrap is None:
            n_bootstrap = self.config.SHAP_N_BOOTSTRAP
        print(f"\n  开始SHAP Bootstrap稳定性分析 (n={n_bootstrap})...")
        print("  【使用KernelSHAP - 兼容性更好的SHAP计算方法】")

        n_samples = len(dataset)
        all_features = np.concatenate([
            dataset.high_risk,
            dataset.metabolomics,
            dataset.proteomics
        ], axis=1)
        if feature_names is None:
            feature_names = (dataset.high_risk_names +
                           dataset.metabolomics_names +
                           dataset.proteomics_names)

        np.random.seed(self.config.RANDOM_SEED)
        bg_indices = np.random.choice(n_samples, min(self.config.SHAP_N_BACKGROUND, n_samples), replace=False)
        background_data = all_features[bg_indices]
        test_indices = np.random.choice(n_samples, min(100, n_samples), replace=False)
        test_data = all_features[test_indices]

        all_rankings = []
        all_shap_values = []

        # 预测试
        print("  进行预测试算...")
        try:
            shap_values_test, _ = self.compute_shap_values(background_data, test_data[:10], feature_names)
            print(f"  预测试算成功，SHAP值形状: {np.array(shap_values_test).shape}")
        except Exception as e:
            print(f"  警告: KernelSHAP预测试算失败: {str(e)}")
            print("  将使用Permutation Importance替代...")
            return self._permutation_importance_analysis(dataset, feature_names)

        for b in range(n_bootstrap):
            boot_indices = resample(range(n_samples), n_samples=min(200, n_samples), random_state=b)
            boot_data = all_features[boot_indices]
            test_subset = boot_data[:min(30, len(boot_data))]
            try:
                shap_values, _ = self.compute_shap_values(background_data, test_subset, feature_names)
                if isinstance(shap_values, list):
                    shap_values = np.array(shap_values)
                mean_shap = np.abs(shap_values).mean(axis=0)
                ranking = np.argsort(np.argsort(-mean_shap)) + 1
                all_rankings.append(ranking)
                all_shap_values.append(mean_shap)
            except Exception as e:
                print(f"    Bootstrap {b+1} 失败: {str(e)[:50]}")
                continue
            if (b + 1) % 10 == 0:
                print(f"    完成 {b+1}/{n_bootstrap} Bootstrap (成功: {len(all_rankings)})...")

        if len(all_rankings) < 5:
            print("  警告: 成功完成的Bootstrap次数过少，使用Permutation Importance替代...")
            return self._permutation_importance_analysis(dataset, feature_names)

        rankings_array = np.array(all_rankings)
        shap_array = np.array(all_shap_values)
        mean_rank = rankings_array.mean(axis=0)
        rank_std = rankings_array.std(axis=0)
        mean_shap = shap_array.mean(axis=0)
        std_shap = shap_array.std(axis=0)
        cv_shap = np.where(mean_shap > 0.001, std_shap / mean_shap, 0)
        rank_stability = 1 - (rank_std / np.max(rank_std)) if np.max(rank_std) > 0 else np.ones_like(rank_std)
        stability_score = mean_shap * rank_stability
        sorted_idx = np.argsort(-stability_score)

        stability_results = {
            'feature_names': feature_names,
            'mean_rank': mean_rank,
            'rank_std': rank_std,
            'mean_shap': mean_shap,
            'std_shap': std_shap,
            'cv_shap': cv_shap,
            'stability_score': stability_score,
            'rankings_array': rankings_array,
            'top_stable_features': [
                {
                    'name': feature_names[i],
                    'mean_rank': int(mean_rank[i]),
                    'rank_std': float(rank_std[i]),
                    'mean_shap': float(mean_shap[i]),
                    'cv_shap': float(cv_shap[i]),
                    'stability_score': float(stability_score[i])
                }
                #for i in sorted_idx[:30]
                for i in sorted_idx[:min(30, len(sorted_idx))]  # 修复
            ]
        }

        print(f"\n  SHAP稳定性分析结果:")
        print(f"  {'='*80}")
        print(f"  {'Rank':<6} {'Feature':<30} {'Mean SHAP':<12} {'Rank Std':<10} {'CV':<8}")
        print(f"  {'-'*80}")
        for i, idx in enumerate(sorted_idx[:20]):
            print(f"  {i+1:<6} {feature_names[idx]:<30} {mean_shap[idx]:<12.6f} "
                  f"{rank_std[idx]:<10.2f} {cv_shap[idx]:<8.3f}")
        return stability_results

    def _permutation_importance_analysis(self, dataset, feature_names):
        print("\n  使用Permutation Importance进行特征重要性分析...")
        n_samples = len(dataset)
        all_features = np.concatenate([
            dataset.high_risk,
            dataset.metabolomics,
            dataset.proteomics
        ], axis=1)
        predictor = self._get_model_predictor()
        base_preds = predictor(all_features)
        from sklearn.metrics import roc_auc_score
        base_auc = roc_auc_score(dataset.labels, base_preds)

        importances = []
        np.random.seed(self.config.RANDOM_SEED)
        for i in range(all_features.shape[1]):
            permuted_features = all_features.copy()
            permuted_features[:, i] = np.random.permutation(permuted_features[:, i])
            perm_preds = predictor(permuted_features)
            perm_auc = roc_auc_score(dataset.labels, perm_preds)
            importance = base_auc - perm_auc
            importances.append(importance)

        importances = np.array(importances)
        sorted_idx = np.argsort(-importances)
        stability_results = {
            'feature_names': feature_names,
            'mean_rank': np.argsort(np.argsort(-importances)) + 1,
            'rank_std': np.zeros_like(importances),
            'mean_shap': importances,
            'std_shap': np.zeros_like(importances),
            'cv_shap': np.zeros_like(importances),
            'stability_score': importances,
            'rankings_array': np.array([np.argsort(np.argsort(-importances)) + 1]),
            'top_stable_features': [
                {
                    'name': feature_names[i],
                    'mean_rank': int(np.argsort(np.argsort(-importances))[i] + 1),
                    'rank_std': 0.0,
                    'mean_shap': float(importances[i]),
                    'cv_shap': 0.0,
                    'stability_score': float(importances[i])
                }
                #for i in sorted_idx[:30]
                for i in sorted_idx[:min(30, len(sorted_idx))]  # 修复
            ],
            'method': 'permutation_importance'
        }
        print(f"\n  Permutation Importance分析结果:")
        print(f"  {'='*80}")
        print(f"  {'Rank':<6} {'Feature':<30} {'Importance':<12}")
        print(f"  {'-'*80}")
        for i, idx in enumerate(sorted_idx[:min(20, len(sorted_idx))]):
            print(f"  {i+1:<6} {feature_names[idx]:<30} {importances[idx]:<12.6f}")
        return stability_results

# ============================================================================
# 原有 SHAP 分析器（保留）
# ============================================================================
class SHAPAnalyzer:
    def __init__(self, model, train_dataset, config, mode='full', model_name='Model'):
        self.model = model
        self.train_dataset = train_dataset
        self.config = config
        self.mode = mode
        self.model_name = model_name
        self.device = config.DEVICE
        self.feature_names = train_dataset.get_all_feature_names()
        self.high_risk_names = train_dataset.high_risk_names
        self.metabolomics_names = train_dataset.metabolomics_names
        self.proteomics_names = train_dataset.proteomics_names
        self.X_combined = train_dataset.get_combined_features()
        if mode == 'high_risk':
            self.feature_names = self.high_risk_names
            self.X_selected = self.train_dataset.high_risk
        elif mode == 'metab':
            self.feature_names = self.metabolomics_names
            self.X_selected = self.train_dataset.metabolomics
        elif mode == 'prot':
            self.feature_names = self.proteomics_names
            self.X_selected = self.train_dataset.proteomics
        else:
            self.X_selected = self.X_combined

        n_background = min(config.SHAP_BACKGROUND_SAMPLES, len(self.X_selected))
        np.random.seed(config.RANDOM_SEED)
        background_idx = np.random.choice(len(self.X_selected), n_background, replace=False)
        self.background_data = self.X_selected[background_idx]

        print(f"\nSHAP 分析器已初始化：{model_name}")
        print(f"  - 模式：{mode}")
        print(f"  - 特征数：{len(self.feature_names)}")
        print(f"  - 背景样本数：{len(self.background_data)}")

    def create_model_wrapper(self):
        mode = self.mode
        high_risk_names = self.high_risk_names
        metabolomics_names = self.metabolomics_names
        proteomics_names = self.proteomics_names
        device = self.device
        model = self.model
        def model_predict(X):
            model.eval()
            if mode == 'high_risk':
                X_high_risk = X
                X_metab = np.zeros((X.shape[0], len(metabolomics_names)))
                X_prot = np.zeros((X.shape[0], len(proteomics_names)))
            elif mode == 'metab':
                X_high_risk = np.zeros((X.shape[0], len(high_risk_names)))
                X_metab = X
                X_prot = np.zeros((X.shape[0], len(proteomics_names)))
            elif mode == 'prot':
                X_high_risk = np.zeros((X.shape[0], len(high_risk_names)))
                X_metab = np.zeros((X.shape[0], len(metabolomics_names)))
                X_prot = X
            else:
                X_high_risk = X[:, :len(high_risk_names)]
                X_metab = X[:, len(high_risk_names):len(high_risk_names) + len(metabolomics_names)]
                X_prot = X[:, len(high_risk_names) + len(metabolomics_names):]
            x_h = torch.FloatTensor(X_high_risk).to(device)
            x_m = torch.FloatTensor(X_metab).to(device)
            x_p = torch.FloatTensor(X_prot).to(device)
            with torch.no_grad():
                probs = model.predict_proba(x_h, x_m, x_p)
            return probs.cpu().numpy().flatten()
        return model_predict

    def compute_shap_values(self, n_samples=None):
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP 库不可用，使用 Gradient-based 重要性替代")
            return self.compute_gradient_importance()
        try:
            print("\n正在计算 SHAP 值...")
            model_predict = self.create_model_wrapper()
            if n_samples is None:
                n_samples = min(self.config.SHAP_EVAL_SAMPLES, len(self.X_selected))
            np.random.seed(self.config.RANDOM_SEED)
            eval_idx = np.random.choice(len(self.X_selected), n_samples, replace=False)
            eval_data = self.X_selected[eval_idx]
            print(f"  - 评估样本数：{n_samples}")
            print(f"  - 背景样本数：{len(self.background_data)}")
            test_pred = model_predict(eval_data[:1])
            print(f"    测试预测结果：{test_pred}")
            nsamples = self.config.SHAP_NSAMPLES
            explainer = shap.KernelExplainer(model_predict, self.background_data)
            shap_values = explainer.shap_values(eval_data, nsamples=nsamples)
            print(f"  - SHAP 值计算完成，形状：{shap_values.shape}")
            return shap_values, explainer.expected_value
        except Exception as e:
            print(f"\n❌ SHAP 计算失败：{e}")
            import traceback
            traceback.print_exc()
            print("  降级使用 Gradient-based 方法...")
            return self.compute_gradient_importance()

    def compute_gradient_importance(self):
        print("\n正在计算 Gradient-based 特征重要性...")
        self.model.eval()
        n_samples = min(self.config.SHAP_EVAL_SAMPLES, len(self.X_selected))
        np.random.seed(self.config.RANDOM_SEED)
        eval_idx = np.random.choice(len(self.X_selected), n_samples, replace=False)
        eval_data = self.X_selected[eval_idx]
        if self.mode == 'full':
            X_high_risk = torch.FloatTensor(eval_data[:, :len(self.high_risk_names)]).to(self.device)
            X_metab = torch.FloatTensor(eval_data[:, len(self.high_risk_names):len(self.high_risk_names) + len(self.metabolomics_names)]).to(self.device)
            X_prot = torch.FloatTensor(eval_data[:, len(self.high_risk_names) + len(self.metabolomics_names):]).to(self.device)
            X_high_risk.requires_grad_(True)
            X_metab.requires_grad_(True)
            X_prot.requires_grad_(True)
            outputs = self.model(X_high_risk, X_metab, X_prot, mode='full')
            probs = torch.sigmoid(outputs)
            probs.sum().backward()
            grad_high_risk = X_high_risk.grad.abs().mean(dim=0).cpu().numpy()
            grad_metab = X_metab.grad.abs().mean(dim=0).cpu().numpy()
            grad_prot = X_prot.grad.abs().mean(dim=0).cpu().numpy()
            gradient_importance = np.hstack([grad_high_risk, grad_metab, grad_prot])
        else:
            if self.mode == 'high_risk':
                X = torch.FloatTensor(eval_data).to(self.device)
            elif self.mode == 'metab':
                X = torch.FloatTensor(eval_data).to(self.device)
            else:
                X = torch.FloatTensor(eval_data).to(self.device)
            X.requires_grad_(True)
            if self.mode == 'high_risk':
                x_h, x_m, x_p = X, torch.zeros(1, len(self.metabolomics_names)).to(self.device).expand(n_samples, -1), torch.zeros(1, len(self.proteomics_names)).to(self.device).expand(n_samples, -1)
            elif self.mode == 'metab':
                x_h, x_m, x_p = torch.zeros(1, len(self.high_risk_names)).to(self.device).expand(n_samples, -1), X, torch.zeros(1, len(self.proteomics_names)).to(self.device).expand(n_samples, -1)
            else:
                x_h, x_m, x_p = torch.zeros(1, len(self.high_risk_names)).to(self.device).expand(n_samples, -1), torch.zeros(1, len(self.metabolomics_names)).to(self.device).expand(n_samples, -1), X
            outputs = self.model(x_h, x_m, x_p, mode=self.mode)
            probs = torch.sigmoid(outputs)
            probs.sum().backward()
            gradient_importance = X.grad.abs().mean(dim=0).cpu().numpy()
        print(f"  - Gradient 重要性计算完成，形状：{gradient_importance.shape}")
        return gradient_importance, None

    def get_feature_ranking(self, shap_values):
        if len(shap_values.shape) == 2:
            mean_shap = np.abs(shap_values).mean(axis=0)
        else:
            mean_shap = np.abs(shap_values).flatten()
        if self.mode == 'high_risk':
            modalities = ['Clinical Factors'] * len(self.feature_names)
        elif self.mode == 'metab':
            modalities = ['Metabolomics'] * len(self.feature_names)
        elif self.mode == 'prot':
            modalities = ['Proteomics'] * len(self.feature_names)
        else:
            modalities = (
                ['Clinical Factors'] * len(self.high_risk_names) +
                ['Metabolomics'] * len(self.metabolomics_names) +
                ['Proteomics'] * len(self.proteomics_names)
            )
        ranking_df = pd.DataFrame({
            'Feature_Name': self.feature_names,
            'SHAP_Value': mean_shap,
            'Modality': modalities
        })
        ranking_df = ranking_df.sort_values('SHAP_Value', ascending=False).reset_index(drop=True)
        ranking_df['Rank'] = ranking_df.index + 1
        return ranking_df

    def plot_shap_summary(self, shap_values, save_path=None):
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP 库不可用，跳过 SHAP 可视化")
            return
        print(f"\n绘制 {self.model_name} SHAP 摘要图...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            features=self.X_selected[:shap_values.shape[0]],
            feature_names=self.feature_names,
            plot_type='bar',
            show=False
        )
        plt.title(f'SHAP Feature Importance - {self.model_name}', fontsize=16)
        ax = plt.gca()
        ax.tick_params(axis='both', labelsize=16)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(16)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  SHAP 摘要图已保存：{save_path}")
        plt.close()

    def plot_top_features(self, ranking_df, top_n=30, save_path=None):
        print(f"\n绘制 {self.model_name} Top {top_n} 特征图...")
        # 修复：top_n 不能超过实际特征数
        top_n = min(top_n, len(ranking_df))
        top_df = ranking_df.head(top_n)
        modality_colors = {
            'Clinical Factors': '#FF6B6B',
            'Metabolomics': '#4ECDC4',
            'Proteomics': '#45B7D1'
        }
        colors = [modality_colors.get(m, '#999999') for m in top_df['Modality']]
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), top_df['SHAP_Value'], color=colors)
        plt.yticks(range(top_n), top_df['Feature_Name'], fontsize=16)
        plt.xlabel('Mean |SHAP Value|', fontsize=16)
        plt.ylabel('Features', fontsize=16)
        plt.title(f'Top {top_n} Features - {self.model_name}', fontsize=16)
        plt.gca().invert_yaxis()
        ax = plt.gca()
        ax.tick_params(axis='both', labelsize=16)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=mod) for mod, color in modality_colors.items()]
        legend = plt.legend(handles=legend_elements, loc='lower right', fontsize=16)
        for text in legend.get_texts():
            text.set_fontsize(16)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Top 特征图已保存：{save_path}")
        plt.close()

    def plot_shap_value_distribution(self, shap_values, save_path=None):
        print(f"\n绘制 {self.model_name} SHAP 值分布图...")
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 左子图（不变）
        axes[0].hist(mean_abs_shap, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Mean |SHAP Value|', fontsize=16)
        axes[0].set_ylabel('Number of Features', fontsize=16)
        axes[0].set_title(f'Distribution of Mean |SHAP| Values\n{self.model_name}', fontsize=16)
        axes[0].tick_params(axis='both', labelsize=16)
        axes[0].grid(axis='y', alpha=0.3)
        q75 = np.percentile(mean_abs_shap, 75)
        q25 = np.percentile(mean_abs_shap, 25)
        iqr = q75 - q25
        upper_bound = q75 + 3 * iqr
        axes[0].axvline(x=upper_bound, color='r', linestyle='--', label=f'Outlier Threshold ({upper_bound:.4f})')
        legend = axes[0].get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(16)

        # 中间子图 - 动态取 top_n
        top_n = min(20, len(mean_abs_shap))
        top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
        top_values = mean_abs_shap[top_idx]
        top_names = [self.feature_names[i] for i in top_idx]
        axes[1].barh(range(top_n), top_values)
        axes[1].set_yticks(range(top_n))
        axes[1].set_yticklabels(top_names, fontsize=16)
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Mean |SHAP Value|', fontsize=16)
        axes[1].set_title(f'Top {top_n} Features - {self.model_name}', fontsize=16)
        axes[1].tick_params(axis='both', labelsize=16)
        axes[1].grid(axis='x', alpha=0.3)

        # 最右边子图 - 修改 X 轴标签
        if self.mode == 'full':
            modality_shap = {}
            mod_start = 0
            for mod_name, mod_len in [('Clinical Factors', len(self.high_risk_names)),
                                   ('Metabolomics', len(self.metabolomics_names)),
                                   ('Proteomics', len(self.proteomics_names))]:
                modality_shap[mod_name] = mean_abs_shap[mod_start:mod_start+mod_len]
                mod_start += mod_len
            data = list(modality_shap.values())
            labels = list(modality_shap.keys())
            bp = axes[2].boxplot(data, labels=labels, patch_artist=True)
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            for patch, color in zip(bp['boxes'], colors[:len(data)]):
                patch.set_facecolor(color)
            axes[2].set_ylabel('Mean |SHAP Value|', fontsize=18)
            axes[2].set_title('SHAP Value Distribution by Modality', fontsize=18)
            axes[2].tick_params(axis='both', labelsize=18)
            # 修复：使用 set_xticks + set_xticklabels，避免 boxplot 自动刻度冲突
            axes[2].set_xticks(range(1, len(labels) + 1))
            axes[2].set_xticklabels(labels, fontsize=18, rotation=45, ha='right')
            axes[2].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  SHAP 值分布图已保存：{save_path}")
        plt.close()

    def save_shap_results(self, ranking_df, output_dir, disease_type):
        print(f"\n保存 {self.model_name} SHAP 结果到：{output_dir}")
        ranking_path = os.path.join(output_dir, f'SHAP_{self.model_name.replace(" ", "_")}_Feature_Ranking_{disease_type}.txt')
        ranking_df.to_csv(ranking_path, sep='\t', index=False)
        print(f"  特征排名已保存：{ranking_path}")

# ============================================================================
# 绘制 ROC 曲线（支持多模型）
# ============================================================================
def plot_roc_curves(results_dict, disease_type, save_path=None, title_prefix=''):
    """results_dict: {model_name: (preds, labels)}"""
     # 添加名称映射（与主函数中的映射一致）
    name_map = {
        '高危因素模型': 'Clinical Factors',
        '代谢组模型': 'Metabolomics',
        '蛋白组模型': 'Proteomics',
        '融合模型 (高危 + 代谢 + 蛋白)': 'Multi-Modal Fusion'
    }
    # ===== 新增：增大全局字体 =====
    plt.rcParams.update({'font.size': 18})        # 基础字体
    plt.rcParams.update({'axes.labelsize': 20})   # 轴标签
    plt.rcParams.update({'xtick.labelsize': 18})  # X轴刻度
    plt.rcParams.update({'ytick.labelsize': 18})  # Y轴刻度
    plt.rcParams.update({'legend.fontsize': 18})  # 图例
    plt.rcParams.update({'axes.titlesize': 20})       # 标题

    plt.figure(figsize=(8, 8))
    for model_name, (preds, labels) in results_dict.items():
        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        # 使用映射后的英文名称
        label_name = name_map.get(model_name, model_name)
        plt.plot(fpr, tpr, lw=2, label=f'{label_name} (AUC = {roc_auc:.3f})')
    #plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=20)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=20)
    if title_prefix:
        plt.title(f'{title_prefix} - {disease_type}', fontsize=18)
    else:
        plt.title(f'ROC Curves - {disease_type}', fontsize=20)
    plt.legend(loc='lower right', fontsize=16)
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC 曲线已保存至：{save_path}")
    plt.close()

# ============================================================================
# 辅助输出训练/验证结果
# ============================================================================
def output_train_val_results(config, train_dataset, val_dataset, all_results, disease_type):
    """
    输出训练集和验证集的预测结果
    使用数据集中真实的临床标签，确保Group列正确
    """
    # 直接从数据集获取真实标签（确保使用真实临床分组，而非预测标签）
    train_true_labels = train_dataset.labels
    val_true_labels = val_dataset.labels
    
    train_ids = train_dataset.get_sample_ids()
    val_ids = val_dataset.get_sample_ids()

    for mode, res in all_results.items():
        eng_name = res['model_name']
        
        # ===== 训练集输出 =====
        # 使用真实标签生成Group列
        train_df = pd.DataFrame({
            'Sample_ID': train_ids[:len(res['train_preds'])],
            'Group': [disease_type if l == 1 else 'control' for l in train_true_labels[:len(res['train_preds'])]],
            'Prediction_Score': res['train_preds'],
            'Dataset': 'Training',
            'Model': eng_name,
            'Disease_Type': disease_type
        })
        train_out = os.path.join(config.OUTPUT_DIR, f'{eng_name}_Training_{disease_type}_Results.txt')
        train_df.to_csv(train_out, sep='\t', index=False)
        print(f"训练集结果已保存：{train_out} ({len(train_df)} 样本)")
        
        # ===== 验证集输出 =====
        # 使用真实标签生成Group列
        val_df = pd.DataFrame({
            'Sample_ID': val_ids[:len(res['val_preds'])],
            'Group': [disease_type if l == 1 else 'control' for l in val_true_labels[:len(res['val_preds'])]],
            'Prediction_Score': res['val_preds'],
            'Dataset': 'Test',
            'Model': eng_name,
            'Disease_Type': disease_type
        })
        
        # 添加校准相关列（如果存在）
        if 'calibrated_threshold' in res:
            val_df['Calibrated_Threshold'] = res['calibrated_threshold']
            val_df['Calibrated_Prediction'] = (np.array(res['val_preds']) >= res['calibrated_threshold']).astype(int)
        if 'brier_score' in res:
            val_df['Brier_Score'] = res['brier_score']
            val_df['Calibration_Slope'] = res['calibration_slope']
            val_df['Calibration_Intercept'] = res['calibration_intercept']
        
        val_out = os.path.join(config.OUTPUT_DIR, f'{eng_name}_Validation_{disease_type}_Results.txt')
        val_df.to_csv(val_out, sep='\t', index=False)
        print(f"验证集结果已保存：{val_out} ({len(val_df)} 样本)")
        
        # ===== 验证数据一致性 =====
        # 打印前5个样本的Group信息用于调试
        if len(train_df) > 0 and len(val_df) > 0:
            print(f"  {eng_name} 训练集前5个样本Group: {train_df['Group'].head(3).tolist()}")
            print(f"  {eng_name} 验证集前5个样本Group: {val_df['Group'].head(3).tolist()}")

# ============================================================================
# 绘制 SHAP 稳定性图（字体加倍）
# ============================================================================
def plot_shap_stability(stability_results, disease_type, output_dir):
    # 增大全局字体
    plt.rcParams.update({'font.size': 20})  # 基础字体从10增大到20
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))  # 增大画布

    top_n = min(20, len(stability_results['stability_score']))
    sorted_idx = np.argsort(-stability_results['stability_score'])[:top_n]
    names = [stability_results['feature_names'][i] for i in sorted_idx]
    mean_shap = stability_results['mean_shap'][sorted_idx]
    rank_std = stability_results['rank_std'][sorted_idx]
    cv_shap = stability_results['cv_shap'][sorted_idx]

    ax1 = axes[0, 0]
    y_pos = np.arange(top_n)
    ax1.barh(y_pos, mean_shap, xerr=rank_std * 0.01, capsize=3, color='steelblue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=20)  # 从8增大到20
    ax1.set_xlabel('Mean |SHAP Value|', fontsize=24)
    #ax1.set_title('SHAP Importance (with Rank Std Dev)', fontsize=24)
    ax1.invert_yaxis()
    ax1.tick_params(axis='x', labelsize=20)

    ax2 = axes[0, 1]
    stability_score = stability_results['stability_score'][sorted_idx]
    # 处理颜色归一化
    max_cv = np.max(cv_shap) if np.max(cv_shap) > 0 else 1
    colors = plt.cm.RdYlGn(1 - cv_shap / max_cv)
    ax2.barh(y_pos, stability_score, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=20)
    ax2.set_xlabel('Stability Score', fontsize=24)
    #ax2.set_title('Feature Stability Score', fontsize=24)
    ax2.invert_yaxis()
    ax2.tick_params(axis='x', labelsize=20)

    ax3 = axes[1, 0]
    ax3.hist(stability_results['cv_shap'], bins=30, color='orange', edgecolor='black', alpha=0.7)
    ax3.axvline(x=1.0, color='red', linestyle='--', label='CV=1.0')
    ax3.set_xlabel('Coefficient of Variation', fontsize=24)
    ax3.set_ylabel('Frequency', fontsize=24)
    #ax3.set_title('Distribution of SHAP Variability', fontsize=24)
    ax3.legend(fontsize=20)
    ax3.tick_params(axis='both', labelsize=20)

    ax4 = axes[1, 1]
    # 最右下子图
    top_15_n = min(15, len(sorted_idx))  # 修复
    top_15_idx = sorted_idx[:top_15_n]
    rank_matrix = stability_results['rankings_array'][:, top_15_idx]
    im = ax4.imshow(rank_matrix.T, aspect='auto', cmap='RdYlGn_r', vmin=1, vmax=50)
    ax4.set_xlabel('Bootstrap Iteration', fontsize=24)
    #ax4.set_ylabel('Feature', fontsize=24)
    ax4.set_yticks(range(15))
    ax4.set_yticklabels([stability_results['feature_names'][i] for i in top_15_idx], fontsize=18)
    #ax4.set_title('Bootstrap Ranking Consistency', fontsize=24)
    ax4.tick_params(axis='x', labelsize=20)
    cbar = plt.colorbar(im, ax=ax4, label='Rank')
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Rank', fontsize=24)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'SHAP_Stability_Plot_{disease_type}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  SHAP稳定性图已保存：{save_path}")
    plt.close()
    # 恢复默认字体
    plt.rcParams.update({'font.size': 10})

# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='PTPE Transformer 增强版（含类别不平衡处理）')
    parser.add_argument('--data_path', type=str, default=Config.DATA_PATH,
                        help='主数据文件路径（含 Training/Test 划分）')
    parser.add_argument('--external_path', type=str, default=None,
                        help='外部验证集文件路径（独立数据集）')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')
    
    # 新增：三个组学特征文件参数
    parser.add_argument('--high_risk_file', type=str, default=None,
                        help='临床高危因素特征文件路径（每行一个特征名）')
    parser.add_argument('--metabolomics_file', type=str, default=None,
                        help='代谢组学特征文件路径（每行一个特征名）')
    parser.add_argument('--proteomics_file', type=str, default=None,
                        help='蛋白组学特征文件路径（每行一个特征名）')
    
    args = parser.parse_args()

    config = Config()
    config.DATA_PATH = args.data_path
    config.EXTERNAL_PATH = args.external_path
    config.OUTPUT_DIR = args.output_dir
    
    # 设置特征文件路径
    config.HIGH_RISK_FILE = args.high_risk_file
    config.METABOLOMICS_FILE = args.metabolomics_file
    config.PROTEOMICS_FILE = args.proteomics_file
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("=" * 60)
    print("PTPE Transformer 模型训练 + 增强分析（含类别不平衡处理）")
    print("=" * 60)
    print(f"计算设备：{config.DEVICE}")
    print(f"主数据路径：{config.DATA_PATH}")
    if config.EXTERNAL_PATH:
        print(f"外部验证集路径：{config.EXTERNAL_PATH}")
    print(f"输出目录：{config.OUTPUT_DIR}")
    print()

    # 1. 加载训练/验证数据（先读取数据文件以获取列名）
    print("正在加载数据...")
    temp_df = pd.read_csv(config.DATA_PATH, sep='\t', engine='python')
    if len(temp_df.columns) < 10:
        temp_df = pd.read_csv(config.DATA_PATH, sep='\s+', engine='python')
    
    all_columns = list(temp_df.columns)
    print(f"数据文件共 {len(all_columns)} 列")
    
    # 根据特征文件加载特征列名并转换为列索引
    print("\n【加载特征文件】")
    
    # 临床高危因素
    if config.HIGH_RISK_FILE:
        high_risk_names = load_feature_columns(config.HIGH_RISK_FILE)
        config.HIGH_RISK_COLS = get_column_indices(temp_df, high_risk_names, "临床高危因素")
    else:
        # 使用默认列范围
        config.HIGH_RISK_COLS = list(range(3, 10))
        print("  使用默认高危因素列范围: 3-9")
    
    # 代谢组学
    if config.METABOLOMICS_FILE:
        metabolomics_names = load_feature_columns(config.METABOLOMICS_FILE)
        config.METABOLOMICS_COLS = get_column_indices(temp_df, metabolomics_names, "代谢组学")
    else:
        # 使用默认列范围
        config.METABOLOMICS_COLS = list(range(10, 151))
        print("  使用默认代谢组学列范围: 10-150")
    
    # 蛋白组学
    if config.PROTEOMICS_FILE:
        proteomics_names = load_feature_columns(config.PROTEOMICS_FILE)
        config.PROTEOMICS_COLS = get_column_indices(temp_df, proteomics_names, "蛋白组学")
    else:
        # 使用默认列范围
        config.PROTEOMICS_COLS = list(range(151, 168))
        print("  使用默认蛋白组学列范围: 151-167")
    
    print(f"\n最终特征配置:")
    print(f"  高危因素: {len(config.HIGH_RISK_COLS)} 个特征")
    print(f"  代谢组学: {len(config.METABOLOMICS_COLS)} 个特征")
    print(f"  蛋白组学: {len(config.PROTEOMICS_COLS)} 个特征")
    print(f"  总特征数: {len(config.HIGH_RISK_COLS) + len(config.METABOLOMICS_COLS) + len(config.PROTEOMICS_COLS)}")
    print()

    # 正式加载数据集
    train_dataset = PTPEDataset(config.DATA_PATH, dataset_type='Training', config=config)
    val_dataset = PTPEDataset(config.DATA_PATH, dataset_type='Test', config=config)
    disease_type = train_dataset.disease_type

    # 打印类别不平衡处理策略
    print("\n【类别不平衡处理策略】")
    print(f"  USE_CLASS_WEIGHT = {config.USE_CLASS_WEIGHT}")
    print(f"  USE_FOCAL_LOSS = {config.USE_FOCAL_LOSS}")
    print(f"  USE_RESAMPLING = {config.USE_RESAMPLING}")
    class_weights = train_dataset.class_weights
    print(f"  类别权重（负类:正类）= {class_weights[0]:.3f}:{class_weights[1]:.3f}")
    print(f"  训练集阳性比例 = {train_dataset.labels.mean():.2%}")
    print()

    # 创建 DataLoader（根据配置选择是否重采样）
    if config.USE_RESAMPLING:
        sampler = train_dataset.get_sampler()
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler,
            num_workers=0, pin_memory=True
        )
        print("  使用 WeightedRandomSampler 进行重采样训练")
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=0, pin_memory=True
        )
        print("  使用标准 shuffle 训练（未启用重采样）")

    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # 定义损失函数（根据配置选择类别权重或 Focal Loss）
    def get_criterion(dataset, config):
        if config.USE_FOCAL_LOSS:
            alpha = dataset.class_weights.numpy()
            alpha = alpha / alpha.sum()
            print(f"  使用 Focal Loss (gamma={config.FOCAL_LOSS_GAMMA}, "
                  f"alpha=[{alpha[0]:.3f}, {alpha[1]:.3f}])")
            return FocalLoss(gamma=config.FOCAL_LOSS_GAMMA, alpha=alpha)
        elif config.USE_CLASS_WEIGHT:
            pos_weight = dataset.class_weights[1] / dataset.class_weights[0]
            print(f"  使用 BCEWithLogitsLoss + pos_weight = {pos_weight:.3f}")
            return nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight]))
        else:
            print("  使用标准 BCEWithLogitsLoss（无类别权重）")
            return nn.BCEWithLogitsLoss()

    # 2. 训练四个模型
    modes = {
        'high_risk': '高危因素模型',
        'metab': '代谢组模型',
        'prot': '蛋白组模型',
        'full': '融合模型 (高危 + 代谢 + 蛋白)'
    }
    mode_name_map = {
        'high_risk': 'Clinical-Factors',
        'metab': 'Metabolomics',
        'prot': 'Proteomics',
        'full': 'Multi-Modal-Fusion'
    }

    results_dict = {}
    best_models = {}
    all_results = {}

    for mode, mode_name in modes.items():
        print("\n" + "=" * 60)
        print(f"训练 {mode_name}")
        print("=" * 60)

        model = MultiModalTransformer(config).to(config.DEVICE)
        criterion = get_criterion(train_dataset, config)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(config.NUM_EPOCHS):
            train_loss, train_preds, train_labels = train_epoch(
                model, train_loader, criterion, optimizer, config.DEVICE, mode=mode
            )
            val_loss, val_preds, val_labels = validate(
                model, val_loader, criterion, config.DEVICE, mode=mode
            )
            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                train_metrics = calculate_metrics(train_preds, train_labels)
                val_metrics = calculate_metrics(val_preds, val_labels)
                print(f"Epoch {epoch+1:3d}/{config.NUM_EPOCHS} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_metrics['accuracy']:.3f} | Val F1: {val_metrics['f1']:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print(f"早停触发于 epoch {epoch+1}")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        best_models[mode] = model

        model_path = os.path.join(config.OUTPUT_DIR, f"{mode_name_map[mode]}_best_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存：{model_path}")

        # 最终评估（使用验证集）
        # 确保模型处于最佳状态
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # 使用validate获取训练集和验证集的预测结果
        # 注意：需要传入一个dummy criterion，但validate会计算loss
        # 创建一个不shuffle的评估用DataLoader
        eval_train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=True
        )
        dummy_criterion = nn.BCEWithLogitsLoss()
        _, train_preds, train_labels = validate(model, eval_train_loader, dummy_criterion, config.DEVICE, mode=mode)
        _, val_preds, val_labels = validate(model, val_loader, dummy_criterion, config.DEVICE, mode=mode)
        val_metrics = calculate_metrics(val_preds, val_labels)
        fpr, tpr, _ = roc_curve(val_labels, val_preds)
        val_auc = auc(fpr, tpr)

        print(f"\n{mode_name} 最终结果:")
        print(f"  训练集 AUC: {auc(*roc_curve(train_labels, train_preds)[:2]):.4f}")
        print(f"  验证集 AUC: {val_auc:.4f}")
        print(f"  验证集准确率：{val_metrics['accuracy']:.3f}")
        print(f"  验证集精确率：{val_metrics['precision']:.3f}")
        print(f"  验证集召回率：{val_metrics['recall']:.3f}")
        print(f"  验证集 F1 分数：{val_metrics['f1']:.3f}")
        print()

        results_dict[mode_name] = (val_preds, val_labels)
        all_results[mode] = {
            'train_preds': train_preds,
            'train_labels': train_labels,
            'val_preds': val_preds,
            'val_labels': val_labels,
            'model_name': mode_name_map[mode],
        }

    # 3. 绘制验证集 ROC 曲线
    roc_save_path = os.path.join(config.OUTPUT_DIR, f'ROC_Curves_Validation_{disease_type}.png')
    plot_roc_curves(results_dict, disease_type, save_path=roc_save_path, title_prefix='Validation Set')

    # 4. 保存训练集和验证集预测结果
    output_train_val_results(config, train_dataset, val_dataset, all_results, disease_type)

    # 5. 校准评估、DCA、阈值校准（验证集）
    print("\n" + "=" * 60)
    print("验证集校准评估与决策曲线分析")
    print("=" * 60)
    calib_results = {}
    for mode, res in all_results.items():
        eng_name = res['model_name']
        y_true = res['val_labels']
        y_prob = res['val_preds']

        print(f"\n--- {eng_name} 阈值校准 ---")
        thresh, cal_metrics = calibrate_threshold(y_true, y_prob, target_specificity=config.CALIBRATION_THRESHOLD)
        res['calibrated_threshold'] = thresh
        res['calibration_metrics'] = cal_metrics

        cal_save = os.path.join(config.OUTPUT_DIR, f'Calibration_{eng_name}_{disease_type}.png')
        cal_res = evaluate_calibration(y_true, y_prob, eng_name,
                                       n_bins=config.CALIBRATION_N_BINS, save_path=cal_save)
        res['brier_score'] = cal_res['brier_score']
        res['calibration_slope'] = cal_res['calibration_slope']
        res['calibration_intercept'] = cal_res['calibration_intercept']

        dca_save = os.path.join(config.OUTPUT_DIR, f'DCA_{eng_name}_{disease_type}.png')
        dca_res = decision_curve_analysis(y_true, y_prob, eng_name,
                                          threshold_range=config.DCA_THRESHOLD_RANGE,
                                          save_path=dca_save)
        res['dca_results'] = dca_res

        calib_results[mode] = cal_res
    # ===== 保存验证集阈值校准结果到文件 =====

    calib_results_list = []
    for mode, res in all_results.items():
        eng_name = res['model_name']
        cal_metrics = res['calibration_metrics']  # 已包含阈值及指标
        row = {
            'Model': eng_name,
            'Calibrated_Threshold': cal_metrics['threshold'],
            'Specificity': cal_metrics['specificity'],
            'Specificity_Lower_CI': cal_metrics['specificity_ci'][0],
            'Specificity_Upper_CI': cal_metrics['specificity_ci'][1],
            'Sensitivity': cal_metrics['sensitivity'],
            'Sensitivity_Lower_CI': cal_metrics['sensitivity_ci'][0],
            'Sensitivity_Upper_CI': cal_metrics['sensitivity_ci'][1],
            'PPV': cal_metrics['ppv'],
            'PPV_Lower_CI': cal_metrics['ppv_ci'][0],
            'PPV_Upper_CI': cal_metrics['ppv_ci'][1],
            'NPV': cal_metrics['npv'],
            'NPV_Lower_CI': cal_metrics['npv_ci'][0],
            'NPV_Upper_CI': cal_metrics['npv_ci'][1],
            'Accuracy': cal_metrics['accuracy'],
            'Accuracy_Lower_CI': cal_metrics['accuracy_ci'][0],
            'Accuracy_Upper_CI': cal_metrics['accuracy_ci'][1],
        }
        calib_results_list.append(row)

    calib_df = pd.DataFrame(calib_results_list)
    calib_file = os.path.join(config.OUTPUT_DIR, f'Threshold_Calibration_Validation_{disease_type}.txt')
    calib_df.to_csv(calib_file, sep='\t', index=False)
    print(f"验证集阈值校准结果已保存：{calib_file}")

    # 6. SHAP 分析（融合模型）
    print("\n" + "=" * 60)
    print("SHAP 特征重要性分析（融合模型）")
    print("=" * 60)
    if SHAP_AVAILABLE and 'full' in best_models:
        full_model = best_models['full']
        shap_analyzer = SHAPAnalyzer(
            model=full_model,
            train_dataset=train_dataset,
            config=config,
            mode='full',
            model_name='Multi-Modal Fusion'
        )
        shap_values, expected_value = shap_analyzer.compute_shap_values()
        if shap_values is not None:
            ranking_df = shap_analyzer.get_feature_ranking(shap_values)
            print(f"\nMulti-Modal Fusion Top 10 特征:")
            print(ranking_df.head(10).to_string(index=False))
            shap_analyzer.plot_shap_summary(shap_values, save_path=os.path.join(config.OUTPUT_DIR, f'SHAP_Summary_Fusion_{disease_type}.png'))
            shap_analyzer.plot_top_features(ranking_df, top_n=30, save_path=os.path.join(config.OUTPUT_DIR, f'SHAP_Top30_Fusion_{disease_type}.png'))
            shap_analyzer.plot_shap_value_distribution(shap_values, save_path=os.path.join(config.OUTPUT_DIR, f'SHAP_Value_Distribution_Fusion_{disease_type}.png'))
            shap_analyzer.save_shap_results(ranking_df, config.OUTPUT_DIR, disease_type)
        
        # ===== 新增：组学重要性分析 =====
        plot_modality_importance(ranking_df, config.OUTPUT_DIR, disease_type)
        # SHAP 稳定性分析
        try:
            stability_analyzer = SHAPStabilityAnalyzer(full_model, config)
            stability_results = stability_analyzer.bootstrap_shap_stability(
                train_dataset, n_bootstrap=config.SHAP_N_BOOTSTRAP
            )
            if stability_results:
                stability_df = pd.DataFrame(stability_results['top_stable_features'])
                stability_path = os.path.join(config.OUTPUT_DIR, f'SHAP_Stability_{disease_type}.txt')
                stability_df.to_csv(stability_path, sep='\t', index=False)
                print(f"\n  SHAP稳定性结果已保存：{stability_path}")
                plot_shap_stability(stability_results, disease_type, config.OUTPUT_DIR)
        except Exception as e:
            print(f"  SHAP稳定性分析失败: {str(e)}")
            print("  继续执行其他分析...")

    # 7. 外部验证集（如果提供）
    if config.EXTERNAL_PATH and os.path.exists(config.EXTERNAL_PATH):
        print("\n" + "=" * 60)
        print("外部验证集评估")
        print("=" * 60)
        # 准备训练集统计量
        train_stats = {
            'high_risk_mean': train_dataset.high_risk_mean,
            'high_risk_std': train_dataset.high_risk_std,
            'metab_mean': train_dataset.metab_mean,
            'metab_std': train_dataset.metab_std,
            'prot_mean': train_dataset.prot_mean,
            'prot_std': train_dataset.prot_std,
            'metab_names': train_dataset.metabolomics_names,
            'prot_names': train_dataset.proteomics_names,
            'feature_names': train_dataset.get_all_feature_names(),
            'high_risk_names': train_dataset.high_risk_names  # 新增：用于外部验证集列名匹配
        }
        ext_dataset = ExternalValidationDataset(
            config.EXTERNAL_PATH, train_stats,
            disease_type=disease_type,
            positive_label=train_dataset.positive_label,
            config=config
        )
        ext_loader = DataLoader(ext_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        # 对所有模型进行外部验证
        fusion_preds = None
        fusion_labels = None
        ext_results = {}   # 初始化 ext_results 字典
        # 获取外部验证集的真实标签
        ext_true_labels = ext_dataset.labels

        for mode, model in best_models.items():
            _, ext_preds, ext_labels = validate(model, ext_loader, nn.BCEWithLogitsLoss(), config.DEVICE, mode=mode)
            ext_results[mode_name_map[mode]] = (ext_preds, ext_labels)

            # 保存融合模型的数据
            if mode == 'full':
                fusion_preds = ext_preds
                fusion_labels = ext_labels

            # 保存外部结果文件（原有代码）
            ext_sample_ids = ext_dataset.get_sample_ids()
            ext_df = pd.DataFrame({
                'Sample_ID': ext_sample_ids,
                #'Group': [train_dataset.positive_label if l == 1 else 'control' for l in ext_labels],
                'Group': [train_dataset.positive_label if l == 1 else 'control' for l in ext_true_labels[:len(ext_preds)]],
                'Prediction_Score': ext_preds,
                'Dataset': 'External',
                'Model': mode_name_map[mode],
                'Disease_Type': disease_type
            })
            ext_out = os.path.join(config.OUTPUT_DIR, f'External_{mode_name_map[mode]}_{disease_type}_Results.txt')
            ext_df.to_csv(ext_out, sep='\t', index=False)
            print(f"外部验证集结果已保存：{ext_out} ({len(ext_df)} 样本)")


        # 绘制外部 ROC 曲线
        ext_roc_path = os.path.join(config.OUTPUT_DIR, f'ROC_Curves_External_{disease_type}.png')
        plot_roc_curves(ext_results, disease_type, save_path=ext_roc_path, title_prefix='External Validation Set')

        # 外部验证集校准和DCA（使用融合模型）
        full_preds = ext_results['Multi-Modal-Fusion'][0]
        full_labels = ext_results['Multi-Modal-Fusion'][1]
        ext_cal_save = os.path.join(config.OUTPUT_DIR, f'Calibration_External_{disease_type}.png')
        evaluate_calibration(full_labels, full_preds, 'External_Validation',
                             n_bins=config.CALIBRATION_N_BINS, save_path=ext_cal_save)
        ext_dca_save = os.path.join(config.OUTPUT_DIR, f'DCA_External_{disease_type}.png')
        decision_curve_analysis(full_labels, full_preds, 'External_Validation',
                                threshold_range=config.DCA_THRESHOLD_RANGE,
                                save_path=ext_dca_save)
        # ======== 外部验证集阈值校准分析（使用内部验证集校准阈值） ========
        # 外部验证集阈值校准分析（使用独立变量）
        if fusion_preds is not None and fusion_labels is not None:
            cal_thresh = all_results['full']['calibrated_threshold']
            print(f"\n--- 外部验证集阈值校准分析 (使用内部验证集校准阈值: {cal_thresh:.4f}) ---")

            # 调试信息
            print(f"  fusion_preds 范围: [{np.min(fusion_preds):.4f}, {np.max(fusion_preds):.4f}]")
            print(f"  阳性样本数: {np.sum(fusion_labels)}, 阴性样本数: {len(fusion_labels)-np.sum(fusion_labels)}")
            print(f"  预测分数 >= 阈值的样本数: {np.sum(np.array(fusion_preds) >= cal_thresh)}")

            # 【修复】先将 list 转换为 numpy 数组，确保元素级比较正确
            fusion_labels_arr = np.array(fusion_labels)
            fusion_preds_arr = np.array(fusion_preds)

            preds_binary = (fusion_preds_arr >= cal_thresh).astype(int)
            tp = np.sum((preds_binary == 1) & (fusion_labels_arr == 1))
            tn = np.sum((preds_binary == 0) & (fusion_labels_arr == 0))
            fp = np.sum((preds_binary == 1) & (fusion_labels_arr == 0))
            fn = np.sum((preds_binary == 0) & (fusion_labels_arr == 1))
            n_pos = np.sum(fusion_labels_arr == 1)
            n_neg = np.sum(fusion_labels_arr == 0)
            n_pred_pos = np.sum(preds_binary == 1)
            n_pred_neg = np.sum(preds_binary == 0)
            n_total = len(fusion_labels_arr)

            sensitivity = tp / n_pos if n_pos > 0 else 0.0
            specificity = tn / n_neg if n_neg > 0 else 0.0
            ppv = tp / n_pred_pos if n_pred_pos > 0 else 0.0
            npv = tn / n_pred_neg if n_pred_neg > 0 else 0.0
            accuracy = (tp + tn) / n_total if n_total > 0 else 0.0

            sens_ci = wilson_ci(n_pos, tp)
            spec_ci = wilson_ci(n_neg, tn)
            ppv_ci = wilson_ci(n_pred_pos, tp)
            npv_ci = wilson_ci(n_pred_neg, tn)
            acc_ci = wilson_ci(n_total, tp + tn)

            print(f"  使用阈值: {cal_thresh:.4f}")
            print(f"  特异度: {specificity:.3f} (95% CI: [{spec_ci[0]:.3f}, {spec_ci[1]:.3f}])")
            print(f"  灵敏度: {sensitivity:.3f} (95% CI: [{sens_ci[0]:.3f}, {sens_ci[1]:.3f}])")
            print(f"  PPV: {ppv:.3f} (95% CI: [{ppv_ci[0]:.3f}, {ppv_ci[1]:.3f}])")
            print(f"  NPV: {npv:.3f} (95% CI: [{npv_ci[0]:.3f}, {npv_ci[1]:.3f}])")
            print(f"  准确率: {accuracy:.3f} (95% CI: [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}])")

            # 保存到文件（独立文件）
            ext_calib_df = pd.DataFrame([{
                'Model': 'Multi-Modal-Fusion',
                'Threshold_Used': cal_thresh,
                'Specificity': specificity,
                'Specificity_Lower_CI': spec_ci[0],
                'Specificity_Upper_CI': spec_ci[1],
                'Sensitivity': sensitivity,
                'Sensitivity_Lower_CI': sens_ci[0],
                'Sensitivity_Upper_CI': sens_ci[1],
                'PPV': ppv,
                'PPV_Lower_CI': ppv_ci[0],
                'PPV_Upper_CI': ppv_ci[1],
                'NPV': npv,
                'NPV_Lower_CI': npv_ci[0],
                'NPV_Upper_CI': npv_ci[1],
                'Accuracy': accuracy,
                'Accuracy_Lower_CI': acc_ci[0],
                'Accuracy_Upper_CI': acc_ci[1]
            }])
            ext_calib_file = os.path.join(config.OUTPUT_DIR, f'Threshold_Calibration_External_{disease_type}.txt')
            ext_calib_df.to_csv(ext_calib_file, sep='\t', index=False)
            print(f"外部验证集阈值校准结果已保存：{ext_calib_file}")

    # 8. 总结
    print("\n" + "=" * 60)
    print("✅ 训练和增强分析完成!")
    print("=" * 60)
    print(f"输出目录：{config.OUTPUT_DIR}")

if __name__ == '__main__':
    main()
