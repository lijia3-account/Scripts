Early Prediction of Preeclampsia Using Multi-Modal Transformer Models Integrating Clinical, Multi-Omics Data
https://www.python.org/
https://pytorch.org/
LICENSE
Overview
This repository contains the complete implementation of a multi-modal Transformer-based deep learning framework for early prediction of preeclampsia (PE) by integrating clinical risk factors, targeted metabolomics, and targeted proteomics data. The framework supports both Preterm Preeclampsia (Preterm_PE) and Term Preeclampsia (Term_PE) prediction tasks, with comprehensive evaluation including calibration assessment, decision curve analysis (DCA), SHAP interpretability analysis, and external validation.
Repository Structure
plain
├── pe_transformer_final_deepseek.py    # Main training & evaluation script (Transformer model)
├── ml_comparison.py                    # Traditional machine learning baseline comparison
├── GW16_analysis.py                    # Gestational week 16-specific analysis utilities
├── ml_comparison.sh                    # Shell script for ML comparison pipeline
├── run_train_cpu_ptpe.sh               # CPU training script for PTPE/TPE models
├── GW16_analysis.sh                    # Shell script for GW16 analysis pipeline
├── high_risk_features.txt              # Clinical high-risk factor feature list
├── metab_features_ptpe.txt             # Metabolomics feature list (PTPE)
├── prot_features_ptpe.txt              # Proteomics feature list (PTPE)
├── Multi-Modal-Fusion_best_model_PTPE.pth   # Pre-trained PTPE fusion model weights
├── Multi-Modal-Fusion_best_model_TPE.pth    # Pre-trained TPE fusion model weights
├── simulated_PTPE_1000samples.txt      # Simulated dataset (1000 samples, for training and interval validation)
├── simulated_PTPE_160samples_EV.txt    # Simulated external validation set (160 samples)
└── README.md                           # This file
Requirements
Dependencies
bash
# Core dependencies
torch>=1.12.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
shap>=0.41.0          # For SHAP interpretability analysis
Installation
bash
# Clone the repository
git clone https://github.com/lijia3-account/PE-Transformer-MultiOmics.git
cd PE-Transformer-MultiOmics

# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib shap
Data Format
Input Data Structure
The main data file should be a tab-delimited text file with the following columns:
表格
Column Index	Column Name	Description
0	ID	Unique sample identifier
1	DATASET	Dataset split: Training or EV (internal validation)
2	Group	Clinical outcome: Preterm_PE / Term_PE or control
3–9	Clinical Factors	High-risk clinical variables (e.g., GA_gestation, Birth_weight, MAP, PMH, BMI, Age, Birth_history, IVF, RPL)
10–150	Metabolomics	Targeted metabolite abundance measurements
151–167	Proteomics	Targeted protein abundance measurements
Feature List Files
high_risk_features.txt: One clinical feature name per line
metab_features_ptpe.txt: One metabolite feature name per line
prot_features_ptpe.txt: One protein feature name per line
These files enable flexible feature selection without modifying the code.
Usage
1. Training the Multi-Modal Transformer Model
bash
# Basic training command
python pe_transformer_final_deepseek.py \
    --data_path /path/to/your/PTPE_control_all_data.txt \
    --output_dir ./output \
    --high_risk_file ./high_risk_features.txt \
    --metabolomics_file ./metab_features_ptpe.txt \
    --proteomics_file ./prot_features_ptpe.txt
2. Training with External Validation
bash
python pe_transformer_final_deepseek.py \
    --data_path /path/to/training_data.txt \
    --external_path /path/to/external_validation.txt \
    --output_dir ./output_with_external \
    --high_risk_file ./high_risk_features.txt \
    --metabolomics_file ./metab_features_ptpe.txt \
    --proteomics_file ./prot_features_ptpe.txt
3. Using Shell Scripts for Batch Execution
bash
# Run PTPE/TPE training on CPU
bash run_train_cpu_ptpe.sh

# Run GW16-specific analysis
bash GW16_analysis.sh

# Run traditional ML comparison
bash ml_comparison.sh
4. Traditional Machine Learning Baseline Comparison
bash
python ml_comparison.py \
    --data_path /path/to/your/data.txt \
    --output_dir ./ml_output
Model Architecture
Multi-Modal Transformer
The framework employs three parallel Transformer encoders for each modality:
plain
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Clinical Risk  │    │  Metabolomics   │    │   Proteomics    │
│   Factors     │    │    Encoder      │    │    Encoder      │
│  (d_model=32) │    │  (d_model=64)   │    │  (d_model=32)   │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                    ┌─────────────┴─────────────┐
                    │      Feature Fusion       │
                    │   (Concatenation + MLP)   │
                    │     Hidden Dim: 64        │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │      Sigmoid Output         │
                    │   (Binary Classification)   │
                    └───────────────────────────┘
Key Components
表格
Component	Description
Positional Encoding	Sinusoidal positional encoding for sequence order
Transformer Encoder	Multi-head self-attention + feed-forward network
CLS Token	Learnable classification token for global representation
Feature Fusion	Concatenation of modality-specific CLS embeddings
Classifier	Two-layer MLP with ReLU activation and dropout
Evaluation Metrics
Discrimination Metrics
AUC-ROC: Area under the receiver operating characteristic curve
Sensitivity (Recall) with 95% Wilson confidence interval
Specificity with 95% Wilson confidence interval
PPV/NPV (Positive/Negative Predictive Value) with 95% CI
Accuracy and F1-score
Calibration Metrics
Brier Score: Mean squared error between predicted probabilities and observed outcomes
Calibration Slope: Linear regression slope of observed vs. predicted probabilities (ideal = 1.0)
Calibration Intercept: Linear regression intercept (ideal = 0.0)
Reliability Diagrams: Visual assessment of probability calibration
Clinical Utility
Decision Curve Analysis (DCA): Net benefit across threshold probability ranges (0.01–0.45)
Comparison against "Treat All" and "Treat None" strategies
Interpretability
SHAP (SHapley Additive exPlanations): Feature importance ranking and stability analysis
Modality Contribution Analysis: Total contribution, mean contribution per feature, and percentage contribution by modality (Clinical Factors / Metabolomics / Proteomics)
Key Features
表格
Feature	Description
Class Imbalance Handling	Configurable BCEWithLogitsLoss with pos_weight, optional Focal Loss, and WeightedRandomSampler
Early Stopping	Patience-based early stopping with learning rate reduction on plateau
Threshold Calibration	Target-specificity-based threshold calibration (default: 90% specificity)
External Validation	Independent dataset evaluation with median imputation for missing values
Bootstrap SHAP Stability	50-iteration bootstrap for robust feature importance estimation
Cross-Modality Ablation	Single-modality and fusion model comparison
Output Files
After training, the following files are generated in the output directory:
plain
output/
├── Clinical-Factors_best_model.pth
├── Metabolomics_best_model.pth
├── Proteomics_best_model.pth
├── Multi-Modal-Fusion_best_model.pth
├── ROC_Curves_Validation_[Disease].png
├── ROC_Curves_External_[Disease].png
├── Calibration_[Model]_[Disease].png
├── DCA_[Model]_[Disease].png
├── SHAP_Summary_Fusion_[Disease].png
├── SHAP_Top30_Fusion_[Disease].png
├── SHAP_Value_Distribution_Fusion_[Disease].png
├── Modality_Importance_[Disease].png
├── SHAP_Stability_Plot_[Disease].png
├── [Model]_Training_[Disease]_Results.txt
├── [Model]_Validation_[Disease]_Results.txt
├── External_[Model]_[Disease]_Results.txt
├── Threshold_Calibration_Validation_[Disease].txt
├── Threshold_Calibration_External_[Disease].txt
└── SHAP_[Model]_Feature_Ranking_[Disease].txt
Simulated Datasets
Two simulated datasets are provided for testing and demonstration:
表格
File	Description	Samples
simulated_PTPE_1000samples.txt	Large-scale simulated dataset with 1000 samples
simulated_PTPE_160samples_EV.txt	Simulated external validation set	(N=160)
These datasets maintain the identical column structure and feature distributions as the original study data, with simulated sample IDs (26B prefix), hospital names, and clinical outcomes.
