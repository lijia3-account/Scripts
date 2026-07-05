#!/bin/bash
/HOME/szfy_whlxy/szfy_whlxy_1/miniconda3/envs/cputorch/bin/python pe_transformer_final_deepseek.py \
  --data_path /HOME/szfy_whlxy/szfy_whlxy_1/AI/data/PTPE_control_all_data.txt \
  --external_path /HOME/szfy_whlxy/szfy_whlxy_1/AI/data/PTPE_control_all_data_EV.txt \
  --output_dir ./results/ptpe \
  --high_risk_file high_risk_features.txt \
  --metabolomics_file metab_features_ptpe.txt \
  --proteomics_file prot_features_ptpe.txt
