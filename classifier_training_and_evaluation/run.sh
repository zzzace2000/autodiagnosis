#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train_mgp_rnn.py --l2_penalty 1e-5
CUDA_VISIBLE_DEVICES=1 python train_mgp_rnn.py --l2_penalty 1e-4
CUDA_VISIBLE_DEVICES=2 python train_mgp_rnn.py --l2_penalty 1e-5 --pos_cls_ratio 5
CUDA_VISIBLE_DEVICES=3 python train_mgp_rnn.py --l2_penalty 1e-5 --num_layers 1

python test_baselines.py --database 0416_10f_6hrs
python test_baselines.py --database 0416_10f_4hrs
python test_baselines.py --database 0416_10f_2hrs

# Train normal forward imputation RNN
CUDA_VISIBLE_DEVICES=3 python train_mgp_rnn.py --l2_penalty 1e-4 --num_hidden 64 --num_layers 1 --rnn_cls ManyToOneRNN \
--mimic_cls MIMIC_ForwardImputation --database 0416_10f_6hrs

# Run the MGP RNN on 2hours and 6 hours
CUDA_VISIBLE_DEVICES=3 python train_mgp_rnn.py --database 0416_10f_2hrs
CUDA_VISIBLE_DEVICES=2 python train_mgp_rnn.py --database 0416_10f_6hrs


CUDA_VISIBLE_DEVICES=3 python train_mgp_rnn.py --database 0416_10f_6hrs --num_layers 1 --num_hidden 64
CUDA_VISIBLE_DEVICES=3 python train_mgp_rnn.py --database 0416_10f_6hrs --num_layers 1 --num_hidden 128

CUDA_VISIBLE_DEVICES=3 python train_mgp_rnn.py --database 0416_10f_6hrs --num_layers 2 --num_hidden 64
CUDA_VISIBLE_DEVICES=3 python train_mgp_rnn.py --database 0416_10f_6hrs --num_layers 3 --num_hidden 32

python -u train_mgp_rnn.py --database 0416_10f_6hrs --random_mode

# Run
./srun.sh -o ../logs/0427-mgp-rnn.log python -u train_mgp_rnn.py --n_mc_smps 15

# Performance not good QQ Run baseline!
#./srun.sh -o ../logs/0428_mimic_discretized_FI_RNN python train_mgp_rnn.py --l2_penalty 1e-4 --num_hidden 64 --num_layers 1 --rnn_cls ManyToOneRNN \
#--mimic_cls MIMIC_ForwardImputation --database 0416_10f_6hrs

# Run a 12 hours ahead and hope to get sth better.
./srun.sh -o ../logs/0428-mgp-rnn.log python -u train_mgp_rnn.py --n_mc_smps 15 --num_hours_pred 12 &

# This one is to train it with subsampled neg case, but val / test are still the same.
./srun.sh -o ../logs/0429-mgp-rnn.log python -u train_mgp_rnn.py --n_mc_smps 15 --num_hours_pred 12 --training_iters 8 &

./srun.sh -o ../logs/0429-mgp-rnn-nh2.log python -u train_mgp_rnn.py --n_mc_smps 15 --num_hours_pred 2 --training_iters 8 &

# Run a 34 features in 6 hours ahead.
./srun.sh -o ../logs/0429-mgp-rnn-nh6-34f.log python -u train_mgp_rnn.py --mimic_cls MIMIC_discretized_database --n_mc_smps 15 --num_hours_pred 6 --training_iters 8 --database norm_34features_hard --num_features 34 --n_covs 0

# Run a Mingie's bao in 34 features and 8 covs
./srun.sh -o ../logs/0504-mgp-rnn-mingie-bao.log python -u train_mgp_rnn.py --mimic_cls MIMIC_discretized_database --n_mc_smps 15 --num_hours_pred 6 --training_iters 8 --database mingie_34features --num_features 34 --n_covs 8

./srun.sh -o ../logs/0508-mgp-rnn-mingie-bao.log python -u train_mgp_rnn.py --mimic_cls MIMIC_window --n_mc_smps 15 --num_hours_pred 6 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8

./srun.sh -o ../logs/0509-mgp-rnn-mingie-bao-lmc3.log python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 15 --num_hours_pred 6 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8

./srun.sh -o ../logs/0510-cache-exp python -u cache_exp.py

# just 1 hour of prediction
python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 15 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 > ../logs/0511-mgp-rnn-mingie-bao-lmc3.log

# Tune parameters with multiple number of hours. This is the standard 6 hours prediction
python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 15 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --num_random_run 15 > ../logs/0511-mgp-rnn-mingie-bao-lmc3-random-test.log

python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 20 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --num_random_run 15 --add_missing --lr 1e-3 --identifier random_try

# Random try
python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 20 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --num_random_run 15 --add_missing --lr 1e-3 --identifier random_try --init_lmc_lengths 0.5 3 12 > ../logs/0512-rnn-random_try.log &

# Run a method that actually records its test performance
python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 20 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 1e-3 --identifier record --init_lmc_lengths 0.5 3 12 --eval_test > ../logs/0512-rnn-record-test.log &

# Run a method that not do the missing ness indicator
python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 20 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --lr 1e-3 --identifier no-missing --init_lmc_lengths 0.5 3 12 --eval_test > ../logs/0512-rnn-no-missingness.log &

# Baseline RNN
python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 20 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 1e-3 --identifier rnn_baseline > ../logs/0512-rnn-real-baseline.log &

# Baseline
python -u test_baselines.py --classifier RF --add_missing &&
python -u test_baselines.py --classifier LR --add_missing

python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 20 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 1e-3 --identifier rnn_baseline --num_random_run 200 > ../logs/0512-rnn-random-run.log &

# Run other 2 GP-RNN hyperparameters search (including dropout rate!)
python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 20 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --num_random_run 50 --add_missing --lr 1e-3 --identifier random_try2 --init_lmc_lengths 0.5 3 12 --output_dir ../models2/ > ../logs/0512-rnn-random_try2.log &

python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 20 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --num_random_run 50 --add_missing --lr 1e-3 --identifier random_try3 --init_lmc_lengths 0.5 3 12 --output_dir ../models3/ > ../logs/0512-rnn-random_try3.log &

# Run a invert missing in normal RNN
python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 20 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --invert_missing --lr 1e-3 --identifier rnn_invert_missing > ../logs/0512-rnn-invert.log &

python -u test_baselines.py --classifier RF --add_missing

# Do with mean imputation
python -u test_baselines.py --classifier RF --add_missing --rnn_imputation mean_imputation &&
python -u test_baselines.py --classifier LR --add_missing --rnn_imputation mean_imputation &&
python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 20 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 5e-4 --identifier rnn_mean --rnn_imputation mean_imputation > ../logs/0513-rnn-mean-baseline.log

# Run a MGP RNN with best auroc to do early stopping
python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 20 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 1e-3 --identifier auroc --init_lmc_lengths 0.5 3 12 --output_dir ../models/ > ../logs/0513-rnn-auroc.log &

# Do a evaluation of all people
python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 20 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 1e-3 --identifier analyze-all --init_lmc_lengths 0.5 3 12 --not_analyze_pos_only --output_dir ../models/ > ../logs/0513-rnn-analyze-all.log &

python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 20 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 1e-3 --identifier rnn_analyze_all --not_analyze_pos_only  > ../logs/0513-rnn-analyze-all.log &

# Do training and evaluation on all people (probably slow...)
python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 20 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 1e-3 --identifier train_all --init_lmc_lengths 0.5 3 12 --not_analyze_pos_only --output_dir ../models/ > ../logs/0514-train-all.log &

python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 20 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 1e-3 --identifier rnn_train_all --not_analyze_pos_only  > ../logs/0514-rnn-train-all.log &

CUDA_VISIBLE_DEVICES=0 python -u cache_exp.py --identifier 0515-15mins-48hrs --RL_interval 0.25 --MIMIC_exp_cls MIMIC_discretized_joint_exp &> ../logs/0515-cache-joint-exp &

CUDA_VISIBLE_DEVICES=0 python -u cache_exp.py --identifier 0515-7_5mins-48hrs-joint-exp --RL_interval 0.125 --MIMIC_exp_cls MIMIC_discretized_joint_exp &> ../logs/0515-cache-joint-exp-7_5mins &

# Run LR and RF on all training cases!
python -u test_baselines.py --classifier RF --add_missing --rnn_imputation mean_imputation --identifier 0516-train-all &> ../logs/0516_train_all_RF &
python -u test_baselines.py --classifier LR --add_missing --rnn_imputation mean_imputation --identifier 0516-train-all &> ../logs/0516_train_all_LR &

# Rerun a MGP-RNN and RNN on the cases where train-only is used
python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 20 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 1e-3 --identifier analyze-pos-only --init_lmc_lengths 0.5 3 12 --analyze_pos_only --output_dir ../models/ > ../logs/0516-mgprnn-analyze-pos-only.log &
python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --rnn_imputation mean_imputation --mimic_cls MIMIC_window --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 1e-3 --identifier rnn-analyze-pos-only --analyze_pos_only --output_dir ../models/ > ../logs/0516-rnn-analyze-pos-only.log &

# Run LR and RF on class weight balance!!
python -u test_baselines.py --classifier RF --add_missing --rnn_imputation mean_imputation --identifier 0516-balanced --class_weight balanced &> ../logs/0516_balanced_RF &
python -u test_baselines.py --classifier LR --add_missing --rnn_imputation mean_imputation --identifier 0516-balanced --class_weight balanced &> ../logs/0516_balanced_LR &

CUDA_VISIBLE_DEVICES=0 python -u cache_exp.py --identifier 0516-30mins-48hrs --RL_interval 0.5 --MIMIC_exp_cls MIMIC_discretized_joint_exp &> ../logs/0516-cache-joint-exp-30mins &

CUDA_VISIBLE_DEVICES=0 python -u cache_exp.py --identifier 0516-60mins-48hrs --RL_interval 1 --MIMIC_exp_cls MIMIC_discretized_joint_exp &> ../logs/0516-cache-joint-exp-60mins &

### BREAK here!!!!!
# TODO: do a table of the classifier performance
    -- I do two things already in the server, but I did not organize them too much.
        1. I run the whole dataset normally
        2. I run a sampling method for the positive cases

# 9/12 Try positive only training with 6 noise features :)
# Run the same parameters
python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 15 --training_iters 30 --database mingie_34features_6noise --num_features 40 --n_covs 8 --add_missing --lr 1e-3 --identifier 34feats_6noise_negsampled --init_lmc_lengths 0.5 3 12 --neg_subsampled --output_dir ../models/ > ../logs/0916-mgprnn-6noise-feats.log &

# Simulated features
python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 10 --training_iters 30 --batch_size 64 --database simulated_6features --num_features 6 --n_covs 0 --lr 1e-3 --identifier simulated_6feats --init_lmc_lengths 1 --output_dir ../models/ > ../logs/0920-mgprnn-simulated-6feats.log &
# --add_missing

# --class_weight balanced
python -u test_baselines.py --database simulated_6features --classifier RF --rnn_imputation mean_imputation --num_features 6 --n_covs 0 --identifier 0922  &> ../logs/0922_RF &
python -u test_baselines.py --database simulated_6features --classifier LR --rnn_imputation mean_imputation --num_features 6 --n_covs 0 --identifier 0922  &> ../logs/0922_LR &

## Cache experience
# Run the baseline
python -u test_baselines.py --classifier RF --add_missing --rnn_imputation mean_imputation --identifier 0928-RF-balanced --class_weight balanced &> ../logs/0928_balanced_RF &
python -u test_baselines.py --classifier RF --add_missing --rnn_imputation mean_imputation --identifier 0928-RF-balanced --class_weight balanced &> ../logs/0928_balanced_RF &
python -u test_baselines.py --classifier LR --add_missing --rnn_imputation mean_imputation --identifier 0516-balanced --class_weight balanced &> ../logs/0516_balanced_LR &

# Train only in classifier on a 12 hour
python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 30 --database mingie_34features_6noise --num_features 40 --n_covs 8 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_6noise_negsampled --output_dir ../models/ --num_random_run 20 > ../logs/1020-rnn-6noise-feats.log &

python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 30 --database mingie_34features_6noise --num_features 40 --n_covs 8 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_6noise_negsampled --output_dir ../models/ --num_random_run 20 --seed 100 > ../logs/1020-rnn-6noise-feats-s100.log &

CUDA_VISIBLE_DEVICES=0 python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 30 --database mingie_34features_6noise --num_features 40 --n_covs 8 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_6noise_negsampled --output_dir ../models/ --num_random_run 20 --seed 200 > ../logs/1020-rnn-6noise-feats-s200.log &

CUDA_VISIBLE_DEVICES=0 python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 30 --database mingie_34features_6noise --num_features 40 --n_covs 8 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_6noise_negsampled --output_dir ../models/ --num_random_run 20 --seed 300 > ../logs/1020-rnn-6noise-feats-s300.log &

# Have not run QAQ!!!!
CUDA_VISIBLE_DEVICES=0 python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 30 --database mingie_34features_6noise_new --num_features 40 --n_covs 8 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_6noise_negsampled --output_dir ../models/ --num_random_run 0 > ../logs/1023-rnn-6noise-new-feats.log &

CUDA_VISIBLE_DEVICES=0 python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 30 --database mingie_34features_6noise_new --num_features 40 --n_covs 8 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_6noise_negsampled --output_dir ../models/ --num_random_run 20 > ../logs/1023-rnn-6noise-new-feats-rand-search-s0.log &

CUDA_VISIBLE_DEVICES=0 python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 30 --database mingie_34features_6noise_new --num_features 40 --n_covs 8 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_6noise_negsampled --output_dir ../models/ --num_random_run 20 --seed 100 > ../logs/1023-rnn-6noise-new-feats-rand-search-s100.log &

CUDA_VISIBLE_DEVICES=0 python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 30 --database mingie_34features_6noise_new --num_features 40 --n_covs 8 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_6noise_negsampled --output_dir ../models/ --num_random_run 20 --seed 200 > ../logs/1023-rnn-6noise-new-feats-rand-search-s200.log &


CUDA_VISIBLE_DEVICES=0 python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_negsampled --output_dir ../models/ --num_random_run 20 --seed 200 > ../logs/1025-rnn-new-feats-rand-search-s200.log &

CUDA_VISIBLE_DEVICES=0 python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_negsampled --output_dir ../models/ --num_random_run 20 --seed 100 --num_hours_pred 12.01 > ../logs/1025-rnn-new-feats-rand-search-s100.log &

CUDA_VISIBLE_DEVICES=0 python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_negsampled --output_dir ../models/ --num_random_run 20 --seed 2 --num_hours_pred 12.01 > ../logs/1025-rnn-new-feats-rand-search-s2.log &

CUDA_VISIBLE_DEVICES=0 python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_negsampled --output_dir ../models/ --num_random_run 20 --seed 222 --num_hours_pred 12.01 > ../logs/1025-rnn-new-feats-rand-search-s222.log &


CUDA_VISIBLE_DEVICES=0 python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 20 --rnn_input_keep_prob 0.9 --rnn_output_keep_prob 0.6 --rnn_state_keep_prob 0.9 --num_hidden 8 --num_layers 1 --l2_penalty 1e-4 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_negsampled --output_dir ../models/  --num_hours_pred 12.01 > ../logs/1025-mgprnn-new-replicate-12hours.log &


python -u test_baselines.py --num_features 40 --classifier RF --add_missing --database mingie_34features_6noise --num_hours_pred 12.01 --identifier 1025-RF-mean-imputation-balanced --rnn_imputation mean_imputation --class_weight balanced &> ../logs/1025-RF-mean-imputation-balanced &

python -u test_baselines.py --num_features 40 --classifier RF --add_missing --database mingie_34features_6noise --num_hours_pred 12.01 --identifier 1025-RF-mean-imputation --rnn_imputation mean_imputation &> ../logs/1025-RF-mean-imputation  &

python -u test_baselines.py --num_features 40 --classifier LR --add_missing --database mingie_34features_6noise --num_hours_pred 12.01 --identifier 1025-LR-mean-imputation-balanced --rnn_imputation mean_imputation --class_weight balanced &> ../logs/1025-LR-mean-imputation-balanced &

python -u test_baselines.py --num_features 40 --classifier LR --add_missing --database mingie_34features_6noise --num_hours_pred 12.01 --identifier 1025-LR-mean-imputation --rnn_imputation mean_imputation &> ../logs/1025-LR-mean-imputation  &


CUDA_VISIBLE_DEVICES=0 python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 40 --rnn_input_keep_prob 0.9 --rnn_output_keep_prob 0.9 --rnn_state_keep_prob 0.6 --num_hidden 8 --num_layers 1 --l2_penalty 1e-4 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 8e-4 --neg_subsampled --identifier 12hours_34feats_negsampled --output_dir ../models/  --num_hours_pred 12.01 > ../logs/1030-mgprnn-new-replicate-12hours.log &

CUDA_VISIBLE_DEVICES=0 python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 40 --rnn_input_keep_prob 0.9 --rnn_output_keep_prob 0.9 --rnn_state_keep_prob 0.6 --num_hidden 8 --num_layers 1 --l2_penalty 1e-4 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 0 --add_missing --lr 8e-4 --neg_subsampled --identifier 12hours_34feats_0cov_negsampled --output_dir ../models/  --num_hours_pred 12.01 > ../logs/1030-mgprnn-new-replicate-12hours-cov0.log &

# The performance is not good! Try a MGP-RNN with the same hyperparams in RNN
CUDA_VISIBLE_DEVICES=0 python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 20 --rnn_input_keep_prob 0.9 --rnn_output_keep_prob 0.7 --rnn_state_keep_prob 0.9 --num_hidden 8 --num_layers 1 --l2_penalty 1e-5 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_negsampled --output_dir ../models/  --num_hours_pred 12.01 > ../logs/1031-mgprnn-new-run-1.log &

CUDA_VISIBLE_DEVICES=0 python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 20 --rnn_input_keep_prob 0.7 --rnn_output_keep_prob 0.7 --rnn_state_keep_prob 0.5 --num_hidden 32 --num_layers 1 --l2_penalty 1e-5 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 8 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_negsampled --output_dir ../models/  --num_hours_pred 12.01 > ../logs/1031-mgprnn-new-run-new.log &

CUDA_VISIBLE_DEVICES=0 python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 20 --rnn_input_keep_prob 0.7 --rnn_output_keep_prob 0.7 --rnn_state_keep_prob 0.5 --num_hidden 32 --num_layers 1 --l2_penalty 1e-5 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 0 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_negsampled_nocov --output_dir ../models/  --num_hours_pred 12.01 > ../logs/1031-mgprnn-new-run-new-no-cov.log &

# Train a RNN w/o any static feature
./srun.sh -o ../logs/1031-rnn-no-cov.log python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --rnn_input_keep_prob 0.7 --rnn_output_keep_prob 0.7 --rnn_state_keep_prob 0.5 --num_hidden 32 --num_layers 1 --l2_penalty 1e-5 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 0 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_negsampled_nocov --output_dir ../models/  --num_hours_pred 12.01 &


CUDA_VISIBLE_DEVICES=0 python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 20 --rnn_input_keep_prob 0.9 --rnn_output_keep_prob 0.7 --rnn_state_keep_prob 0.9 --num_hidden 8 --num_layers 1 --l2_penalty 1e-5 --training_iters 30 --database mingie_34features --num_features 34 --n_covs 0 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_34feats_negsampled_no_cov --output_dir ../models/  --num_hours_pred 12.01 > ../logs/1102-mgprnn-new-run-no-cov.log &

# Do a new MGPRNN!!!!!
./srun.sh -o ../logs/1115-new-bao-rnn.log python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --rnn_input_keep_prob 0.7 --rnn_output_keep_prob 0.7 --rnn_state_keep_prob 0.5 --num_hidden 32 --num_layers 1 --l2_penalty 1e-5 --training_iters 30 --database mingjie_39features_38covs --num_features 39 --n_covs 38 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_39feats_negsampled_38cov --output_dir ../models/  --num_hours_pred 12.01 --overwrite 1 &
CUDA_VISIBLE_DEVICES=0  python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --rnn_input_keep_prob 0.7 --rnn_output_keep_prob 0.7 --rnn_state_keep_prob 0.5 --num_hidden 32 --num_layers 1 --l2_penalty 1e-5 --training_iters 30 --database mingjie_39features_38covs --num_features 39 --n_covs 38 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_39feats_negsampled_38cov --output_dir ../models/  --num_hours_pred 12.01 --overwrite 1 &

# Do a random search
./srun.sh -o ../logs/1115-new-bao-rnn-rand-search-s200.log python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 40 --database mingjie_39features_38covs --num_features 39 --n_covs 38 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_39feats_negsampled_38cov_rand --output_dir ../models/ --num_random_run 20 --seed 200 &
./srun.sh -o ../logs/1115-new-bao-rnn-rand-search-s300.log python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 40 --database mingjie_39features_38covs --num_features 39 --n_covs 38 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_39feats_negsampled_38cov_rand --output_dir ../models/ --num_random_run 20 --seed 300 &
./srun.sh -o ../logs/1115-new-bao-rnn-rand-search-s100.log python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 40 --database mingjie_39features_38covs --num_features 39 --n_covs 38 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_39feats_negsampled_38cov_rand --output_dir ../models/ --num_random_run 20 --seed 100 &
./srun.sh -o ../logs/1115-new-bao-rnn-rand-search-s10.log python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 40 --database mingjie_39features_38covs --num_features 39 --n_covs 38 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_39feats_negsampled_38cov_rand --output_dir ../models/ --num_random_run 20 --seed 10 &

# Do a new database with MGP-RNN!!
./srun.sh -o ../logs/1116-mgprnn-new-run.log python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 20 --rnn_input_keep_prob 0.7 --rnn_output_keep_prob 0.9 --rnn_state_keep_prob 0.7 --num_hidden 64 --num_layers 1 --l2_penalty 1e-7 --training_iters 30 --database mingjie_39features_38covs --num_features 39 --n_covs 38 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_39feats_38cov_negsampled_mgprnn_1 --output_dir ../models/ --num_hours_pred 12.01 &

./srun.sh -o ../logs/1116-mgprnn-new-run2.log python -u train_mgp_rnn.py --rnn_cls ManyToOneLMC_MGP_RNN --mimic_cls MIMIC_window --n_mc_smps 20 --rnn_input_keep_prob 0.9 --rnn_output_keep_prob 0.7 --rnn_state_keep_prob 0.5 --num_hidden 128 --num_layers 2 --l2_penalty 1e-7 --training_iters 30 --database mingjie_39features_38covs --num_features 39 --n_covs 38 --add_missing --lr 1e-3 --neg_subsampled --identifier 12hours_39feats_38cov_negsampled_mgprnn_2 --output_dir ../models/ --num_hours_pred 12.01 &

# Train RNN regression for transition dynamics
./srun.sh -o ../logs/0118-new-bao-rnn-trans-model-rand-s0 python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN_Regressor --mimic_cls MIMIC_window --metric 'r2' --n_mc_smps 1 --training_iters 40 --database mingjie_39features_38covs --num_features 39 --n_covs 38 --add_missing --lr 1e-3 --neg_subsampled --identifier trans_12hours_39feats_negsampled_38cov_rand --output_dir ../models/ --num_random_run 20 --seed 0 &
./srun.sh -o ../logs/0118-new-bao-rnn-trans-model-rand-s100 python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN_Regressor --mimic_cls MIMIC_window --metric 'r2' --n_mc_smps 1 --training_iters 40 --database mingjie_39features_38covs --num_features 39 --n_covs 38 --add_missing --lr 1e-3 --neg_subsampled --identifier trans_12hours_39feats_negsampled_38cov_rand --output_dir ../models/ --num_random_run 20 --seed 100 &
./srun.sh -o ../logs/0118-new-bao-rnn-trans-model-rand-s200 python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN_Regressor --mimic_cls MIMIC_window --metric 'r2' --n_mc_smps 1 --training_iters 40 --database mingjie_39features_38covs --num_features 39 --n_covs 38 --add_missing --lr 1e-3 --neg_subsampled --identifier trans_12hours_39feats_negsampled_38cov_rand --output_dir ../models/ --num_random_run 20 --seed 200 &
./srun.sh -o ../logs/0118-new-bao-rnn-trans-model-rand-s300 python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN_Regressor --mimic_cls MIMIC_window --metric 'r2' --n_mc_smps 1 --training_iters 40 --database mingjie_39features_38covs --num_features 39 --n_covs 38 --add_missing --lr 1e-3 --neg_subsampled --identifier trans_12hours_39feats_negsampled_38cov_rand --output_dir ../models/ --num_random_run 20 --seed 300 &


# The best is ../models/1116-12hours_39feats_negsampled_38cov_rand-mimic-nh128-nl2-c1e-07-keeprate0.9_0.7_0.5-npred12-miss1-n_mc_1-MIMIC_window-mingjie_39features_38covs-ManyToOneRNN

# Train a RNN with 24 hours as prediction :)
./srun.sh -o ../logs/0117-rnn-24hours-run.log python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --rnn_input_keep_prob 0.9 --rnn_output_keep_prob 0.7 --rnn_state_keep_prob 0.5 --num_hidden 128 --num_layers 2 --l2_penalty 1e-7 --training_iters 40 --database mingjie_39features_38covs --num_features 39 --n_covs 38 --add_missing --lr 1e-3 --neg_subsampled --identifier 24hours_39feats_38cov_negsampled_rnn --output_dir ../models/ --num_hours_pred 24.01 &

# Train a RNN with random search as 24 hrs!
./srun.sh -o ../logs/0117-rnn-rand-search-s10.log python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 40 --database mingjie_39features_38covs --num_features 39 --n_covs 38 --add_missing --lr 1e-3 --neg_subsampled --identifier 24hours_39feats_negsampled_38cov_rand --output_dir ../models/ --num_random_run 40 --seed 10 &


# Run baselines!
python -u test_baselines.py --num_features 39 --classifier RF --add_missing --database mingjie_39features_38covs --num_hours_pred 24.01 --identifier 0122-RF-mean-imputation-balanced --rnn_imputation mean_imputation --class_weight balanced &> ../logs/0122-RF-mean-imputation-balanced &

python -u test_baselines.py --num_features 39 --classifier RF --add_missing --database mingjie_39features_38covs --num_hours_pred 24.01 --identifier 0122-RF-mean-imputation --rnn_imputation mean_imputation &> ../logs/0122-RF-mean-imputation  &

python -u test_baselines.py --num_features 39 --classifier LR --add_missing --database mingjie_39features_38covs --num_hours_pred 24.01 --identifier 0122-LR-mean-imputation-balanced --rnn_imputation mean_imputation --class_weight balanced &> ../logs/0122-LR-mean-imputation-balanced &

python -u test_baselines.py --num_features 39 --classifier LR --add_missing --database mingjie_39features_38covs --num_hours_pred 24.01 --identifier 0122-LR-mean-imputation --rnn_imputation mean_imputation &> ../logs/0122-LR-mean-imputation  &

# Run a RNN survival model with 12 hours
CUDA_VISIBLE_DEVICES=0  python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN_Survival --mimic_cls MIMIC_window --n_mc_smps 1 --rnn_input_keep_prob 0.7 --rnn_output_keep_prob 0.7 --rnn_state_keep_prob 0.5 --num_hidden 32 --num_layers 1 --l2_penalty 1e-5 --training_iters 30 --database mingjie_39features_38covs --num_features 39 --n_covs 38 --add_missing --lr 1e-3 --neg_subsampled --identifier surv_12hours_39f38c_negsampled --output_dir ../models/  --num_hours_pred 12.01 --overwrite 1

# Train a survival model in 12 hrs
./srun.sh -o ../logs/0302-rnn-surv-rand-search-s10.log python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN_Survival --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 40 --database mingjie_39features_38covs --num_features 39 --n_covs 38 --add_missing --lr 1e-3 --neg_subsampled --identifier surv_12hours_39f38c_negsampled_rand --output_dir ../models/ --num_random_run 40 --seed 10 &

for seed in 100 200 300 400
do
./srun.sh -o ../logs/0302-rnn-surv-rand-search-s${seed}.log python -u train_mgp_rnn.py --rnn_cls ManyToOneRNN_Survival --mimic_cls MIMIC_window --n_mc_smps 1 --training_iters 40 --database mingjie_39features_38covs --num_features 39 --n_covs 38 --add_missing --lr 1e-3 --neg_subsampled --identifier surv_12hours_39f38c_negsampled_rand --output_dir ../models/ --num_random_run 10 --seed ${seed} &
done

python train_mgp_rnn.py \
    --database_dir ../data/ \
    --database mocked_data \
    --num_hours_warmup 0 \
    --min_measurements_in_warmup 0 \
    --neg_subsampled 0
