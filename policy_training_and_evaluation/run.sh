

for i in {1..6}
do
    python run_k_tails_dqn.py &
done

python run_k_tails_dqn.py --my_identifier 1024-rand-try-mgp-rnn --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --action_cost_coef 1e-4 --gain_coef 10 --pos_label_fold_coef 1 --only_pos_reward 1 \
--cache_dir ../RL_exp_cache/1020-15mins-24hrs-joint-indep-measurement/ --rl_state_dim 16

python run_k_tails_dqn.py --my_identifier 1024-rand-try-mgp-rnn --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --action_cost_coef 1e-3 --gain_coef 10 --pos_label_fold_coef 100 --only_pos_reward 1 \
--cache_dir ../RL_exp_cache/1020-15mins-24hrs-joint-indep-measurement/ --rl_state_dim 16 --rand 1 &

python run_k_tails_dqn.py --my_identifier 1024-rand-try-mgp-rnn-g0.1 --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --action_cost_coef 1e-3 --gain_coef 10 --pos_label_fold_coef 100 --only_pos_reward 1 \
--cache_dir ../RL_exp_cache/1020-15mins-24hrs-joint-indep-measurement/ --rl_state_dim 16 --gamma 0.1 &

python run_k_tails_dqn.py --my_identifier 1024-rand-try-mgp-rnn-g0.3 --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --action_cost_coef 1e-3 --gain_coef 10 --pos_label_fold_coef 100 --only_pos_reward 1 \
--cache_dir ../RL_exp_cache/1020-15mins-24hrs-joint-indep-measurement/ --rl_state_dim 16 --gamma 0.3 &

python run_k_tails_dqn.py --my_identifier 1024-rand-try-mgp-rnn-g0.5 --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --action_cost_coef 1e-3 --gain_coef 10 --pos_label_fold_coef 100 --only_pos_reward 1 \
--cache_dir ../RL_exp_cache/1020-15mins-24hrs-joint-indep-measurement/ --rl_state_dim 16 --gamma 0.5 &

python run_k_tails_dqn.py --my_identifier 1024-rand-try-mgp-rnn-g0.1 --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --action_cost_coef 1e-3 --gain_coef 10 --pos_label_fold_coef 100 --only_pos_reward 1 \
--cache_dir ../RL_exp_cache/1020-15mins-24hrs-joint-indep-measurement/ --rl_state_dim 16 --gamma 0.1 &

python run_k_tails_dqn.py --my_identifier 1024-rand-try-mgp-rnn-g0.8 --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --action_cost_coef 1e-3 --gain_coef 10 --pos_label_fold_coef 100 --only_pos_reward 1 \
--cache_dir ../RL_exp_cache/1020-15mins-24hrs-joint-indep-measurement/ --rl_state_dim 16 --gamma 0.8 &

python run_k_tails_dqn.py --my_identifier 1024-rand-try-mgp-rnn-g0.1-a1e-2 --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --action_cost_coef 1e-2 --gain_coef 10 --pos_label_fold_coef 100 --only_pos_reward 1 \
--cache_dir ../RL_exp_cache/1020-15mins-24hrs-joint-indep-measurement/ --rl_state_dim 16 --gamma 0.1 &

python run_k_tails_dqn.py --my_identifier 1024-rand-try-mgp-rnn-g0.1-a0.1 --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --action_cost_coef 0.1 --gain_coef 10 --pos_label_fold_coef 100 --only_pos_reward 1 \
--cache_dir ../RL_exp_cache/1020-15mins-24hrs-joint-indep-measurement/ --rl_state_dim 16 --gamma 0.1 &

python run_k_tails_dqn.py --my_identifier 1024-rand-try-mgp-rnn-g0.1-a1 --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --action_cost_coef 1 --gain_coef 10 --pos_label_fold_coef 100 --only_pos_reward 1 \
--cache_dir ../RL_exp_cache/1020-15mins-24hrs-joint-indep-measurement/ --rl_state_dim 16 --gamma 0.1 &

python run_k_tails_dqn.py --my_identifier 1024-rand-try-mgp-rnn-g0.1-a1e-4 --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-4 --action_cost_coef 1e-4 --gain_coef 10 --pos_label_fold_coef 100 --only_pos_reward 1 \
--cache_dir ../RL_exp_cache/1020-15mins-24hrs-joint-indep-measurement/ --rl_state_dim 16 --gamma 0.1 &

python run_k_tails_dqn.py --my_identifier 1024-rand-try-mgp-rnn-g0.1 --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --action_cost_coef 1e-3 --gain_coef 10 --pos_label_fold_coef 100 --only_pos_reward 1 \
--cache_dir ../RL_exp_cache/1020-15mins-24hrs-joint-indep-measurement/ --rl_state_dim 16 --gamma 0.1 \
--action_cost_coef 1e-3 --num_hours_pred 12.01 &

python run_k_tails_dqn.py --my_identifier 1024-rand-try-mgp-rnn-g0.1 --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --action_cost_coef 1e-3 --gain_coef 10 --pos_label_fold_coef 100 --only_pos_reward 1 \
--cache_dir ../RL_exp_cache/1020-15mins-24hrs-joint-indep-measurement/ --rl_state_dim 16 --gamma 0.1 \
--action_cost_coef 1e-3 --num_hours_pred 12.01 &

for action_cost in "1e-3" "1e-4" "1e-5"
do
echo ${action_cost}
python run_k_tails_dqn.py --my_identifier 1025-grid-search --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --cache_dir ../RL_exp_cache/1023-15mins-24hrs-joint-indep-measurement-rnn/ --rl_state_dim 32 \
--pos_label_fold_coef 1 --only_pos_reward 1 --normalized_state 1 --gamma 1 --action_cost_coef ${action_cost} &
done

for gamma in "0.99" "0.5" "0.1"
do
echo ${gamma}
python run_k_tails_dqn.py --my_identifier 1025-grid-search --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --cache_dir ../RL_exp_cache/1023-15mins-24hrs-joint-indep-measurement-rnn/ --rl_state_dim 32 \
--pos_label_fold_coef 1 --only_pos_reward 1 --normalized_state 1 --gamma ${gamma} --action_cost_coef 0.01 &

action_cost_coef=0.05
gamma=0.5done

python run_k_tails_dqn.py --my_identifier 1025-grid-search --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --cache_dir ../RL_exp_cache/1023-15mins-24hrs-joint-indep-measurement-rnn/ --rl_state_dim 32 \
--pos_label_fold_coef 1 --only_pos_reward 1 --normalized_state 1 --gamma ${gamma} --action_cost_coef ${action_cost_coef} &

action_cost_coef=0.1
gamma=0.2
python run_k_tails_dqn.py --my_identifier 1025-grid-search --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --cache_dir ../RL_exp_cache/1023-15mins-24hrs-joint-indep-measurement-rnn/ --rl_state_dim 32 \
--pos_label_fold_coef 1 --only_pos_reward 1 --normalized_state 1 --gamma ${gamma} --action_cost_coef ${action_cost_coef} &

action_cost_coef=0.5
gamma=0.2
python run_k_tails_dqn.py --my_identifier 1025-grid-search --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --cache_dir ../RL_exp_cache/1023-15mins-24hrs-joint-indep-measurement-rnn/ --rl_state_dim 32 \
--pos_label_fold_coef 1 --only_pos_reward 1 --normalized_state 1 --gamma ${gamma} --action_cost_coef ${action_cost_coef} &


action_cost_coef=0.01
for gamma in "0.85" "0.7" "0.5"
do
python run_k_tails_dqn.py --my_identifier 1025-grid-search --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --cache_dir ../RL_exp_cache/1023-15mins-24hrs-joint-indep-measurement-rnn/ --rl_state_dim 32 \
--pos_label_fold_coef 1 --only_pos_reward 1 --normalized_state 1 --gamma ${gamma} --action_cost_coef ${action_cost_coef} &
done

gamma=0.2
for action_cost_coef in "0.02" "0.01" "0.005"
do
python run_k_tails_dqn.py --my_identifier 1025-grid-search --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --cache_dir ../RL_exp_cache/1023-15mins-24hrs-joint-indep-measurement-rnn/ --rl_state_dim 32 \
--pos_label_fold_coef 1 --only_pos_reward 1 --normalized_state 1 --gamma ${gamma} --action_cost_coef ${action_cost_coef} &
done

gamma=0.5
for action_cost_coef in "0.005"
do
python run_k_tails_dqn.py --my_identifier 1025-grid-search --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --cache_dir ../RL_exp_cache/1023-15mins-24hrs-joint-indep-measurement-rnn/ --rl_state_dim 32 \
--pos_label_fold_coef 1 --only_pos_reward 1 --normalized_state 1 --gamma ${gamma} --action_cost_coef ${action_cost_coef} &
done

action_cost_coef=0.02
gamma=0.95
for pos_label_fold_coef in "1" "10" "100"
do
python run_k_tails_dqn.py --my_identifier 1025-grid-search --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --cache_dir ../RL_exp_cache/1023-15mins-24hrs-joint-indep-measurement-rnn/ --rl_state_dim 32 \
--pos_label_fold_coef ${pos_label_fold_coef} --only_pos_reward 1 --normalized_state 1 --gamma ${gamma} --action_cost_coef ${action_cost_coef} &
done


pos_label_fold_coef=10
gamma=0.95
for action_cost_coef in "0.05" "0.1" "0.15"
do
python run_k_tails_dqn.py --my_identifier 1025-grid-search --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --cache_dir ../RL_exp_cache/1023-15mins-24hrs-joint-indep-measurement-rnn/ --rl_state_dim 32 \
--pos_label_fold_coef ${pos_label_fold_coef} --only_pos_reward 1 --normalized_state 1 --gamma ${gamma} --action_cost_coef ${action_cost_coef} &
done

pos_label_fold_coef=1
gamma=0.95
for action_cost_coef in "0.008" "0.01" "0.02"
do
python run_k_tails_dqn.py --my_identifier 1025-grid-search --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --cache_dir ../RL_exp_cache/1023-15mins-24hrs-joint-indep-measurement-rnn/ --rl_state_dim 32 \
--pos_label_fold_coef ${pos_label_fold_coef} --only_pos_reward 0 --normalized_state 1 --gamma ${gamma} --action_cost_coef ${action_cost_coef} &
done

# !!!! action costs!! Run all actions on the ac0.01 pos_only 0 gamma 0.95
action_cost_coef=0.01
gamma=0.95
pos_label_fold_coef=30
only_pos_reward=1
python run_k_tails_dqn.py --my_identifier 1025-grid-search --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --cache_dir ../RL_exp_cache/1024-15mins-24hrs-joint-indep-measurement-rnn-all-pataient/ --rl_state_dim 32 --pos_label_fold_coef ${pos_label_fold_coef} --only_pos_reward ${only_pos_reward} --normalized_stat 1 --gamma ${gamma} --action_cost_coef ${action_cost_coef} --train_batch_size 1024 &

gamma=0.95
pos_label_fold_coef=1
only_pos_reward=1
#for action_cost_coef in "0.005" "0.01" "0.015"
for action_cost_coef in "0.2" "0.5" "0.8"
do
echo ${action_cost_coef}
python run_k_tails_dqn.py --my_identifier 1026-per-action-cost --normalized_state 1 --lr 2e-5 \
--reg_constant 1e-3 --cache_dir ../RL_exp_cache/1024-15mins-24hrs-joint-indep-measurement-rnn-all-pataient/ --rl_state_dim 32 --pos_label_fold_coef ${pos_label_fold_coef} --only_pos_reward ${only_pos_reward} --normalized_stat 1 --gamma ${gamma} --action_cost_coef ${action_cost_coef} --train_batch_size 1024 --load_per_action_cost 1 &
done


# Run DQN with sequential caching records...
python run_k_tails_dqn.py --my_identifier 1104- --normalized_state 0 --lr 2e-5 --reg_constant 1e-3 --cache_dir ../RL_exp_cache/1104-15mins-24hrs-random-order-mgp-rnn-neg_sampled/ --rl_state_dim 32 --pos_label_fold_coef ${pos_label_fold_coef} --only_pos_reward ${only_pos_reward} --normalized_stat 1 --gamma ${gamma} --action_cost_coef ${action_cost_coef} --train_batch_size 1024 --load_per_action_cost 1 &

python run_k_tails_dqn.py --my_identifier 1104- --normalized_state 0 --lr 2e-5 --reg_constant 1e-3 --cache_dir ../RL_exp_cache/1104-15mins-24hrs-random-order-mgp-rnn-neg_sampled/ --rl_state_dim 32 --pos_label_fold_coef ${pos_label_fold_coef} --only_pos_reward ${only_pos_reward} --normalized_stat 1 --gamma ${gamma} --action_cost_coef ${action_cost_coef} --train_batch_size 1024 --load_per_action_cost 1 &

# Run sequential DQN record
ac=0.
for lr in 1e-4 1e-5 1e-6
do
./srun.sh -o ../logs/1120_dqn_ac${ac}_lr${lr} python -u run_sequential_dqn.py --my_identifier 1120_ac${ac}_lr${lr} --debug 0 --replace 1 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --normalized_state 1 --num_shared_all_layers 2 --num_shared_dueling_layers 1 --num_hidden_units 32 --lr ${lr} --reg_constant 1e-3 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1117-15mins-24hrs-random-order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 --load_per_action_cost 1 &
done

# Find suitable action cost...
lr=1e-5
for ac in 1e-3 2e-3 5e-3
do
python -u run_sequential_dqn.py --my_identifier 1120_ac${ac}_lr${lr} --debug 0 --replace 1 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --normalized_state 1 --num_shared_all_layers 2 --num_shared_dueling_layers 1 --num_hidden_units 32 --lr ${lr} --reg_constant 1e-3 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1117-15mins-24hrs-random-order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 --load_per_action_cost 1 &> ../logs/1120_dqn_ac${ac}_lr${lr} &
done

# Run a ktail DQN with random search
./srun.sh -o ../logs/1121_dqn_ac0_rand4 python -u run_sequential_dqn.py --my_identifier 1121_ac0_rand --rand 1 --num_random_run 20 --seed 3 --debug 0 --replace 0 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --cache_dir ../RL_exp_cache/1117-15mins-24hrs-random-order-rnn-neg_sampled/ --action_cost_coef 0 --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 --load_per_action_cost 1 &
./srun.sh -o ../logs/1121_dqn_ac0_rand2 python -u run_sequential_dqn.py --my_identifier 1121_ac0_rand --rand 1 --num_random_run 20 --seed 10 --debug 0 --replace 0 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --cache_dir ../RL_exp_cache/1117-15mins-24hrs-random-order-rnn-neg_sampled/ --action_cost_coef 0 --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 --load_per_action_cost 1 &
./srun.sh -o ../logs/1121_dqn_ac0_rand5 python -u run_sequential_dqn.py --my_identifier 1121_ac0_rand --rand 1 --num_random_run 20 --seed 700 --debug 0 --replace 0 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --cache_dir ../RL_exp_cache/1117-15mins-24hrs-random-order-rnn-neg_sampled/ --action_cost_coef 0 --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 --load_per_action_cost 1 &


python -u run_sequential_dqn.py --my_identifier 1121_ac0_rand --rand 0 --debug 1 --num_random_run 20 --seed 505 --debug 0 --replace 0 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --cache_dir ../RL_exp_cache/1117-15mins-24hrs-random-order-rnn-neg_sampled/ --action_cost_coef 0 --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 --load_per_action_cost 1 &> ../logs/1121_dqn_ac0_rand6 &
python -u run_sequential_dqn.py --my_identifier 1121_ac0_rand --rand 1 --num_random_run 20 --seed 510 --debug 0 --replace 0 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --cache_dir ../RL_exp_cache/1117-15mins-24hrs-random-order-rnn-neg_sampled/ --action_cost_coef 0 --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 --load_per_action_cost 1 &> ../logs/1121_dqn_ac0_rand7 &


# DQN model might be good here
# ../models/dqn_mimic-1121_ac0_rand-g1-ac0.0-fold1.0-only_pos0-sd167-ad40-gamma0.9-nn-10000-1-1-32-lr-0.005-0.001-0.7-s-512-3000-i-50-500-3-1/test_
# test_loss: 0.052
for ac in 0 1e-4 1e-3 5e-3 0.01 0.05
do
./srun.sh -o ../logs/1122_dqn_ac${ac} python -u run_sequential_dqn.py --my_identifier 1122_ac${ac} --replace 1 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 32 --lr 0.005 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1117-15mins-24hrs-random-order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
done

# - Try new caching with new visualization about histograms...
for ac in 0 1e-4 1e-3 5e-3 0.01 0.05
do
./srun.sh -o ../logs/1124_dqn_ac${ac}_some_arch python -u run_sequential_dqn.py --my_identifier 1124_random_order_ac${ac} --replace 1 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1124-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
done

# Seems that lr too large. All the validation loss just keep going up...
ac=5e-4
./srun.sh -o ../logs/1124_dqn_ac${ac}_random_search1 python -u run_sequential_dqn.py --my_identifier 1124_random_order_search --rand 1 --num_random_run 20 --seed 3 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1124-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
./srun.sh -o ../logs/1124_dqn_ac${ac}_random_search2 python -u run_sequential_dqn.py --my_identifier 1124_random_order_search --rand 1 --num_random_run 20 --seed 30 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1124-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
./srun.sh -o ../logs/1124_dqn_ac${ac}_random_search3 python -u run_sequential_dqn.py --my_identifier 1124_random_order_search --rand 1 --num_random_run 20 --seed 40 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1124-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
./srun.sh -o ../logs/1124_dqn_ac${ac}_random_search4 python -u run_sequential_dqn.py --my_identifier 1124_random_order_search --rand 1 --num_random_run 20 --seed 400 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1124-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &



./srun.sh -o ../logs/1121_dqn_ac0_rand4 python -u run_sequential_dqn.py --my_identifier 1121_ac0_rand --rand 1 --num_random_run 20 --seed 3 --debug 0 --replace 0 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --cache_dir ../RL_exp_cache/1117-15mins-24hrs-random-order-rnn-neg_sampled/ --action_cost_coef 0 --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 --load_per_action_cost 1 &

ac=5e-4
./srun.sh -o ../logs/1125_dqn_ac${ac}_random_search1 python -u run_sequential_dqn.py --my_identifier 1125_random_order_search --rand 1 --num_random_run 20 --seed 3 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1124-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
./srun.sh -o ../logs/1125_dqn_ac${ac}_random_search2 python -u run_sequential_dqn.py --my_identifier 1125_random_order_search --rand 1 --num_random_run 20 --seed 30 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1124-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
./srun.sh -o ../logs/1125_dqn_ac${ac}_random_search3 python -u run_sequential_dqn.py --my_identifier 1125_random_order_search --rand 1 --num_random_run 20 --seed 40 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1124-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
./srun.sh -o ../logs/1125_dqn_ac${ac}_random_search4 python -u run_sequential_dqn.py --my_identifier 1125_random_order_search --rand 1 --num_random_run 20 --seed 400 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.9 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1124-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &

# Rerun to fix gamma bug for the thing :)
ac=5e-4
./srun.sh -o ../logs/1215_dqn_ac${ac}_random_search1 python -u run_sequential_dqn.py --my_identifier 1215_random_order_search --rand 1 --num_random_run 20 --seed 3 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1128-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
./srun.sh -o ../logs/1215_dqn_ac${ac}_random_search2 python -u run_sequential_dqn.py --my_identifier 1215_random_order_search --rand 1 --num_random_run 20 --seed 30 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1128-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
./srun.sh -o ../logs/1215_dqn_ac${ac}_random_search3 python -u run_sequential_dqn.py --my_identifier 1215_random_order_search --rand 1 --num_random_run 20 --seed 40 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1128-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
./srun.sh -o ../logs/1215_dqn_ac${ac}_random_search4 python -u run_sequential_dqn.py --my_identifier 1215_random_order_search --rand 1 --num_random_run 20 --seed 400 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1128-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &

# Remove the normalization of state: I think the random order will bias sample statistics and not the same as

for f in ../models/dqn_mimic-1215_random_order_search*/; do
    echo $f
    python run_value_estimator_regression_based.py --identifier 1221_ --policy_dir $f
done

## 12/21 bug. Rerun!!!
ac=5e-4
./srun.sh -o ../logs/1221_dqn_ac${ac}_random_search1_rerun python -u run_sequential_dqn.py --my_identifier 1221_random_order_search --rand 1 --num_random_run 20 --seed 3 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1128-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
./srun.sh -o ../logs/1221_dqn_ac${ac}_random_search2 python -u run_sequential_dqn.py --my_identifier 1221_random_order_search --rand 1 --num_random_run 20 --seed 30 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1128-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
./srun.sh -o ../logs/1221_dqn_ac${ac}_random_search3 python -u run_sequential_dqn.py --my_identifier 1221_random_order_search --rand 1 --num_random_run 20 --seed 40 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1128-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
./srun.sh -o ../logs/1221_dqn_ac${ac}_random_search4 python -u run_sequential_dqn.py --my_identifier 1221_random_order_search --rand 1 --num_random_run 20 --seed 400 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1128-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &

for f in ../models/dqn_mimic-1221_random_order_search*/; do
    echo $f
    python run_value_estimator_regression_based.py --identifier 1221_ --policy_dir $f
done

# Run the policy with different action costs w/ best promising results
# The best: ../models/dqn_mimic-1221_random_order_search-g1-ac5.0e-04-gamma0.95-fold1.0-only_pos0-sd167-ad40-nn-10000-3-1-256-lr-0.0001-reg-0.0001-0.5-s-256-5000-i-50-500-3-1

for ac in 0 1e-5 5e-5 1e-4 2e-4 1e-3 2e-3 0.01
do
./srun.sh -o ../logs/1228_dqn_ac${ac}_roc_curve python -u run_sequential_dqn.py --my_identifier 1228_roc_curve --rand 1 --num_random_run 20 --seed 400 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 3 --num_shared_dueling_layers 1 --num_hidden_units 256 --lr 0.0001 --reg_constant 0.0001 --keep_prob 0.5 --cache_dir ../RL_exp_cache/1128-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
done

## I mess up by having a random flag! So rerun by making the arch deterministic
for ac in 0 1e-5 5e-5 1e-4 2e-4 1e-3 2e-3 0.01
do
./srun.sh -o ../logs/0101_dqn_ac${ac}_roc_curve python -u run_sequential_dqn.py --my_identifier 0101_roc_curve --rand 0 --num_random_run 20 --seed 400 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 3 --num_shared_dueling_layers 1 --num_hidden_units 256 --lr 0.0001 --reg_constant 0.0001 --keep_prob 0.5 --cache_dir ../RL_exp_cache/1128-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
done

# Run value estimator
for f in ../models/dqn_mimic-0101_roc_curve*/; do
    echo $f
    python run_value_estimator_regression_based.py --identifier 0101_ --policy_dir $f
done

## I mess up by having a terrible mistake that the action cost is not negative...
ac=5e-4
for idx in 1 2 3 4; do
./srun.sh -o ../logs/0107_dqn_ac${ac}_random_search${idx}_rerun python -u run_sequential_dqn.py --my_identifier 0107_random_order_search --rand 1 --num_random_run 20 --seed ${idx} --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1128-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
done

# Parallel estimate the folder
N=4
task(){
   sleep 0.5; ./srun.sh -o logs/0107_$2_val_est python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 0107_ --mode 1 ;
}
(
for f in ../models/dqn_mimic-0107*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)

# Run in vws39
for f in ../models/dqn_mimic-0107_random_order_search*/; do
    echo $f
    python run_value_estimator_regression_based.py --identifier 0107_ --policy_dir $f --mode 1
done

# Do rerun again
ac=5e-4
for idx in 1 4; do
./srun.sh -o ../logs/0107_dqn_ac${ac}_random_search${idx}_rerun python -u run_sequential_dqn.py --my_identifier 0107_random_order_search --rand 1 --num_random_run 20 --seed ${idx} --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1128-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512 &
done

# Estimate
N=4
task(){
   sleep 0.5; python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 0107_ --mode 1 ;
}
(
for f in ../models/dqn_mimic-0107*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)

## This is the best
# dqn_mimic-0107_random_order_search-g1-ac5.0e-04-gamma0.95-fold1.0-only_pos0-sd167-ad40-nn-10000-3-1-128-lr-0.0001-reg-0.001-0.7-s-256-5000-i-50-500-3-1
#for ac in 0 1e-5 5e-5 1e-4 2e-4 1e-3 2e-3 0.01 0.05
for ac in 1
do
./srun.sh -o ../logs/0108_dqn_ac${ac}_roc_curve python -u run_sequential_dqn.py --my_identifier 0108_roc_curve --rand 0 --seed 1 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 3 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.0001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1128-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --train_batch_size 512
#./srun.sh -o ../logs/0108_val_est_ac${ac} python ./run_value_estimator_regression_based.py --policy_dir  --identifier 0108_ --mode 1
done

# Estimate
N=4
task(){
   sleep 0.5; python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 0108_ --mode 1 ;
}
(
for f in ../models/dqn_mimic-0108*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)

# Estimate an ac1
python ./run_value_estimator_regression_based.py --policy_dir ../models/dqn_mimic-0108_roc_curve-g1-ac1.0e+00-gamma0.95-fold1.0-only_pos0-sd167-ad40-nn-10000-3-1-128-lr-0.0001-reg-0.001-0.7-s-512-5000-i-50-500-3-1/ --identifier 0108_ --mode 1

# Run an only_pos_reward
only_pos_reward=1
for ac in 0 1e-5 1e-4 1e-3 0.01 0.1 1
do
./srun.sh -o ../logs/0109_dqn_ac${ac}_roc_curve python -u run_sequential_dqn.py --my_identifier 0109_roc_curve --rand 0 --seed 1 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 3 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.0001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1128-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward ${only_pos_reward} --train_batch_size 512 &
done

N=4
task(){
   sleep 0.5; python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 0109_ --mode 1 ;
}
(
for f in ../models/dqn_mimic-0109*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)

# Run per time evaluation for all the policy in 0109_ which has different ac coef & only_pos_reward=1
for f in ../models/dqn_mimic-0109*/; do
    echo $f
    python run_value_estimator_regression_based.py --identifier 0109_per_time_ --policy_dir $f --mode 3
done

for f in ../models/dqn_mimic-0108*/; do
    echo $f
    python run_value_estimator_regression_based.py --identifier 0108_per_time_ --policy_dir $f --mode 3
done

for f in ../models/dqn_mimic-0107*/; do
    echo $f
    python run_value_estimator_regression_based.py --identifier 0107_per_time_ --policy_dir $f --mode 3
done

# 20190118 ---------------------
# train different reward estimator and transition model
cache_dir='../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled/'
mimic_exp_cls='MIMIC_cache_discretized_exp_env_v3'
model_type='NNRegressor'
model_use='rew_model'

# [cur_state, cur_action] -> [next_state - cur_state]
proc_exp_func='process_exp_func_trans_model_timepoint_latent_latent_normal_regressor'
input_dim=167
output_dim=128
normalized_state=1
identifier='per_time_latent_latent'
log_fname='../logs/0118_per_time_rew_m_per_time_latent_latent_1'
./srun.sh -o $log_fname -u python run_reward_estimator.py --identifier $identifier \
    --model_type $model_type --model_use $model_use --proc_exp_func $proc_exp_func \
    --normalized_state $normalized_state --input_dim $input_dim --output_dim $output_dim \
    --rand 1 &

log_fname='../logs/0118_per_time_rew_m_per_time_latent_latent_2'
./srun.sh -o $log_fname -u python run_reward_estimator.py --identifier $identifier \
    --model_type $model_type --model_use $model_use --proc_exp_func $proc_exp_func \
    --normalized_state $normalized_state --input_dim $input_dim --output_dim $output_dim \
    --rand 1 &

log_fname='../logs/0118_per_time_rew_m_per_time_latent_latent_3'
./srun.sh -o $log_fname -u python run_reward_estimator.py --identifier $identifier \
    --model_type $model_type --model_use $model_use --proc_exp_func $proc_exp_func \
    --normalized_state $normalized_state --input_dim $input_dim --output_dim $output_dim \
    --rand 1 &

# [cur_obs, cur_action] -> [next_state - cur_state]
proc_exp_func='process_exp_func_trans_model_timepoint_obs_latent_normal_regressor'
input_dim=78
output_dim=128
normalized_state=0
./srun.sh -o $log_fname -u python run_reward_estimator.py --identifier $identifier \
    --model_type $model_type --model_use $model_use --proc_exp_func $proc_exp_func \
    --normalized_state $normalized_state --input_dim $input_dim --output_dim $output_dim \
    --rand 1 &

# [cur_state, cur_action] -> [next_obs - cur_obs]
proc_exp_func='process_exp_func_trans_model_timepoint_latent_obs_normal_regressor'
input_dim=167
output_dim=39
normalized_state=0
./srun.sh -o ../logs/0118_per_time_rew_m --identifier $identifier \
    --model_type $model_type --model_use $model_use --proc_exp_func $proc_exp_func \
    --normalized_state $normalized_state --input_dim $input_dim --output_dim $output_dim \
    --rand 1

# [cur_obs, cur_obs] -> [next_obs - cur_obs]
proc_exp_func='process_exp_func_trans_model_timepoint_obs_obs_normal_regressor'
input_dim=78
output_dim=39
normalized_state=0
./srun.sh -o ../logs/0118_per_time_rew_m --identifier $identifier \
    --model_type $model_type --model_use $model_use --proc_exp_func $proc_exp_func \
    --normalized_state $normalized_state --input_dim $input_dim --output_dim $output_dim \
    --rand 1

# Rerun the dqn to take into account the true label! Also try random search for higher action cost e.g. 1e-3
ac=1e-3
depend_on_labels=1
for idx in 1 2 3 4; do
./srun.sh -o ../logs/0117_dplabels_dqn_ac${ac}_random_search${idx}_rerun python -u run_sequential_dqn.py --my_identifier 0117_dplabels_random_order_search --rand 1 --num_random_run 20 --seed ${idx} --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/1128-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --depend_on_labels ${depend_on_labels} --train_batch_size 512 &
done

# Estimate 0117 to get the new plot for these new agent that takes into account the label
N=4
task(){
   sleep 0.5; python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 0117_ --mode 3 ;
}
(
for f in ../models/dqn_mimic-0117*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)

# Try 0118 where 24 hrs are all the true label to avoid fake label problems...
ac=1e-3
depend_on_labels=0
for idx in 1 2 3 4; do
./srun.sh -o ../logs/0118_24hrs_dqn_ac${ac}_random_search${idx} python -u run_sequential_dqn.py --my_identifier 0118_24hrs_random_order_search --rand 1 --num_random_run 20 --seed ${idx} --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 1 --num_shared_dueling_layers 1 --num_hidden_units 128 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --depend_on_labels ${depend_on_labels} --train_batch_size 512 &
done

# Run random run for the combinations
# 1. Not sure about <1> if normalization help? <2>
for model_type in "StateToProbGainPerTimeEstimator" "ObsToProbGainPerTimeEstimator" "StateToStateDiffPerTimeEstimator" "ObsToObsPerTimeEstimator" "StateToObsPerTimeEstimator"; do
echo ${model_type}
./srun.sh -o ../logs/0118_random_search_h_to_p_${model_type} -u python run_reward_estimator.py --seed 2 --identifier 0118_rand --model_type ${model_type} --normalized_state 1 --rand 1 --num_runs 30 &
done

# Run eval on 0118
N=4
task(){
   sleep 0.5; python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 0118_ --mode 3 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled;
}
(
for f in ../models/dqn_mimic-0118*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)

# Rerun the State without normalization!!!!
for model_type in "StateToProbGainPerTimeEstimator" "StateToStateDiffPerTimeEstimator" "StateToObsPerTimeEstimator"; do
echo ${model_type}
./srun.sh -o ../logs/0118_random_search_h_to_p_${model_type} -u python run_reward_estimator.py --seed 2 --identifier 0118_rand_norm0 --model_type ${model_type} --normalized_state 0 --rand 1 --num_runs 40 &
done

## O.K. normalization helps a lot in predicting State_diff and Prob_Gain

# Run the reward evaluation for all the rew models
rnn_dir="../models/0117-24hours_39feats_38cov_negsampled_rnn-mimic-nh128-nl2-c1e-07-keeprate0.9_0.7_0.5-npred24-miss1-n_mc_1-MIMIC_window-mingjie_39features_38covs-ManyToOneRNN/"
cache_dir="../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled/"
N=4
task(){
   sleep 0.5; python -u run_reward_estimator.py --identifier 0118_debug --eval_mode 1 --eval_dir $1 --rnn_dir ${rnn_dir} --cache_dir ${cache_dir};
}
(
for f in ../models/0118_rand-StateToStateDiffPerTimeEstimator*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)

## What's next? All models seem terrible
# State -> Prob Gain: pea 0.07
# State -> State Diff: pea 0.17 -> Prob Gain: 0.16!
# Obs -> Prob Gain: pea 0.054

## Run everything more!!! To search more throoughly
for model_type in "StateToStateDiffPerTimeEstimator" "StateToStatePerTimeEstimator" "StateToProbGainPerTimeEstimator"; do
echo ${model_type}
./srun.sh -o ../logs/0118_random_search_h_to_p_${model_type} -u python run_reward_estimator.py --seed 20 --identifier 0118_new --model_type ${model_type} --normalized_state 1 --rand 1 --num_runs 40 &
done

## Debug of all the 0118 policy
for f in ../models/dqn_mimic-0118*; do
   echo "$f"
   python ./run_value_estimator_regression_based.py --policy_dir $f --identifier 0118_debug --mode 3 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ../models/0118_rand-StateToStateDiffPerTimeEstimator-0119-hl3-hu128-lr0.001-reg0.001-kp0.7-n1
   break
done

# Evaluate by new estimator!!!
N=4
task(){
   sleep 0.5; python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 0118_new_est_ --mode 3 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ../models/0118_rand-StateToStateDiffPerTimeEstimator-0119-hl3-hu128-lr0.001-reg0.001-kp0.7-n1;
}
(
for f in ../models/dqn_mimic-0118*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)


rnn_dir="../models/0117-24hours_39feats_38cov_negsampled_rnn-mimic-nh128-nl2-c1e-07-keeprate0.9_0.7_0.5-npred24-miss1-n_mc_1-MIMIC_window-mingjie_39features_38covs-ManyToOneRNN/"
cache_dir="../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled/"
N=4
task(){
   sleep 0.5; python -u run_reward_estimator.py --identifier 0118_new2 --eval_mode 1 --eval_dir $1 --rnn_dir ${rnn_dir} --cache_dir ${cache_dir};
}
(
for f in ../models/0118*StateToStateDiffPerTimeEstimator*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)


# Run a series of action cost based on this best model
for ac in 0 1e-4 5e-4 1e-3 5e-3 1e-2 0.1 1; do
./srun.sh -o ../logs/0119_24hrs_dqn_ac${ac} python -u run_sequential_dqn.py --my_identifier 0119_24hrs --rand 0 --num_random_run 20 --seed 10 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 4 --num_shared_dueling_layers 1 --num_hidden_units 32 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --depend_on_labels 0 --train_batch_size 64 &
done

../models/dqn_mimic-0118_24hrs_random_order_search-g1-ac1.0e-03-gamma0.95-fold1.0-only_pos0-sd167-ad40-nn-10000-4-1-32-lr-0.001-reg-0.001-0.7-s-256-5000-i-50-500-3-1

## Run the To Obs Diff...
for model_type in "StateToObsDiffPerTimeEstimator" "ObsToObsDiffPerTimeEstimator"; do
echo ${model_type}
./srun.sh -o ../logs/0118_random_search_h_to_p_${model_type} -u python run_reward_estimator.py --seed 20 --identifier 0118_new --model_type ${model_type} --normalized_state 1 --rand 1 --num_runs 40 &
done

## Training all of them
for seed in 20 30; do
    for model_type in "StateToStateDiffPerTimeEstimator" "StateToStatePerTimeEstimator" "StateToProbGainPerTimeEstimator"; do
    echo ${model_type}
    ./srun.sh -o ../logs/0119_random_search_h_to_p_${model_type}_s${seed} -u python run_reward_estimator.py --seed ${seed} --identifier 0119_new --model_type ${model_type} --normalized_state 1 --rand 1 --num_runs 40 --max_training_iters 100 &
    done
done

seed=10
for model_type in "StateToObsDiffPerTimeEstimator" "StateToObsPerTimeEstimator" "ObsToProbGainPerTimeEstimator"; do
echo ${model_type}
./srun.sh -o ../logs/0119_random_search_h_to_p_${model_type}_s${seed} -u python run_reward_estimator.py --seed ${seed} --identifier 0119_new --model_type ${model_type} --normalized_state 1 --rand 1 --num_runs 20 --max_training_iters 100 &
done

# Fight!
seed=20
for model_type in "StateToProbGainPerTimeEstimator" "StateToStateDiffPerTimeEstimator"; do
echo ${model_type}
./srun.sh -o ../logs/0119_random_search_h_to_p_${model_type}_s${seed}_norm0 -u python run_reward_estimator.py --seed ${seed} --identifier 0119_new_norm0 --model_type ${model_type} --normalized_state 0 --rand 1 --num_runs 20 --max_training_iters 100 &
done

#
seed=10
for model_type in  "ObsToObsDiffPerTimeEstimator" "ObsToObsPerTimeEstimator"; do
echo ${model_type}
./srun.sh -o ../logs/0119_random_search_h_to_p_${model_type}_s${seed} -u python run_reward_estimator.py --seed ${seed} --identifier 0119_new --model_type ${model_type} --normalized_state 1 --rand 1 --num_runs 20 --max_training_iters 100 &
done
#

## policy evaluation
reward_dir="../models/0119_new-StateToStateDiffPerTimeEstimator-0120-hl2-hu128-lr0.001-reg0.001-kp0.7-n1/"
N=4
task(){
   sleep 0.5; python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 0119_new_est_ --mode 3 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir};
}
(
for f in ../models/dqn_mimic-0118*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)

## policy evaluation
reward_dir="../models/0119_new-StateToStateDiffPerTimeEstimator-0120-hl2-hu128-lr0.001-reg0.001-kp0.7-n1/"
N=1
task(){
   sleep 0.5; python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 0119_ac_roc_ --mode 3 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir};
}
(
for f in ../models/dqn_mimic-0119*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)

reward_dir="../models/0119_new-StateToStateDiffPerTimeEstimator-0120-hl2-hu128-lr0.001-reg0.001-kp0.7-n1/"
python ./run_value_estimator_regression_based.py --identifier 0119_per_time_rand --mode 4 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir}

# Run a random run for all the policy for all the action costs
for ac in 0 1e-4 5e-4 1e-3 5e-3 1e-2; do
./srun.sh -o ../logs/0120_24hrs_dqn_ac${ac}_rand python -u run_sequential_dqn.py --my_identifier 0120_24hrs_rand_ac_and_arch_ --rand 1 --num_random_run 40 --seed 10 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 4 --num_shared_dueling_layers 1 --num_hidden_units 32 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --depend_on_labels 0 --train_batch_size 64 &
done

# Mingjie
for ac in 0 1e-4 5e-4 1e-3 5e-3 1e-2; do
./srun.sh -o ../logs/0120_24hrs_dqn_ac${ac}_rand_s400 python -u run_sequential_dqn.py --my_identifier 0120_24hrs_rand_ac_and_arch_ --rand 1 --num_random_run 40 --seed 400 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 4 --num_shared_dueling_layers 1 --num_hidden_units 32 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --depend_on_labels 0 --train_batch_size 64 &
done

reward_dir="../models/0119_new-StateToStateDiffPerTimeEstimator-0120-hl2-hu128-lr0.001-reg0.001-kp0.7-n1/"
N=4
task(){
   sleep 0.5; python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 0119_2_new_est_ --mode 3 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir};
}
(
for f in ../models/dqn_mimic-0118*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)

## policy evaluation
reward_dir="../models/0119_new-StateToStateDiffPerTimeEstimator-0120-hl2-hu128-lr0.001-reg0.001-kp0.7-n1/"
N=1
task(){
   sleep 0.5; python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 0119_2_ac_roc_ --mode 3 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir};
}
(
for f in ../models/dqn_mimic-0119*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)


reward_dir="../models/0119_new-StateToStateDiffPerTimeEstimator-0120-hl2-hu128-lr0.001-reg0.001-kp0.7-n1/"
N=8
task(){
   sleep 0.5; python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 0120_new_est_ --mode 3 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir};
}
(
for f in ../models/dqn_mimic-0120*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)


## What should I do? Fina another bug of using "neg_sampled" instead of "all" for the per_time_env_exp
## Luckily, the test set is still the same :)
## OK. Run estimators on new training set and see if performance improves :)
for seed in 10 20 30; do
    for model_type in "StateToStateDiffPerTimeEstimator" "StateToProbGainPerTimeEstimator"; do
    echo ${model_type}
    ./srun.sh -o ../logs/0121_random_search_h_to_p_${model_type}_s${seed} -u python run_reward_estimator.py --seed ${seed} --identifier 0121_with_larger_training_ --model_type ${model_type} --normalized_state 1 --rand 1 --num_runs 20 --max_training_iters 100 --cache_dir ../RL_exp_cache/0121-30mins-24hrs-20order-rnn-neg_sampled/ &
    done
done

## Test that when I have large action costs, will it be all 0!
## TODO: And policy search is still running. Still wait for more results :)
for ac in 1; do
./srun.sh -o ../logs/0120_24hrs_dqn_ac${ac}_rand python -u run_sequential_dqn.py --my_identifier 0120_24hrs_rand_ac_and_arch_ --rand 1 --num_random_run 10 --seed 10 --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 4 --num_shared_dueling_layers 1 --num_hidden_units 32 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --depend_on_labels 0 --train_batch_size 64 &
done

## New estimation. Since I want to get a per-patient summary for test and val
reward_dir="../models/0119_new-StateToStateDiffPerTimeEstimator-0120-hl2-hu128-lr0.001-reg0.001-kp0.7-n1/"
N=8
task(){
   sleep 0.5; python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 0121_ --mode 3 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir};
}
(
for f in ../models/dqn_mimic-0120*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)

# Find a bug in random policy evaluations
reward_dir="../models/0119_new-StateToStateDiffPerTimeEstimator-0120-hl2-hu128-lr0.001-reg0.001-kp0.7-n1/"
policy_dir="../models/dqn_mimic-0120_24hrs_rand_ac_and_arch_-g1-ac1.0e-02-gamma0.95-fold1.0-only_pos0-sd167-ad40-nn-10000-2-1-64-lr-0.001-reg-0.0001-0.7-s-256-5000-i-50-500-3-1/"
python ./run_value_estimator_regression_based.py --identifier 0121_per_time_rand_debug --mode 4 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir} --policy_dir ${policy_dir}

## waiting
## <2> See if the neg_subsample affect the reward policy
## <3> All the evaluation is running (random / more policy eval)

## Maybe new random policy??? New random policy!!!!!!!!!!!!

# More negative? -0.017 ~ 0
reward_dir="../models/0121_with_larger_training_-StateToProbGainPerTimeEstimator-0121-hl1-hu256-lr0.0001-reg0.0001-kp0.9-n1/"
policy_dir="../models/dqn_mimic-0120_24hrs_rand_ac_and_arch_-g1-ac1.0e-02-gamma0.95-fold1.0-only_pos0-sd167-ad40-nn-10000-2-1-64-lr-0.001-reg-0.0001-0.7-s-256-5000-i-50-500-3-1/"
python ./run_value_estimator_regression_based.py --identifier 0121_per_time_rand_debug --mode 4 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir} --policy_dir ${policy_dir}


# More positive -0.01~0.002
reward_dir="../models/0121_with_larger_training_-StateToStateDiffPerTimeEstimator-0121-hl1-hu64-lr0.001-reg0.0001-kp0.7-n1"
policy_dir="../models/dqn_mimic-0120_24hrs_rand_ac_and_arch_-g1-ac1.0e-02-gamma0.95-fold1.0-only_pos0-sd167-ad40-nn-10000-2-1-64-lr-0.001-reg-0.0001-0.7-s-256-5000-i-50-500-3-1/"
python ./run_value_estimator_regression_based.py --identifier 0121_per_time_rand_debug --mode 4 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir} --policy_dir ${policy_dir}

# Original: run more times to see if I could solve
reward_dir="../models/0119_new-StateToStateDiffPerTimeEstimator-0120-hl2-hu128-lr0.001-reg0.001-kp0.7-n1/"
policy_dir="../models/dqn_mimic-0120_24hrs_rand_ac_and_arch_-g1-ac1.0e-02-gamma0.95-fold1.0-only_pos0-sd167-ad40-nn-10000-2-1-64-lr-0.001-reg-0.0001-0.7-s-256-5000-i-50-500-3-1/"
for time_pass_freq_scale_factor in 0 0.05 0.1 0.2 0.3 0.5 0.8 1; do
    python ./run_value_estimator_regression_based.py --identifier 0121_per_time_rand_debug --mode 4 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir} --policy_dir ${policy_dir} --time_pass_freq_scale_factor ${time_pass_freq_scale_factor} &
done

# Ryb a new rew estimator!
# Original: run more times to see if I could solve
reward_dir="../models/0121_with_larger_training_-StateToStateDiffPerTimeEstimator-0121-hl1-hu64-lr0.001-reg0.0001-kp0.7-n1"
policy_dir="../models/dqn_mimic-0120_24hrs_rand_ac_and_arch_-g1-ac1.0e-02-gamma0.95-fold1.0-only_pos0-sd167-ad40-nn-10000-2-1-64-lr-0.001-reg-0.0001-0.7-s-256-5000-i-50-500-3-1/"
for time_pass_freq_scale_factor in 0 0.05 0.1 0.2 0.3 0.5 0.8 1; do
    python ./run_value_estimator_regression_based.py --identifier 0121_per_time_rand_new_rew_est --mode 4 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir} --policy_dir ${policy_dir} --time_pass_freq_scale_factor ${time_pass_freq_scale_factor} &
done

#
reward_dir="../models/0121_with_larger_training_-StateToProbGainPerTimeEstimator-0121-hl1-hu64-lr0.001-reg0.0001-kp0.7-n1"
policy_dir="../models/dqn_mimic-0120_24hrs_rand_ac_and_arch_-g1-ac1.0e-02-gamma0.95-fold1.0-only_pos0-sd167-ad40-nn-10000-2-1-64-lr-0.001-reg-0.0001-0.7-s-256-5000-i-50-500-3-1/"
for time_pass_freq_scale_factor in 0 0.05 0.1 0.2 0.3 0.5 0.8 1; do
    python ./run_value_estimator_regression_based.py --identifier 0121_per_time_rand_new_rew_est2 --mode 4 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir} --policy_dir ${policy_dir} --time_pass_freq_scale_factor ${time_pass_freq_scale_factor} &
done

## New estimation!!! To fix the stupid random policy trend
reward_dir="../models/0121_with_larger_training_-StateToProbGainPerTimeEstimator-0121-hl1-hu64-lr0.001-reg0.0001-kp0.7-n1/"
N=6
task(){
   sleep 0.5; python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 2_0121_ --mode 3 --cache_dir ../RL_exp_cache/0121-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir};
}
(
for f in ../models/dqn_mimic-0120*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)

## Run a reverse estimation on another machine
reward_dir="../models/0121_with_larger_training_-StateToProbGainPerTimeEstimator-0121-hl1-hu64-lr0.001-reg0.0001-kp0.7-n1/"
N=8
task(){
   sleep 0.5; python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 2_0121_ --mode 3 --cache_dir ../RL_exp_cache/0121-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir};
}
(
files=(../models/dqn_mimic-0120*)
for ((j=${#files[@]}-1; j>=0; j--)); do
   ((i=i%N)); ((i++==0)) && wait
   f="${files[$j]}"
   echo "$f" "$i"
   task "$f" "$i" &
done
)


  echo ${f}
done

files=()
for f in ../models/dqn_mimic-0120*; do
  echo ${f}
done

reward_dir="../models/0121_with_larger_training_-StateToProbGainPerTimeEstimator-0121-hl1-hu64-lr0.001-reg0.0001-kp0.7-n1/"
policy_dir="../models/dqn_mimic-0120_24hrs_rand_ac_and_arch_-g1-ac1.0e+00-gamma0.95-fold1.0-only_pos0-sd167-ad40-nn-10000-4-1-256-lr-0.01-reg-0.01-0.5-s-256-5000-i-50-500-3-1/"
python ./run_value_estimator_regression_based.py --policy_dir ${policy_dir} --identifier 2_0121_ --mode 3 --cache_dir ../RL_exp_cache/0121-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir}

# Run more random policy
reward_dir="../models/0121_with_larger_training_-StateToProbGainPerTimeEstimator-0121-hl1-hu64-lr0.001-reg0.0001-kp0.7-n1"
policy_dir="../models/dqn_mimic-0120_24hrs_rand_ac_and_arch_-g1-ac1.0e-02-gamma0.95-fold1.0-only_pos0-sd167-ad40-nn-10000-2-1-64-lr-0.001-reg-0.0001-0.7-s-256-5000-i-50-500-3-1/"
for time_pass_freq_scale_factor in 0.85 0.9 0.92 0.94 0.95 0.96 0.98 1.; do
    python ./run_value_estimator_regression_based.py --identifier 2_0122_random_policy_ --mode 4 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir} --policy_dir ${policy_dir} --time_pass_freq_scale_factor ${time_pass_freq_scale_factor} &
done

# Run more random policy
reward_dir="../models/0121_with_larger_training_-StateToProbGainPerTimeEstimator-0121-hl1-hu64-lr0.001-reg0.0001-kp0.7-n1"
policy_dir="../models/dqn_mimic-0120_24hrs_rand_ac_and_arch_-g1-ac1.0e-02-gamma0.95-fold1.0-only_pos0-sd167-ad40-nn-10000-2-1-64-lr-0.001-reg-0.0001-0.7-s-256-5000-i-50-500-3-1/"
for time_pass_freq_scale_factor in 0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    python ./run_value_estimator_regression_based.py --identifier 2_0122_random_policy_ --mode 4 --cache_dir ../RL_exp_cache/0117-30mins-24hrs-20order-rnn-neg_sampled --reward_estimator_dir ${reward_dir} --policy_dir ${policy_dir} --time_pass_freq_scale_factor ${time_pass_freq_scale_factor} &
done

## Rebutal
# Run everything in independent reward :)
for seed in 300
do
for ac in 0 1e-4 5e-4 1e-3 5e-3 1e-2; do
./srun.sh -o ../logs/0311_24hrs_dqn_indep_ac${ac}_rand_s${seed} python -u run_sequential_dqn.py --my_identifier 0311_indep_24hrs_rand_ac_and_arch_ --dqn_cls KTailsDuelingDQN --rand 1 --num_random_run 40 --seed ${seed} --replace 0 --debug 0 --rl_state_dim 128 --action_dim 39 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 4 --num_shared_dueling_layers 1 --num_hidden_units 32 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_cls MIMIC_cache_discretized_joint_exp_independent_measurement --cache_dir ../RL_exp_cache/0311-30mins-24hrs-indep-rnn-neg_sampled/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --depend_on_labels 0 --train_batch_size 64 &
done
done

# If finish, run the evaluation and get a excel
reward_dir="../models/0121_with_larger_training_-StateToProbGainPerTimeEstimator-0121-hl1-hu64-lr0.001-reg0.0001-kp0.7-n1/"
N=6
task(){
   sleep 0.5; python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 0311_indep --mode 3  --reward_estimator_dir ${reward_dir};
}
(
for f in ../models/dqn_mimic-0311*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)

## Run the last observation dqn
for seed in 300 200
do
for ac in 0 1e-4 5e-4 1e-3 5e-3 1e-2; do
./srun.sh -o ../logs/0312_24hrs_dqn_obs_input_ac${ac}_rand_s${seed} python -u run_sequential_dqn.py --my_identifier 0312_obs_input_24hrs_rand_ac_and_arch_ --dqn_cls LastObsSequencialDuelingDQN --rand 1 --num_random_run 40 --seed ${seed} --replace 0 --debug 0 --rl_state_dim 155 --action_dim 40 --gamma 0.95 --normalized_state 0 --num_shared_all_layers 4 --num_shared_dueling_layers 1 --num_hidden_units 32 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_cls MIMIC_cache_discretized_joint_exp_random_order_with_obs --cache_dir ../RL_exp_cache/0312-30mins-24hrs-20order-rnn-neg_sampled-with-obs/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --depend_on_labels 0 --train_batch_size 64 &
done
done

# If finish, run the evaluation and get a excel
reward_dir="../models/0121_with_larger_training_-StateToProbGainPerTimeEstimator-0121-hl1-hu64-lr0.001-reg0.0001-kp0.7-n1/"
N=6
task(){
   sleep 0.5; python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 0313_last_obs --mode 3  --reward_estimator_dir ${reward_dir} ;
}
(
for f in ../models/dqn_mimic-0312_obs*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)

## Wondering if the small bug actually affects the performance of dqn. Rerun
for seed in 300; do
for ac in 0 1e-4 5e-4 1e-3 5e-3 1e-2; do
./srun.sh -o ../logs/0314_24hrs_dqn_ac${ac}_rand_s${seed} python -u run_sequential_dqn.py --my_identifier 0314_24hrs_rand_ac_and_arch --dqn_cls SequencialDuelingDQN --rand 1 --num_random_run 30 --seed ${seed} --replace 0 --debug 0 --rl_state_dim 167 --action_dim 40 --gamma 0.95 --normalized_state 1 --num_shared_all_layers 4 --num_shared_dueling_layers 1 --num_hidden_units 32 --lr 0.001 --reg_constant 0.001 --keep_prob 0.7 --cache_cls MIMIC_cache_discretized_joint_exp_random_order_with_obs --cache_dir ../RL_exp_cache/0312-30mins-24hrs-20order-rnn-neg_sampled-with-obs/ --action_cost_coef ${ac} --pos_label_fold_coef 1 --only_pos_reward 0 --depend_on_labels 0 --train_batch_size 64 &
done
done

## If above finish, run the following
reward_dir="../models/0121_with_larger_training_-StateToProbGainPerTimeEstimator-0121-hl1-hu64-lr0.001-reg0.0001-kp0.7-n1/"
N=6
task(){
   sleep 0.5; python ./run_value_estimator_regression_based.py --policy_dir $1 --identifier 0314_seq_dqn_ --mode 3  --reward_estimator_dir ${reward_dir} ;
}
(
for f in ../models/dqn_mimic-0314*; do
   ((i=i%N)); ((i++==0)) && wait
   echo "$f" "$i"
   task "$f" "$i" &
done
)

