python train.py --domain_name hammer-v0 \
  --reward_type sparse --num_updates 1 \
  --observation_type state --encoder_type identity --work_dir ~/experiments_ferm/hammer-v0 \
  --agent curl_sac \
  --seed 0105 --critic_lr 0.001 --actor_lr 0.001 --eval_freq 10000 --batch_size 128 \
  --num_train_steps 2000000 --save_tb --demo_model_dir expert/FetchPickAndPlace-v1 \
  --demo_samples 10000 \
  --replay_buffer_capacity 1000000 \
  --critic_target_update_freq 1 \
  --critic_tau 0.01 --init_steps 0 --warmup_offline_sac 200
