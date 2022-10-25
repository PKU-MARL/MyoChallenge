1. myosuite/myosuite/agents/train_myosuite.sh myo 
2. python hydra_mjrl_launcher.py --config-name ppo_local.yaml  env=myoChallengeDieReorientP2-v0
3. MJPL python myosuite/utils/examine_env.py --env_name myoChallengeDieReorientP2-v0 --policy_path /home/chengdong/Desktop/MyoChallenge/myosuite/myosuite/agents/outputs/2022-10-22/11-17-40/npg_myoChallengeDieReorientP1-v0/iterations/best_policy.pickle