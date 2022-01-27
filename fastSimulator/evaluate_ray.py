import argparse
from gym.spaces import Dict, Tuple, Box, Discrete
import os
import yaml

import ray
import ray.tune as tune
from ray.tune.registry import register_env
from gym_fastag.envs.fastag_env import FastAg, ContPruneBaseActNormObsWrapper, DisPruneBaseActNormObsWrapper, \
    BinPruneBaseActNormObsWrapper, BinPruneIrrActionNormObsWrapper, DisPruneIrrActionNormObsWrapper, \
    ContPruneIrrActionNormObsWrapper
from ray.rllib.utils.test_utils import check_learning_achieved

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    type=str,
    default="PPO",
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf2",
    help="The DL framework specifier.")
parser.add_argument("--num-cpus", type=int, default=8)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
         "be achieved within --stop-timesteps AND --stop-iters.")
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=100,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=300000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=50.0,
    help="Reward at which we stop training.")
parser.add_argument(
    "--env-config-path",
    type=str,
    default="/gym_fastag/envs/config/plants_5types_57x57garden_real_fixed_test.yaml",
    help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--env-test-config-path",
    type=str,
    default="/gym_fastag/envs/config/plants_5types_57x57garden_real_fixed_test.yaml",
    help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--num-gpus", type=int, default=0,
    help="Number gpus for ray cluster"
)
parser.add_argument('--env-names', nargs='+', default=['FastAgConPruneBaseAct-v0'], type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(num_cpus=args.num_cpus or None, num_gpus=args.num_gpus, local_mode=args.local_mode)

    if 'FastAgConPruneBaseAct-v0' in args.env_names:
        register_env("FastAgConPruneBaseAct-v0", lambda c: ContPruneBaseActNormObsWrapper(FastAg(c)))
    if 'FastAgDisPruneBaseAct-v0' in args.env_names:
        register_env("FastAgDisPruneBaseAct-v0", lambda c: DisPruneBaseActNormObsWrapper(FastAg(c)))
    if 'FastAgBinPruneBaseAct-v0' in args.env_names:
        register_env("FastAgBinPruneBaseAct-v0", lambda c: BinPruneBaseActNormObsWrapper(FastAg(c)))

    if 'FastAgConPruneIrrAct-v0' in args.env_names:
        register_env("FastAgConPruneIrrAct-v0", lambda c: ContPruneIrrActionNormObsWrapper(FastAg(c)))
    if 'FastAgDisPruneIrrAct-v0' in args.env_names:
        register_env("FastAgDisPruneIrrAct-v0", lambda c: DisPruneIrrActionNormObsWrapper(FastAg(c)))
    if 'FastAgBinPruneIrrAct-v0' in args.env_names:
        register_env("FastAgBinPruneIrrAct-v0", lambda c: BinPruneIrrActionNormObsWrapper(FastAg(c)))

    with open(args.env_config_path) as setup_file:
        load_env_config = yaml.safe_load(setup_file)

    with open(args.env_test_config_path) as setup_file:
        load_test_config = yaml.safe_load(setup_file)

    config = {
        "env": tune.grid_search(args.env_names),
        "env_config": load_env_config,
        #"eager_tracing": True,
        "entropy_coeff": 0.001,  # We don't want high entropy in this Env.
        "gamma": 0.99,  # 0.0 for no discounting of future returns (bandit problem).
        "lambda": 0.95,
        "lr": 0.001,
        "model": {"use_lstm": tune.grid_search([True, False]),
                  "lstm_cell_size": 64,
                  "max_seq_len": 20,
                  "fcnet_hiddens": tune.grid_search([[128, 128]]),
                  "vf_share_layers": True,
                  },
        "train_batch_size": 1200,
        "sgd_minibatch_size": 300,
        "num_envs_per_worker": 1,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 0,
        "num_sgd_iter": 4,
        "num_workers": 3,
        "vf_loss_coeff": 1.0,
        "kl_target": 0.01,
        "kl_coeff": 0,
        "framework": args.framework,
        "rollout_fragment_length": 100,
        "batch_mode": "complete_episodes",
        "evaluation_interval": 10,
        # Run 1 episode each time evaluation runs.
        "evaluation_num_episodes": 10,
        # Override the env config for evaluation.
        "evaluation_config": {
            "explore": False,
            "env_config": load_test_config
        }
    }

    stop = {
        #"training_iteration": args.stop_iters,
        "episode_reward_mean": 45,
        "time_total_s": 3600,
        "timesteps_total": args.stop_timesteps,
    }

    results = tune.run(args.run, local_dir="~/ray_results/", config=config, metric="episode_reward_mean", mode="max",
                       stop=stop,
                       checkpoint_at_end=True, verbose=1)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
