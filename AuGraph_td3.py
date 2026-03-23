import sys
import signal
import numpy as np
import random
import torch
from ray import tune
import ray
from AuGraph_env import AuGraphEnv
from AuGraph_model import AuGraphModel
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.models.catalog import ModelCatalog

seed_num = 0
np.random.seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)

def cleanup():
    try:
        if ray.is_initialized():
            print("Shutting down Ray...")
            ray.shutdown()
    except Exception as e:
        print(f"Error during ray.shutdown(): {e}")

def handle_exit(signum, frame):
    print(f"Received signal {signum}, cleaning up before exit...")
    cleanup()
    sys.exit(0)

# 捕获 Ctrl+C 和 kill
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

tunerun = None

try:
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    # 注册模型
    ModelCatalog.register_custom_model('augraph_model', AuGraphModel)

    tunerun = tune.run(
        DDPGTrainer,
        local_dir="./result",
        config={
            'env': AuGraphEnv,
            'framework': 'torch',
            'seed': seed_num,
            'num_gpus': 0,

            "use_state_preprocessor": True,
            "actor_hiddens": [128, 64],
            "actor_hidden_activation": "relu",
            "critic_hiddens": [128, 64],
            "critic_hidden_activation": "relu",
            "n_step": 1,

            'model': {
                'custom_model': 'augraph_model',
                'conv_filters': [[36, [3, 3], 1], [18, [2, 2], 1], [6, [2, 2], 1]],
                'conv_activation': 'relu',
                "post_fcnet_hiddens": [256],
                "post_fcnet_activation": 'relu',
            },

            "twin_q": True,
            "policy_delay": 1,
            "smooth_target_policy": True,
            "target_noise": 0.2,
            "target_noise_clip": 0.5,

            "evaluation_interval": None,
            "evaluation_duration": 10,

            "explore": True,
            "exploration_config": {
                "type": "GaussianNoise",
                "random_timesteps": 10000,
                # Gaussian stddev of action noise for exploration.
                "stddev": 0.05,
                "initial_scale": 1.0,
                "final_scale": 1.0,
                "scale_timesteps": 1,
            },

            'timesteps_per_iteration': 100,
            "evaluation_config": {
                "explore": False
            },
            "buffer_size": 10000,
            # "replay_buffer_config": {
            #     "type": "MultiAgentReplayBuffer",
            #     "capacity": 20000,
            # },
            "store_buffer_in_checkpoints": False,
            "prioritized_replay": True,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
            "prioritized_replay_eps": 1e-4,
            "compress_observations": False,
            "training_intensity": None,

            "critic_lr": 1e-4,
            "actor_lr": 1e-4,
            "target_network_update_freq": 3000,
            "tau": 0.001,
            "use_huber": False,
            "huber_threshold": 1.0,
            "l2_reg": 1e-6,
            "grad_clip": None,
            "clip_rewards": False,
            "learning_starts": 5000,
            "rollout_fragment_length": 10,
            "train_batch_size": 128,
            'gamma': 0.98,

            "num_workers": 0,
            "worker_side_prioritization": False,
            "min_time_s_per_reporting": 1,
        },
        checkpoint_at_end=True,
        checkpoint_freq=1,
        stop={
            'training_iteration': 500
        }
    )

    best_checkpoint = tunerun.get_best_checkpoint(
        trial=tunerun.get_best_logdir('episode_reward_mean', 'max'),
        metric='episode_reward_mean',
        mode='max'
    )
    print("Best checkpoint:", best_checkpoint)

except Exception as e:
    print(f"Training failed: {e}")
    raise

finally:
    cleanup()