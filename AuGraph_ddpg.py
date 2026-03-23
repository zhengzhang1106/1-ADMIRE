import numpy as np
import random
import torch
from ray import tune
import ray
from AuGraph_env import AuGraphEnv
from AuGraph_model import AuGraphModel
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.models.catalog import ModelCatalog
import os

## 设置随机种子
seed_num = 0
np.random.seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)

# 运行ray
ray.shutdown()
ray.init()
ModelCatalog.register_custom_model('augraph_model', AuGraphModel)  # 使用自定义模型
tunerun = tune.run(
    DDPGTrainer,
    local_dir="./result",
    config={
        # 其他
        'env': AuGraphEnv,
        'framework': 'torch',
        'seed': seed_num,
        # 'num_gpus': int(os.environ.get("RLLIB_NUM_GPUS", "0")),  # GPU
        'num_gpus': 0,  # GPU，需要<1

        # ========= Model ============
        # 在进入actor和critic的隐藏层之前，会先运行'model'里的参数
        "use_state_preprocessor": True,  # 可以使用自定义model
        "actor_hiddens": [128, 64],
        "actor_hidden_activation": "relu",
        # Postprocess the critic network model output with these hidden layers;
        # again, if use_state_preprocessor is True, then the state will be
        # preprocessed by the model specified with the "model" config option first.
        "critic_hiddens": [128, 64],
        "critic_hidden_activation": "relu",
        "n_step": 1,  # N-step Q learning
        # 自定义模型
        'model': {
            'custom_model': 'augraph_model',
            'conv_filters': [[36, [3, 3], 1], [18, [2, 2], 1], [6, [2, 2], 1]],
            'conv_activation': 'relu',  # tune.grid_search(['relu','tanh']),
            "post_fcnet_hiddens": [256],
            "post_fcnet_activation": 'relu',  # tune.grid_search(['relu','tanh'])
        },

        # === Twin Delayed DDPG (TD3) and Soft Actor-Critic (SAC) tricks ===
        "twin_q": True,  # twin Q-net
        "policy_delay": 1,  # delayed policy update，1-4都可以
        "smooth_target_policy": True,  # target policy smoothing
        'target_noise': 0.2,  # gaussian stddev of target action noise for smoothing 0.3
        "target_noise_clip": 0.5,  # target noise limit (bound),不超过0.5

        # === Evaluation ===
        "evaluation_interval": None,
        "evaluation_num_episodes": 10,  # Number of episodes to run per evaluation period.

        # === Exploration ===
        "explore": True,
        "exploration_config": {
            # TD3 uses simple Gaussian noise on top of deterministic NN-output
            # actions (after a possible pure random phase of n timesteps).
            "type": "GaussianNoise",
            # For how many timesteps should we return completely random actions,
            # before we start adding (scaled) noise?
            "random_timesteps": 10000, # 4000
            # Gaussian stddev of action noise for exploration.
            "stddev": 0.05,   #0.15,0.2不太好 #0.2
            # Scaling settings by which the Gaussian noise is scaled before
            # being added to the actions. NOTE: The scale timesteps start only
            # after(!) any random steps have been finished.
            # By default, do not anneal over time (fixed 1.0).
            "initial_scale": 1.0,
            "final_scale": 1.0,
            "scale_timesteps": 1,
        },
        # Number of env steps to optimize for before returning
        'timesteps_per_iteration': 100,  # 每次迭代step数量
        # Extra configuration that disables exploration.
        "evaluation_config": {
            "explore": False
        },

        # === Replay buffer ===
        'buffer_size': 10000,
        # If True prioritized replay buffer will be used.
        "prioritized_replay": True,
        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.6,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.4,
        # Epsilon to add to the TD errors when updating priorities.
        'prioritized_replay_eps': 1e-4,
        # Whether to LZ4 compress observations
        "compress_observations": False,
        # If set, this will fix the ratio of replayed from a buffer and learned on
        # timesteps to sampled from an environment and stored in the replay buffe4
        # timesteps. Otherwise, the replay will proceed at the native ratio
        # determined by (train_batch_size / rollout_fragment_length).
        "training_intensity": None,

        # ========= Optimization ==============
        # cl是al的0.1-1倍
        # Learning rate for the critic (Q-function) optimizer.
        # 'critic_lr': tune.grid_search([1e-4, 5e-5]),
        'critic_lr': 3e-4,
        # Learning rate for the actor (policy) optimizer.
        # 'actor_lr': tune.grid_search([1e-4, 5e-5]),
        'actor_lr': 1e-4,
        # Update the target network every `target_network_update_freq` steps.
        'target_network_update_freq': 3000, #0
        # Update the target by \tau * policy + (1-\tau) * target_policy
        'tau': 0.001,  # 软更新系数略小于1,和知乎里面刚好的相反的
        # If True, use huber loss instead of squared loss for critic network
        # Conventionally, no need to clip gradients if using a huber loss
        "use_huber": False,  # no Huber loss
        # Threshold of a huber loss
        "huber_threshold": 1.0,
        # Weights for L2 regularization,TD3 no l2 regularisation
        "l2_reg": 1e-6,
        # If not None, clip gradients during optimization at this value
        "grad_clip": None,
        "clip_rewards": False,
        # How many steps of the model to sample before learning starts.
        'learning_starts': 5000,
        # Update the replay buffer with this many samples at once.
        "rollout_fragment_length": 10,
        # Size of a batched sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        'train_batch_size': 128,
        'gamma': 0.98,  # 奖励衰减

        # === Parallelism ===
        # Number of workers for collecting samples with. This only makes sense
        # to increase if your environment is particularly slow to sample, or if
        # you're using the Async or Ape-X optimizers.
        "num_workers": 0,
        # Whether to compute priorities on workers.
        "worker_side_prioritization": False,
        # Prevent iterations from going lower than this time span
        "min_iter_time_s": 1,
        # "num_gpus_per_worker": 0,

    },
    checkpoint_at_end=True,  # 结束时存储检查点
    checkpoint_freq=1,      # 检查点之间的训练迭代次数
    # 隔几个training_iteration存储一次
    # restore=path #载入检查点
    stop={
        'training_iteration': 350  # 训练轮次
    }
)

# 保存一个最大的训练好的agent
best_checkpoint = tunerun.get_best_checkpoint(
    trial=tunerun.get_best_logdir('episode_reward_mean', 'max'),
    metric='episode_reward_mean',
    mode='max'
)
print(best_checkpoint)

ray.shutdown()