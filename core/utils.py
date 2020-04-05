import typing
import collections
import torch
from torch import nn
from atari_wrappers import wrap_deepmind, wrap_pytorch, make_atari
import random
import ray
from core.model import AtariNet

Buffers = typing.Dict[str, typing.List[torch.Tensor]]
Trajectory = collections.namedtuple(
    'Trajectory',
    [
        'frame',
        'reward',
        'done',
        'episode_return',
        'episode_step',
        'policy_logits',
        'baseline',
        'last_action',
        'action',
        'mask',
        'initial_agent_state',
    ]
)
TrainingState = collections.namedtuple(
    'TrainingState',
    [
        'network',
        'optimizer',
        'scheduler',
    ]
)


def append_to_buffer(buffer, env_output, agent_output, index):
    for key in env_output:
        buffer[key][index, ...] = env_output[key]
    if agent_output is not None:
        for key in agent_output:
            buffer[key][index, ...] = agent_output[key]
    buffer['mask'][index, ...] = 1


def create_buffer(flags, obs_shape, num_actions, agent_state):
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
        mask=dict(size=(T + 1,), dtype=torch.bool),
    )
    buffers = {key: torch.zeros(**specs[key]) for key in specs}
    buffers['initial_agent_state'] = agent_state

    buffers['mask'][:, ...] = 1

    return buffers


class ExperienceReplay(object):
    """
        Once a learner samples a trajectory, we need to hold a separate buffer to ensure those trajectories aren't
            thrown away in the middle.
        We need to maintain - for each trajectory - the current index the agent is at.
        If trajectory isn't a multiple of unroll_length - set the mask to 0, this will ensure it isn't learned from.
        If a trajectory (for a single learner) is deemed "too off-policy" then simply re-sample trajectories
            instead of feeding the learner with empty transitions.
    """
    def __init__(
            self,
            flags
    ):
        self.flags = flags

        gym_env = create_env(self.flags)
        self.obs_shape, self.action_dim = gym_env.observation_space.shape, gym_env.action_space.n

        net = AtariNet(gym_env.observation_space.shape[0], gym_env.action_space.n, self.flags.use_lstm, self.flags.use_resnet)
        self.agent_state = net.initial_state(batch_size=1)

        self.max_size = self.flags.replay_size
        self.trajectories = []
        self.sampled_trajectories = [[[] for _ in range(self.flags.replay_batch_size)] for _ in range(self.flags.num_learner_threads)]
        self.current_size = 0
        self.buffers = [None for _ in range(self.flags.replay_batch_size)]
        self.initial_states = [[None for _ in range(self.flags.replay_batch_size)] for _ in range(self.flags.num_learner_threads)]
        self.indices = [[None for _ in range(self.flags.replay_batch_size)] for _ in range(self.flags.num_learner_threads)]

    def sample(self, learner_idx):
        if self.flags.replay_batch_size <= 0 or self.current_size == 0:
            return None

        batch = []

        for traj_index in range(self.flags.replay_batch_size):
            if self.indices[learner_idx][traj_index] is None:
                self.sampled_trajectories[learner_idx][traj_index] = random.sample(self.trajectories, 1)[0]
                self.indices[learner_idx][traj_index] = 0
                self.initial_states[learner_idx][traj_index] = self.agent_state

            batch.append(create_buffer(self.flags, self.obs_shape, self.action_dim, self.initial_states[learner_idx][traj_index]))
            max_idx = min(self.indices[learner_idx][traj_index] + self.flags.unroll_length, len(self.sampled_trajectories[learner_idx][traj_index]))
            buffer_index = 0
            for idx in range(self.indices[learner_idx][traj_index], max_idx):
                append_to_buffer(batch[-1], self.sampled_trajectories[learner_idx][traj_index][idx], None, buffer_index)
                buffer_index += 1

            if max_idx - self.indices[learner_idx][traj_index] < self.flags.unroll_length:
                batch[-1]['mask'][max_idx - self.indices[learner_idx][traj_index]:, ...] = 0

        return batch

    def update_state_and_status(self, learner_idx, agent_states, status_list):
        for traj_index in range(self.flags.replay_batch_size):
            # Either the reminder of the trajectory isn't relevant OR we finished with the trajectory
            if status_list[learner_idx].item() == 0 or \
                    len(self.sampled_trajectories[learner_idx][traj_index]) < self.indices[learner_idx][traj_index] + self.flags.unroll_length:
                self.indices[learner_idx][traj_index] = None
            else:
                self.indices[learner_idx][traj_index] += self.flags.unroll_length

            self.initial_states[learner_idx][traj_index] = tuple(agent_state[:, traj_index, :].unsqueeze(1) for agent_state in agent_states)

    def add_trajectory(self, trajectory):
        while self.current_size > self.max_size:
            self.current_size -= len(self.trajectories[0])
            self.trajectories.pop(0)
        self.trajectories.append(trajectory)
        self.current_size += len(trajectory)


def create_env(flags):
    return wrap_pytorch(
        wrap_deepmind(
            make_atari(flags.env),
            clip_rewards=False,
            frame_stack=True,
            scale=False,
        )
    )
