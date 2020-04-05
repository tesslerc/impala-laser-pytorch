import logging
import os
import traceback

import torch

from core import prof, environment
from core.utils import create_env, create_buffer, append_to_buffer
from core.model import AtariNet
import ray

logger = logging.getLogger(__name__)


@ray.remote
class Actor(object):
    def __init__(self, flags, actor_index):
        self.flags = flags
        self.actor_index = actor_index
        self.logger = logging.getLogger(__name__)
        self.logger.error("Actor %i started.", self.actor_index)

        self.timings = prof.Timings()  # Keep track of how fast things are.

        self.gym_env = create_env(flags)
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        self.gym_env.seed(seed)
        self.env = environment.Environment(self.gym_env)
        self.env_output = self.env.initial()

        self.net = AtariNet(self.gym_env.observation_space.shape[0], self.gym_env.action_space.n, self.flags.use_lstm, self.flags.use_resnet)
        self.agent_state = self.net.initial_state(batch_size=1)
        self.agent_output, _ = self.net(self.env_output, self.agent_state)
        self.params_idx = -1

    def act(
            self,
            params,
            params_idx
    ):
        try:
            # Prevent re-loading identical params
            if self.params_idx < params_idx:
                self.params_idx = params_idx
                self.net.load_state_dict(params)
            trajectory = []

            self.timings.reset()

            buffer = create_buffer(self.flags, self.gym_env.observation_space.shape, self.gym_env.action_space.n, self.agent_state)
            # Write old rollout end.
            append_to_buffer(buffer, self.env_output, self.agent_output, 0)
            self.timings.time("create")

            # Do new rollout.
            for t in range(self.flags.unroll_length):
                with torch.no_grad():
                    self.agent_output, self.agent_state = self.net(self.env_output, self.agent_state)

                self.timings.time("model")

                self.env_output = self.env.step(self.agent_output["action"])

                self.timings.time("step")

                append_to_buffer(buffer, self.env_output, self.agent_output, t + 1)

                trajectory.append({**self.env_output, **self.agent_output})
                self.timings.time("write")

            return buffer, self.actor_index

        except KeyboardInterrupt:
            pass  # Return silently.
        except Exception as e:
            logger.error("Exception in worker process %i", self.actor_index)
            traceback.print_exc()
            print()
            raise e

    def print_timings(self):
        self.logger.error("Actor: %s", self.timings.summary())
        return 0
