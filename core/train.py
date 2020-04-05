import logging
import time

import ray
import torch

from core.actor import Actor
from core.utils import create_env
from core.learner import Learner


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    ray.init()

    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")

    flags.replay_batch_size = int(flags.batch_size * flags.replay_ratio)

    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
    ]
    logger = logging.getLogger("logfile")

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logger.error("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logger.error("Not using CUDA.")
        flags.device = torch.device("cpu")

    env = create_env(flags)

    actors = []
    for i in range(flags.num_actors):
        actors.append(Actor.remote(flags, i,))

    learner = Learner.remote(flags, actors, env.observation_space.shape[0], env.action_space.n, stat_keys)
    learner_handle = learner.train.remote()

    ray.wait([learner_handle])
    ray.wait([actors[0].print_timings.remote()])
