import ray
import threading
import torch
from torch import nn
from torch.nn import functional as F
import logging
import timeit
import pprint
import os
from core import prof
from core.utils import Trajectory, ExperienceReplay
from core import vtrace
from core.model import AtariNet
from core import file_writer


logger = logging.getLogger(__name__)


@ray.remote(num_gpus=torch.cuda.device_count())
class Learner(object):
    def __init__(
            self,
            flags,
            actors,
            in_channels,
            action_dim,
            stat_keys
    ):
        self.flags = flags
        self.actors = actors

        self.net = AtariNet(in_channels, action_dim, self.flags.use_lstm, self.flags.use_resnet).to(device=self.flags.device)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(),
                                             lr=flags.learning_rate,
                                             momentum=flags.momentum,
                                             eps=flags.epsilon,
                                             alpha=flags.alpha,
                                             )

        self.replay_memory = None
        if self.flags.replay_size > 0:
            self.replay_memory = ExperienceReplay(self.flags)

        def lr_lambda(epoch):
            return 1 - min(epoch * self.flags.unroll_length * self.flags.batch_size, self.flags.total_steps) / self.flags.total_steps

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (self.flags.savedir, self.flags.xpid, "model.tar"))
        )

        self.timer = timeit.default_timer
        self.last_time = self.timer()
        self.stat_keys = stat_keys

        self.logger = logging.getLogger(__name__)

        self.replay_batch = None

    def train(self):
        params_idx = 0
        params_id = ray.put({k: v.cpu() for k, v in self.net.state_dict().items()})

        rollouts = [actor.act.remote(params_id, params_idx) for actor in self.actors]
        trajectories = [[] for _ in range(self.flags.num_actors)]
        stats = {}
        step = 0
        start_step = 0
        last_log_time = self.timer()
        last_time = self.timer()
        batch_lock = threading.Lock()
        learn_lock = threading.Lock()
        log_lock = threading.Lock()

        def batch_and_learn(learner_idx):
            """Thread target for the learning process."""
            nonlocal step, stats, params_idx, params_id, rollouts, trajectories, last_log_time, last_time, start_step
            timings = prof.Timings()
            batch = []

            while step < self.flags.total_steps:
                # Obtain batch of data. Learners don't do this in parallel to ensure maximal throughput.
                with batch_lock:
                    while len(batch) < self.flags.batch_size:
                        done_id, rollouts = ray.wait(rollouts)

                        # get the results of the task from the object store
                        rollout, actor_id = ray.get(done_id)[0]
                        batch.append(rollout)
                        # start a new task on the same actor object
                        rollouts.extend([self.actors[actor_id].act.remote(params_id, params_idx)])

                        if self.replay_memory is not None:
                            # add trajectory to replay memory
                            for idx in range(self.flags.unroll_length + 1):
                                transition = {}
                                for key in rollout:
                                    if key not in ['initial_agent_state', 'mask']:
                                        transition[key] = rollout[key][idx]
                                trajectories[actor_id].append(transition)
                                if transition["done"]:
                                    self.replay_memory.add_trajectory(trajectories[actor_id])
                                    trajectories[actor_id] = []

                    timings.time("dequeue")

                batch, agent_state, used_replay = get_batch(learner_idx, batch, self.replay_memory, timings, self.flags, step)

                # Perform a learning step and update the network parameters.
                with learn_lock:
                    tmp_stats, agent_state, mask = learn(
                        self.flags, self.net, batch, agent_state, self.optimizer, self.scheduler
                    )
                    params_idx += 1
                    params_id = ray.put({k: v.cpu() for k, v in self.net.state_dict().items()})
                    timings.time("learn")
                batch = []

                # For LASER (https://arxiv.org/abs/1909.11583) update the replay status. As the memory treats each
                # learner individually, they can access the data structures in parallel.
                if used_replay:
                    self.replay_memory.update_state_and_status(
                        learner_idx,
                        tuple(t[:, self.flags.batch_size:].cpu() for t in agent_state),
                        mask[-1, self.flags.batch_size:].cpu())

                timings.time("update replay")

                # Logging results in updating the step counter AND printing to console.
                # This requires locking for concurrency.
                with log_lock:
                    step += self.flags.unroll_length * self.flags.batch_size

                    if 'episode_returns' in stats.keys():
                        tmp_stats['episode_returns'] += stats['episode_returns']
                    stats = tmp_stats

                    if self.timer() - last_log_time > 5:
                        last_log_time = self.timer()
                        print_tuple = get_stats(self.flags, stats, self.timer, last_time, step, start_step)
                        start_step = step
                        last_time = self.timer()
                        log(self.flags, print_tuple, self.logger, step)
                        timings.time("log")

            if learner_idx == 0:
                self.logger.error("Batch and learn: %s", timings.summary())

        threads = []
        for i in range(self.flags.num_learner_threads):
            thread = threading.Thread(
                target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        return 0

    def checkpoint(self):
        if self.flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", self.checkpointpath)
        save_dict = {
            "flags": vars(self.flags),
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

        torch.save(save_dict, self.checkpointpath)

    def print_timings(self):
        self.logger.error("Batch and learn: %s", self.timings.summary())
        return 0


def log(flags, print_tuple, logger, step):
    if print_tuple is not None:
        sps, total_loss, mean_return, pformat = print_tuple
        if flags.suppress_logs:
            logger.error(
                "Steps %i @ %.1f SPS. Loss %f. %s",
                step,
                sps,
                total_loss,
                mean_return,
            )
        else:
            logger.error(
                "Stats:\n%s\nSteps %i @ %.1f SPS. Loss %f. %s",
                pformat,
                step,
                sps,
                total_loss,
                mean_return,
            )


def get_batch(learner_idx, online_batch, replay_memory, timings, flags, step):
    replay_batch = None
    if replay_memory is not None and step >= flags.learn_start:
        replay_batch = replay_memory.sample(learner_idx)

    timings.time("replay")

    batch = {}
    for key in Trajectory._fields:
        if key != 'initial_agent_state':
            batch_list = [online_batch[i][key] for i in range(flags.batch_size)]
            if replay_batch is not None:
                batch_list += [replay_batch[idx][key] for idx in range(flags.replay_batch_size)]

            batch[key] = torch.stack(batch_list, dim=1)

    agent_state_list = [online_batch[i]['initial_agent_state'] for i in range(flags.batch_size)]
    if replay_batch is not None:
        agent_state_list += [replay_batch[idx]['initial_agent_state'] for idx in range(flags.replay_batch_size)]

    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*agent_state_list)
    )
    timings.time("batch")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
    )
    timings.time("device")

    return batch, agent_state, replay_batch is not None


def learn(
        flags,
        model,
        batch,
        initial_agent_state,
        optimizer,
        scheduler,
):
    """Performs a learning (optimization) step."""
    learner_outputs, next_agent_state = model(batch, initial_agent_state)

    # Take final value function slice for bootstrapping.
    bootstrap_value = learner_outputs["baseline"][-1]

    # Move from obs[t] -> action[t] to action[t] -> obs[t].
    batch = {key: tensor[1:] for key, tensor in batch.items()}
    learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

    rewards = batch["reward"]
    if flags.reward_clipping == "abs_one":
        clipped_rewards = torch.clamp(rewards, -1, 1)
    elif flags.reward_clipping == "none":
        clipped_rewards = rewards

    discounts = (~batch["done"]).float() * flags.discounting

    vtrace_returns = vtrace.from_logits(
        behavior_policy_logits=batch["policy_logits"],
        target_policy_logits=learner_outputs["policy_logits"],
        actions=batch["action"],
        discounts=discounts,
        rewards=clipped_rewards,
        values=learner_outputs["baseline"],
        bootstrap_value=bootstrap_value,
    )

    pg_loss = compute_policy_gradient_loss(
        learner_outputs["policy_logits"],
        batch["action"],
        vtrace_returns.pg_advantages * batch["mask"],
    )
    baseline_loss = flags.baseline_cost * compute_baseline_loss(
        (vtrace_returns.vs - learner_outputs["baseline"]) * vtrace_returns.mask * batch["mask"]
    )
    # Compute entropy only on the "on-policy" data!
    entropy_loss = flags.entropy_cost * compute_entropy_loss(
        learner_outputs["policy_logits"][:, :flags.batch_size]
    )

    total_loss = pg_loss + baseline_loss + entropy_loss

    # Only keep the on-policy data!
    on_policy_data = {key: tensor[:, :flags.batch_size] for key, tensor in batch.items()}

    episode_returns = on_policy_data["episode_return"][on_policy_data["done"]]
    stats = {
        "episode_returns": tuple(episode_returns.cpu().numpy()),
        "mean_episode_return": torch.mean(episode_returns).item(),
        "total_loss": total_loss.item(),
        "pg_loss": pg_loss.item(),
        "baseline_loss": baseline_loss.item(),
        "entropy_loss": entropy_loss.item(),
    }

    optimizer.zero_grad()

    total_loss.backward()
    if flags.grad_norm_clipping > 0:
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
    optimizer.step()
    scheduler.step()

    return stats, next_agent_state, vtrace_returns.mask


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def get_stats(flags, stats, timer, last_time, step, start_step):
    if flags.episodes_for_avg is not None:
        # periodic logging, otherwise it's too spammy.
        if len(stats.get("episode_returns", ())) < flags.episodes_for_avg:
            return None

    if stats.get("episode_returns", None):
        ret = sum(stats["episode_returns"]) / len(stats["episode_returns"])
        mean_return = ("Return per episode: %.1f. " % ret)
        stats['mean_episode_return'] = ret
    else:
        mean_return = ""

    sps = (step - start_step) / (timer() - last_time)

    pformat = pprint.pformat(stats)
    stats['episode_returns'] = ()

    total_loss = stats.get("total_loss", float("inf"))
    return sps, total_loss, mean_return, pformat
