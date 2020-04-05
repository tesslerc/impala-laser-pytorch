# Impala / LASER
A PyTorch implementation of [IMPALA: Scalable Distributed
Deep-RL with Importance Weighted Actor-Learner Architectures
by Espeholt, Soyer, Munos et al.](https://arxiv.org/abs/1802.01561) and [LASER: Off-Policy Actor-Critic with Shared Experience Replay by Schmitt, Hessel and Simonyan](https://arxiv.org/abs/1909.11583).

This is an extension of [TorchBeast](https://github.com/facebookresearch/torchbeast). Specifically, I took only the MonoBeast implementation and replaced the multiprocessing parallelism to use [Ray](https://github.com/ray-project/ray/). While this has been tested on a single machine, it should work out of the box on a cluster (once Ray has been properly set up).

## Performance
On my machine (6 core CPU and GTX 1080TI) I saw 3000 steps per second on the basic model (without the LSTM and ResNet), 2000 SPS with the LSTM and 900 SPS with LSTM and ResNet.

## Requirements

gym[atari]>=0.14.0  # Installs gym and atari.
gitpython>=2.1  # For logging metadata.
opencv-python  # for atari
ray # for parallelism

```shell
$ python main.py --env PongNoFrameskip-v4
```

By default, this uses only a few actors (each with their instance
of the environment). Let's change the default settings (try this on a
beefy machine!):

```shell
$ python3.6 monobeast.py \
    --env PongNoFrameskip-v4 \
    --num_learner_threads 4 \
    --num_actors 45 \
    --total_steps 200000000 \
    --learning_rate 0.0004 \
    --epsilon 0.01 \
    --entropy_cost 0.01 \
    --batch_size 4 \
    --unroll_length 79 \
    --episodes_for_avg 100 \
    --suppress_logs \
    --discounting 0.99 \
    --use_lstm \
    --use_resnet
```

For LASER, set the flags ```--replay_size``` and ```--replay_ratio``` which define both the size of the replay memory and the relative number of replay samples per each on-policy sample taken.

Similar to MonoBeast, the implementation is simple. Each actor runs in a separate process with its dedicated instance of the environment and runs the PyTorch model on the CPU to create actions. The resulting rollout trajectories (environment-agent interactions) are sent to the learner. In the main process, the learner consumes these rollouts and uses them to update the model's weights. The learner is parallelised using multithreading, such as building the batch while the gradient step is performed.

## (Very rough) overview of the system (taken from TorchBeast readme.md)

```
|-----------------|     |-----------------|                  |-----------------|
|     ACTOR 1     |     |     ACTOR 2     |                  |     ACTOR n     |
|-------|         |     |-------|         |                  |-------|         |
|       |  .......|     |       |  .......|     .   .   .    |       |  .......|
|  Env  |<-.Model.|     |  Env  |<-.Model.|                  |  Env  |<-.Model.|
|       |->.......|     |       |->.......|                  |       |->.......|
|-----------------|     |-----------------|                  |-----------------|
   ^     I                 ^     I                              ^     I
   |     I                 |     I                              |     I Actors
   |     I rollout         |     I rollout               weights|     I send
   |     I                 |     I                     /--------/     I rollouts
   |     I          weights|     I                     |              I (frames,
   |     I                 |     I                     |              I  actions
   |     I                 |     v                     |              I  etc)
   |     L=======>|--------------------------------------|<===========J
   |              |.........      LEARNER                |
   \--------------|..Model.. Consumes rollouts, updates  |
     Learner      |.........       model weights         |
      sends       |--------------------------------------|
     weights
```

The system has two main components, actors and a learner.

Actors generate rollouts (tensors from a number of steps of
environment-agent interactions, including environment frames, agent
actions and policy logits, and other data).

The learner consumes that experience, computes a loss and updates the
weights. The new weights are then propagated to the actors.
