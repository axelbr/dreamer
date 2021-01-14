import argparse
import collections
import functools
import json
import os
import pathlib
import sys
import time
import math
import string

import matplotlib.pyplot as plt
import tensorflow as tf
from agents.gap_follower import GapFollower

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
import wrappers
from datetime import datetime

#tf.config.run_functions_eagerly(run_eagerly=True)

def define_config():
  config = tools.AttrDict()
  # General.
  config.logdir = pathlib.Path("./logs/racecar_{}/".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
  config.seed = 0
  config.multi_test = False       # if `true`, run 5 experiments by varying the seeds
  config.steps = 5e6
  config.eval_every = 1e4
  config.log_every = 1e3
  config.log_scalars = True
  config.log_images = True
  config.log_videos = True
  config.gpu_growth = True
  config.precision = 32
  config.obs_type = 'lidar'
  # Environment.
  config.task = 'racecar_austria'
  config.parallel = 'none'
  config.action_repeat = 4
  config.time_limit = 4000
  config.prefill_agent = 'gap_follower'
  config.prefill = 10000
  config.eval_noise = 0.0
  config.clip_rewards = 'none'
  config.clip_rewards_min = -1
  config.clip_rewards_max = 1
  # Model.
  config.encoded_obs_dim = 1080
  config.deter_size = 200
  config.stoch_size = 30
  config.num_units = 400
  config.reward_out_dist = 'normal'
  config.dense_act = 'elu'
  config.cnn_act = 'relu'
  config.cnn_depth = 128
  config.pcont = True
  config.free_nats = 3.0
  config.kl_scale = 1.0
  config.pcont_scale = 10.0
  config.weight_decay = 0.0
  config.weight_decay_pattern = r'.*'
  # Training.
  config.batch_size = 50
  config.batch_length = 50
  config.train_every = 1000
  config.train_steps = 100
  config.pretrain = 100
  config.model_lr = 6e-4
  config.value_lr = 8e-5
  config.actor_lr = 8e-5
  config.grad_clip = 1.0
  config.dataset_balance = False
  # Behavior.
  config.discount = 0.99
  config.disclam = 0.95
  config.horizon = 20
  config.action_dist = 'tanh_normal'
  config.action_init_std = 5.0
  config.expl = 'additive_gaussian'
  config.expl_amount = 0.3
  config.expl_decay = 0.0
  config.expl_min = 0.3
  return config


best_return_so_far = - np.Inf
class Dreamer(tools.Module):

  def __init__(self, config, datadir, actspace, obspace, writer):
    self._c = config
    self._actspace = actspace['A']
    self._obspace = obspace['A']
    self._actdim = actspace.n if hasattr(actspace, 'n') else self._actspace.shape[0]
    self._writer = writer
    self._random = np.random.RandomState(config.seed)
    with tf.device('cpu:0'):
      self._step = tf.Variable(count_steps(datadir, config), dtype=tf.int64)
    self._should_pretrain = tools.Once()
    self._should_train = tools.Every(config.train_every)
    self._should_log = tools.Every(config.log_every)
    self._last_log = None
    self._last_time = time.time()
    self._metrics = collections.defaultdict(tf.metrics.Mean)
    self._metrics['expl_amount']  # Create variable for checkpoint.
    self._float = prec.global_policy().compute_dtype
    self._strategy = tf.distribute.MirroredStrategy()
    with self._strategy.scope():
      self._dataset = iter(self._strategy.experimental_distribute_dataset(
          load_dataset(datadir, self._c)))
      self._build_model()

  def __call__(self, obs, reset, state=None, training=True):
    step = self._step.numpy().item()
    tf.summary.experimental.set_step(step)
    if state is not None and reset.any():
      mask = tf.cast(1 - reset, self._float)[:, None]
      mask = tf.cast(1 - reset, self._float)[:, None]
      state = tf.nest.map_structure(lambda x: x * mask, state)
    if self._should_train(step) and training:
      log = self._should_log(step)
      n = self._c.pretrain if self._should_pretrain() else self._c.train_steps
      print(f'Training for {n} steps.')
      with self._strategy.scope():
        for train_step in range(n):
          print(f'[Train Step] # {train_step}')
          log_images = self._c.log_images and log and train_step == 0
          self.train(next(self._dataset), log_images)
      if log:
        self._write_summaries()
    action, state = self.policy(obs, state, training)
    if training:
      self._step.assign_add(len(reset) * self._c.action_repeat)
    return action, state

  @tf.function
  def policy(self, obs, state, training):
    if state is None:
      latent = self._dynamics.initial(len(obs[self._c.obs_type]))
      action = tf.zeros((len(obs[self._c.obs_type]), self._actdim), self._float)
    else:
      latent, action = state
    embed = self._encode(preprocess(obs, self._c))
    latent, _ = self._dynamics.obs_step(latent, action, embed)
    feat = self._dynamics.get_feat(latent)
    if training:
      action = self._actor(feat).sample()
    else:
      action = self._actor(feat).mode()
    action = self._exploration(action, training)
    state = (latent, action)
    return action, state

  def load(self, filename):
    super().load(filename)
    self._should_pretrain()

  @tf.function()
  def train(self, data, log_images=False):
    self._strategy.experimental_run_v2(self._train, args=(data, log_images))

  def _train(self, data, log_images):
    with tf.GradientTape() as model_tape:
      embed = self._encode(data)
      post, prior = self._dynamics.observe(embed, data['action'])
      feat = self._dynamics.get_feat(post)
      image_pred = self._decode(feat)
      reward_pred = self._reward(feat)
      likes = tools.AttrDict()
      likes.image = tf.reduce_mean(image_pred.log_prob(data[self._c.obs_type]))
      likes.reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))
      if self._c.pcont:
        pcont_pred = self._pcont(feat)
        pcont_target = self._c.discount * data['discount']
        likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
        likes.pcont *= self._c.pcont_scale
      prior_dist = self._dynamics.get_dist(prior)
      post_dist = self._dynamics.get_dist(post)
      div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
      div = tf.maximum(div, self._c.free_nats)
      model_loss = self._c.kl_scale * div - sum(likes.values())
      model_loss /= float(self._strategy.num_replicas_in_sync)

    with tf.GradientTape() as actor_tape:
      imag_feat = self._imagine_ahead(post)
      reward = tf.cast(self._reward(imag_feat).mode(), 'float')       # cast: to address the output of bernoulli
      if self._c.pcont:
        pcont = self._pcont(imag_feat).mean()
      else:
        pcont = self._c.discount * tf.ones_like(reward)
      value = self._value(imag_feat).mode()
      returns = tools.lambda_return(
          reward[:-1], value[:-1], pcont[:-1],
          bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
      discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
          [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
      actor_loss = -tf.reduce_mean(discount * returns)
      actor_loss /= float(self._strategy.num_replicas_in_sync)

    with tf.GradientTape() as value_tape:
      value_pred = self._value(imag_feat)[:-1]
      target = tf.stop_gradient(returns)
      value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))
      value_loss /= float(self._strategy.num_replicas_in_sync)

    model_norm = self._model_opt(model_tape, model_loss)
    actor_norm = self._actor_opt(actor_tape, actor_loss)
    value_norm = self._value_opt(value_tape, value_loss)

    if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
      if self._c.log_scalars:
        self._scalar_summaries(
            data, feat, prior_dist, post_dist, likes, div,
            model_loss, value_loss, actor_loss, model_norm, value_norm,
            actor_norm)
      if tf.equal(log_images, True):
        self._image_summaries(data, embed, image_pred)
        self._reward_summaries(data, reward_pred)

  def _build_model(self):
    acts = dict(
        elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
        leaky_relu=tf.nn.leaky_relu)
    cnn_act = acts[self._c.cnn_act]
    act = acts[self._c.dense_act]

    if self._c.obs_type == 'image':
      self._encode = models.ConvEncoder(self._c.cnn_depth, cnn_act)
      self._decode = models.ConvDecoder(self._c.cnn_depth, cnn_act)
    elif self._c.obs_type == 'lidar':
      self._encode = models.IdentityEncoder()
      self._decode = models.MLPLidarDecoder(self._c.cnn_depth, self._obspace['lidar'].shape)
    self._dynamics = models.RSSM(self._c.stoch_size, self._c.deter_size, self._c.deter_size)

    self._reward = models.DenseDecoder((), 2, self._c.num_units, dist=self._c.reward_out_dist, act=act)
    if self._c.pcont:
      self._pcont = models.DenseDecoder(
          (), 3, self._c.num_units, 'binary', act=act)
    self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
    self._actor = models.ActionDecoder(
        self._actdim, 4, self._c.num_units, self._c.action_dist,
        init_std=self._c.action_init_std, act=act)
    model_modules = [self._encode, self._dynamics, self._decode, self._reward]
    if self._c.pcont:
      model_modules.append(self._pcont)
    Optimizer = functools.partial(
        tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
        wdpattern=self._c.weight_decay_pattern)
    self._model_opt = Optimizer('model', model_modules, self._c.model_lr)
    self._value_opt = Optimizer('value', [self._value], self._c.value_lr)
    self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)
    # Do a train step to initialize all variables, including optimizer
    # statistics. Ideally, we would use batch size zero, but that doesn't work
    # in multi-GPU mode.
    self.train(next(self._dataset))

  def _exploration(self, action, training):
    if training:
      amount = self._c.expl_amount
      if self._c.expl_decay:
        amount *= 0.5 ** (tf.cast(self._step, tf.float32) / self._c.expl_decay)
      if self._c.expl_min:
        amount = tf.maximum(self._c.expl_min, amount)
      self._metrics['expl_amount'].update_state(amount)
    elif self._c.eval_noise:
      amount = self._c.eval_noise
    else:
      return action
    if self._c.expl == 'additive_gaussian':
      return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
    if self._c.expl == 'completely_random':
      return tf.random.uniform(action.shape, -1, 1)
    if self._c.expl == 'epsilon_greedy':
      indices = tfd.Categorical(0 * action).sample()
      return tf.where(
          tf.random.uniform(action.shape[:1], 0, 1) < amount,
          tf.one_hot(indices, action.shape[-1], dtype=self._float),
          action)
    raise NotImplementedError(self._c.expl)

  def _imagine_ahead(self, post):
    if self._c.pcont:  # Last step could be terminal.
      post = {k: v[:, :-1] for k, v in post.items()}
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in post.items()}
    policy = lambda state: self._actor(
        tf.stop_gradient(self._dynamics.get_feat(state))).sample()
    states = tools.static_scan(
        lambda prev, _: self._dynamics.img_step(prev, policy(prev)),
          tf.range(self._c.horizon), start)
    imag_feat = self._dynamics.get_feat(states)
    return imag_feat

  def _scalar_summaries(
      self, data, feat, prior_dist, post_dist, likes, div,
      model_loss, value_loss, actor_loss, model_norm, value_norm,
      actor_norm):
    self._metrics['model_grad_norm'].update_state(model_norm)
    self._metrics['value_grad_norm'].update_state(value_norm)
    self._metrics['actor_grad_norm'].update_state(actor_norm)
    self._metrics['prior_ent'].update_state(prior_dist.entropy())
    self._metrics['post_ent'].update_state(post_dist.entropy())
    for name, logprob in likes.items():
      self._metrics[name + '_loss'].update_state(-logprob)
    self._metrics['div'].update_state(div)
    self._metrics['model_loss'].update_state(model_loss)
    self._metrics['value_loss'].update_state(value_loss)
    self._metrics['actor_loss'].update_state(actor_loss)
    self._metrics['action_ent'].update_state(self._actor(feat).entropy())

  def _image_summaries(self, data, embed, image_pred):
    summary_size = 6  # nr images to be shown
    summary_length = 5  # nr step (length) of each gif
    if self._c.obs_type == 'image':
      truth = data['image'][:summary_size] + 0.5
      recon = image_pred.mode()[:summary_size]
      init, _ = self._dynamics.observe(embed[:summary_size, :summary_length],
                                       data['action'][:summary_size, :summary_length])
      init = {k: v[:, -1] for k, v in init.items()}
      prior = self._dynamics.imagine(data['action'][:summary_size, summary_length:], init)
      openl = self._decode(self._dynamics.get_feat(prior)).mode()
      model = tf.concat([recon[:, :summary_length] + 0.5, openl + 0.5], 1)
      error = (model - truth + 1) / 2
      openl = tf.concat([truth, model, error], 2)
    elif self._c.obs_type == 'lidar':
      truth = data['lidar'][:summary_size] + 0.5
      recon = image_pred.mode()[:summary_size]
      init, _ = self._dynamics.observe(embed[:summary_size, :summary_length],
                                       data['action'][:summary_size, :summary_length])
      init = {k: v[:, -1] for k, v in init.items()}
      prior = self._dynamics.imagine(data['action'][:summary_size, summary_length:], init)
      openl = self._decode(self._dynamics.get_feat(prior)).mode()
      model = tf.concat([recon[:, :summary_length] + 0.5, openl + 0.5], 1)
      truth_img = tools.lidar_to_image(truth)
      model_img = tools.lidar_to_image(model)
      error = model_img - truth_img
      openl = tf.concat([truth_img, model_img, error], 2)
    tools.graph_summary(self._writer, tools.video_summary,
                        'agent/train/autoencoder', openl, self._step, int(100/self._c.action_repeat))

  def _reward_summaries(self, data, reward_pred):
    summary_size = 6  # nr images to be shown
    truth = tools.reward_to_image(data['reward'][:summary_size])
    model = tools.reward_to_image(reward_pred.mode()[:summary_size])
    error = model - truth
    video_image = tf.concat([truth, model, error], 1)  # note: no T dimension, then stack over dim 1
    video_image = tf.expand_dims(video_image, axis=1)  # since no gif, expand dim=1 (T), B,H,W,C -> B,T,H,W,C
    tools.graph_summary(self._writer, tools.video_summary,
                        'agent/train/reward', video_image, self._step, int(100/self._c.action_repeat))

  def _write_summaries(self):
    step = int(self._step.numpy())
    metrics = [(k, float(v.result())) for k, v in self._metrics.items()]
    if self._last_log is not None:
      duration = time.time() - self._last_time
      self._last_time += duration
      metrics.append(('fps', (step - self._last_log) / duration))
    self._last_log = step
    [m.reset_states() for m in self._metrics.values()]
    with (self._c.logdir / 'metrics.jsonl').open('a') as f:
      f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
    [tf.summary.scalar('agent/' + k, m) for k, m in metrics]
    print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
    self._writer.flush()

def count_steps(datadir, config):
  return tools.count_episodes(datadir)[1] * config.action_repeat

def preprocess(obs, config):
  dtype = prec.global_policy().compute_dtype
  obs = obs.copy()
  with tf.device('cpu:0'):
    if 'image' in obs:
      obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
    obs['lidar'] = tf.cast(obs['lidar'], dtype) / 15.0 - 0.5
    clip_rewards = dict(none=lambda x: x, tanh=tf.tanh,
                        clip=lambda x: tf.clip_by_value(x, config.clip_rewards_min, config.clip_rewards_max))[config.clip_rewards]
    obs['reward'] = clip_rewards(obs['reward'])
  return obs

def load_dataset(directory, config):
  episode = next(tools.load_episodes(directory, 1))
  types = {k: v.dtype for k, v in episode.items()}
  shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}
  generator = lambda: tools.load_episodes(
      directory, config.train_steps, config.batch_length,
      config.dataset_balance)
  dataset = tf.data.Dataset.from_generator(generator, types, shapes)
  dataset = dataset.map(functools.partial(preprocess, config=config))
  dataset = dataset.batch(config.batch_size, drop_remainder=True)
  dataset = dataset.prefetch(10)
  return dataset

def summarize_episode(episodes, config, datadir, writer, prefix):
  global best_return_so_far
  # note: in multi-agent, each agent produce 1 episode
  episode = episodes[0]   # we summarize w.r.t. the episode of the first agent
  episodes, steps = tools.count_episodes(datadir)
  episode_len = len(episode['reward']) - 1
  length = episode_len * config.action_repeat
  ret = episode['reward'].sum()
  print(f'{prefix.title()} episode of length {episode_len} ({length} sim steps) with return {ret:.1f}.')
  metrics = [
      (f'{prefix}/return', float(episode['reward'].sum())),
      (f'{prefix}/length', len(episode['reward']) - 1),
      (f'{prefix}/progress', float(episode['progress'][-1])),
      (f'episodes', episodes)]
  step = count_steps(datadir, config)
  with (config.logdir / 'metrics.jsonl').open('a') as f:
    f.write(json.dumps(dict([('step', step)] + metrics)) + '\n')
  with writer.as_default():  # Env might run in a different thread.
    tf.summary.experimental.set_step(step)
    [tf.summary.scalar('sim/' + k, v) for k, v in metrics]


def render_episode(videos, config, datadir):
  if not config.log_videos:
    return
  step = count_steps(datadir, config)
  video_dir = config.logdir / f'video/{step}'
  video_dir.mkdir(parents=True, exist_ok=True)
  import imageio
  for filename, video in videos.items():
    writer = imageio.get_writer(f'{video_dir}/{filename}.mp4', fps=100 // config.action_repeat)
    for image in video:
      writer.append_data(image)
    writer.close()

def make_env(config, writer, prefix, datadir, store, gui=False):
  suite, track = config.task.split('_', 1)
  if suite == 'racecar':
    env = wrappers.RaceCarWrapper(track=track, prefix=prefix, id='A', rendering=gui)
    env = wrappers.ActionRepeat(env, config.action_repeat)
    env = wrappers.ReduceActionSpace(env, low=[0.005, -1.0], high=[1.0, 1.0])
  else:
    raise NotImplementedError(suite)
  env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
  if prefix == 'test':
    render_callbacks = []
    render_callbacks.append(lambda videos: render_episode(videos, config, datadir))
    env = wrappers.Render(env, render_callbacks)
  callbacks = []
  if store:
    callbacks.append(lambda episodes: tools.save_episodes(datadir, episodes))
  callbacks.append(
      lambda episodes: summarize_episode(episodes, config, datadir, writer, prefix))
  env = wrappers.Collect(env, callbacks, config.precision)
  #env = wrappers.RewardObs(env)  # do we really need it?
  return env


def write_config_summary(config):
  from datetime import datetime
  text = f'created at {datetime.now().strftime("%m-%d-%Y %H:%M:%S")}\n\n'
  for key in vars(config):
    text += f'{key}:{getattr(config, key)}\n'
  with open(os.path.join(config.logdir, 'config.txt'), 'w') as f:
    f.write(text)

def main(config):
  if config.gpu_growth:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    prec.set_policy(prec.Policy('mixed_float16'))
  config.steps = int(config.steps)
  config.logdir.mkdir(parents=True, exist_ok=True)
  checkpoint_dir = config.logdir / 'checkpoints'
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  best_checkpoint_dir = checkpoint_dir / 'best'
  best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
  write_config_summary(config)
  print('Logdir', config.logdir)

  # Create environments.
  datadir = config.logdir / 'episodes'
  writer = tf.summary.create_file_writer(
      str(config.logdir), max_queue=1000, flush_millis=20000)
  writer.set_as_default()

  train_env = make_env(config, writer, 'train', datadir, store=True, gui=False)
  test_env = make_env(config, writer, 'test', datadir, store=False, gui=False)
  agent_ids = train_env.agent_ids

  actspace = train_env.action_space
  obspace = train_env.observation_space

  # Prefill dataset with random episodes.
  step = count_steps(datadir, config)
  prefill = max(0, config.prefill - step)
  print(f'Prefill dataset ({config.prefill_agent}) with {prefill} steps.')

  if config.prefill_agent=='random':
    id = agent_ids[0]
    random_agent = lambda o, d, s: ([train_env.action_space[id].sample()], None)  # note: it must work as single agent
    tools.simulate(random_agent, train_env, prefill / config.action_repeat, agents_ids=agent_ids)
  elif config.prefill_agent=='gap_follower':
    gapfollower = GapFollower()
    gap_follower_agent = lambda o, d, s: ([gapfollower.action(o)], None)
    tools.simulate(gap_follower_agent, train_env, prefill / config.action_repeat, agents_ids=agent_ids)
  else:
    raise NotImplementedError(f'prefill agent {config.prefill_agent} not implemented')
  writer.flush()

  # Train and regularly evaluate the agent.
  step = count_steps(datadir, config)
  print(f'Simulating agent for {config.steps-step} steps.')
  agent = Dreamer(config, datadir, actspace, obspace, writer)
  # Resume last checkpoint (checkpoints are `{checkpoint_dir}/{step}.pkl`
  checkpoints = sorted(checkpoint_dir.glob('*pkl'), key=lambda f: int(f.name.split('.')[0]))
  if len(checkpoints):
    last_checkpoint = checkpoints[-1]
    agent.load(last_checkpoint)
    print('Load checkpoint.')

  simulation_state = None
  best_test_return = 0.0
  while step < config.steps:
    # Evaluation phase
    print('Start evaluation.')
    _, cum_reward = tools.simulate(
        functools.partial(agent, training=False), test_env, episodes=1, agents_ids=agent_ids)
    writer.flush()
    # Save best model
    if (cum_reward > best_test_return):
      best_test_return = cum_reward
      for model in [agent._encode, agent._dynamics, agent._decode, agent._reward, agent._actor]:
        model.save(best_checkpoint_dir / f'{model._name}.pkl')
      agent.save(best_checkpoint_dir / 'variables.pkl')    # store also the whole model
    # Save regular checkpoint
    step = count_steps(datadir, config)
    agent.save(checkpoint_dir / f'{step}.pkl')
    # Training phase
    print('Start collection.')
    steps = config.eval_every // config.action_repeat
    simulation_state, _ = tools.simulate(agent, train_env, steps, sim_state=simulation_state, agents_ids=agent_ids)
    step = count_steps(datadir, config)

if __name__ == '__main__':
  try:
    import colored_traceback
    colored_traceback.add_hook()
  except ImportError:
    pass
  parser = argparse.ArgumentParser()
  for key, value in define_config().items():
    parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
  args = parser.parse_args()
  if args.multi_test:
    base_logdir = args.logdir
    for seed in [123456789, 234567891, 345678912, 456789123, 567891234]:
      args.seed = seed
      args.logdir = base_logdir / f'seed{seed}'
      main(args)
  else:
    main(args)
