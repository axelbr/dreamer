import pathlib
from datetime import datetime

from dreamer import tools


def default():
  config = tools.AttrDict()
  # General.
  config.logdir = pathlib.Path("./logs/racecar_{}/".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
  config.seed = 0
  config.steps = int(5e6)
  config.eval_every = 1e4
  config.log_every = 1e3
  config.log_scalars = True
  config.log_images = False
  config.gpu_growth = True
  config.precision = 32
  config.obs_type = 'lidar'
  # Environment.
  config.task = 'racecar_MultiAgentAustria'
  config.envs = 1
  config.parallel = 'none'
  config.action_repeat = 4
  config.time_limit = 3000
  config.prefill = 300
  config.eval_noise = 0.0
  config.clip_rewards = 'none'
  # Model.
  config.encoded_obs_dim = 16
  config.deter_size = 200
  config.stoch_size = 30
  config.num_units = 400
  config.dense_act = 'elu'
  config.cnn_act = 'relu'
  config.cnn_depth = 32
  config.pcont = False
  config.free_nats = 3.0
  config.kl_scale = 1.0
  config.pcont_scale = 10.0
  config.weight_decay = 0.0
  config.weight_decay_pattern = r'.*'
  # Pretrained model
  config.use_pretrained_encoder = False
  config.check_load_pretrained_encoder = tools.Once() if config.use_pretrained_encoder else lambda : False
  config.pretrained_encoder_path = "racing_dreamer/pretrained_models/pretrained_encoder"
  # Training.
  config.batch_size = 64
  config.batch_length = 50
  config.train_every = 1000
  config.train_steps = 100
  config.pretrain = 100
  config.model_lr = 6e-4
  config.value_lr = 8e-5
  config.actor_lr = 8e-5
  config.grad_clip = 100.0
  config.dataset_balance = False
  # Behavior.
  config.discount = 0.99
  config.disclam = 0.95
  config.horizon = 15
  config.action_dist = 'tanh_normal'
  config.action_init_std = 5.0
  config.expl = 'additive_gaussian'
  config.expl_amount = 0.3
  config.expl_decay = 0.0
  config.expl_min = 0.0
  return config