import models
import numpy as np

encoded_obs_dim = 128
lidar_shape = 1000

encode = models.LidarEncoder(output_dim=encoded_obs_dim)
decode = models.LidarDecoder(output_dim=lidar_shape)

obs = {}
obs['lidar'] = np.zeros((1, 1000))
latent = encode(obs)

print(latent.shape)