import models
import numpy as np
import h5py as h5
import os

encoded_obs_dim = 128
lidar_shape = 1000

encode = models.LidarEncoder(output_dim=encoded_obs_dim)
decode = models.LidarDecoder(output_dim=lidar_shape)

output_dir = "out"
dataset_filename = "dataset_random_starts_austria_2000episodes_1000maxobs.h5"
data = h5.File(os.path.join(output_dir, dataset_filename), "r")
print(data.keys())



obs = {}
obs['lidar'] = np.zeros((1, 1000))
latent = encode(obs)

print(latent.shape)