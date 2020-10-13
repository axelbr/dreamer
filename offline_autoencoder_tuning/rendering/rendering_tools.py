import tensorflow as tf
import tools
import imageio
import numpy as np

def _image_summaries(lidar, embed, image_pred, name=""):
    recon = image_pred.mode()
    if len(lidar.shape)<3:
        lidar = tf.expand_dims(lidar, axis=0)
    if len(recon.shape) < 3:
        recon = tf.expand_dims(recon, axis=0)
    lidar_img = tools.lidar_to_image(tf.cast(lidar, dtype='float16'))
    recon_img = tools.lidar_to_image(tf.cast(recon, dtype='float16'))
    video = tf.concat([lidar_img, recon_img], 1)
    gif_summary(video, name=name)

def gif_summary(video, name="", fps=30):
    frames = []
    for i in range(video.shape[0]):
        frames.append(video[i].numpy().astype(np.uint8))
    imageio.mimsave("offline_autoencoder_tuning/gif/seq_lidar_{}.gif".format(name), frames)