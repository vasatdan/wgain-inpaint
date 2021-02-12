#!/usr/bin/python3 -u
# coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import data
from tensorflow import keras

from WGAIN_model import *

print(tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))


##########################################################################
batch_size = 32
side = 128

path = "./datasets/celeba/%s128"
celeba_train = data.Dataset.list_files(path%'train' + str('/*.jpg'))
celeba_train_d = prepare_training_dataset(celeba_train, batch_size, side)

# create the model
generator = build_generator(side)
critic = build_critic(side)
# print summaries
generator.summary()
critic.summary()
# prepare optimizers
generator_optimizer = tf.keras.optimizers.Adam(0.00005)
critic_optimizer = tf.keras.optimizers.Adam(0.00005)


@tf.function
def train_step(origX, newX, mask, randZ):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as critic_tape:
        g_out = generator([newX, mask, randZ])
        opinion_generated = critic([g_out, mask])
        opinion_real = critic([origX, mask])
        critic_w_l = wass_c_loss(opinion_generated, opinion_real)
        gen_w_l = 0.005*wass_g_loss(opinion_generated)
        gen_mse_l = g_loss(g_out, origX)
        gen_loss = gen_mse_l + gen_w_l

    g_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    c_gradients = critic_tape.gradient(critic_w_l, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(c_gradients,critic.trainable_variables))
    generator_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    return gen_mse_l, gen_w_l, critic_w_l

##### Main training
g_history = []
m_history = []
d_history = []

# Path to save
model_path = "saved_models_celeba/"

# main loop
for epoch in range(1,201):
    print("Epoch: %d" % epoch)
    # Store selected losses
    losses_g = []
    losses_m = []
    losses_d = []
    for i, (origX, newX, mask, randZ) in enumerate(celeba_train_d):
        print(".", end='', flush=True)
        mse_l, genw_l, critic_l = train_step(origX, newX, mask, randZ)
        losses_m.append(mse_l)
        losses_g.append(genw_l)
        losses_d.append(critic_l)
    print()
    g_history.append(np.mean(losses_g))
    m_history.append(np.mean(losses_m))
    d_history.append(np.mean(losses_d))
    print("gen W loss:", np.mean(losses_g))
    print("gen mae loss:", np.mean(losses_m))
    print("critic loss:", np.mean(losses_d))
    generator.save_weights("saved_models_celeba/gultimate_latest")
    critic.save_weights("saved_models_celeba/cultimate_latest")
    if epoch%1==0:
        generator.save_weights("saved_models_celeba/gultimate_ckpt%d" % epoch)
        critic.save_weights("saved_models_celeba/cultimate_ckpt%d" % epoch)


# Final save of weights
generator.save_weights("saved_models_celeba/gultimate_final")
critic.save_weights("saved_models_celeba/cultimate_final")
