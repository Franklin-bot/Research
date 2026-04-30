import math
import time

import numpy as np
import tensorflow.compat.v1 as tf
from tqdm.auto import tqdm

tf.disable_v2_behavior()


def xavier_init(fan_in, fan_out, constant=1):
    """Xavier initialization of network weights."""
    low = -constant * np.sqrt(1.0 / (fan_in + fan_out))
    high = constant * np.sqrt(1.0 / (fan_in + fan_out))
    return tf.random_uniform(
        (fan_in, fan_out),
        minval=low,
        maxval=high,
        dtype=tf.float64,
    )


class VariationalAutoencoder(object):
    """TF1-style VAE wrapped with the current training/inference interface."""

    def __init__(
        self,
        network_architecture,
        transfer_fct=tf.nn.relu,
        learning_rate=0.001,
        batch_size=100,
        vae_mode=False,
        vae_mode_modalities=False,
    ):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.vae_mode = vae_mode
        self.vae_mode_modalities = vae_mode_modalities

        self.n_mc = 4
        self.n_vis = 4

        self.n_input = network_architecture["n_input"]
        self.n_z = network_architecture["n_z"]

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float64, [None, self.n_input], name="InputData")
            self.x_noiseless = tf.placeholder(
                tf.float64,
                [None, self.n_input],
                name="NoiselessData",
            )
            self.layers = {}

            self.n_epoch = tf.placeholder_with_default(
                tf.zeros([], tf.float64),
                shape=[],
                name="EpochValue",
            )

            self._create_network()
            self._create_loss_optimizer()

            init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(init)
        self.profiler = None

    def cleanup(self):
        try:
            self.sess.close()
        except Exception:
            pass
        self.profiler = None

    def _slice_input(self, input_layer, size_mod):
        slices = []
        count = 0
        for size in self.network_architecture[size_mod]:
            new_slice = tf.slice(input_layer, [0, count], [-1, size])
            count += size
            slices.append(new_slice)
        return slices

    def _create_partial_network(self, name, input_layer):
        with tf.name_scope(name):
            self.layers[name] = [input_layer]
            for layer_size in self.network_architecture[name]:
                h = tf.Variable(
                    xavier_init(int(self.layers[name][-1].get_shape()[1]), layer_size)
                )
                b = tf.Variable(tf.zeros([layer_size], dtype=tf.float64))
                layer = self.transfer_fct(tf.add(tf.matmul(self.layers[name][-1], h), b))
                self.layers[name].append(layer)

    def _create_variational_network(self, input_layer, latent_size):
        input_layer_size = int(input_layer.get_shape()[1])

        h_mean = tf.Variable(xavier_init(input_layer_size, latent_size))
        h_var = tf.Variable(xavier_init(input_layer_size, latent_size))
        b_mean = tf.Variable(tf.zeros([latent_size], dtype=tf.float64))
        b_var = tf.Variable(tf.zeros([latent_size], dtype=tf.float64))
        mean = tf.add(tf.matmul(input_layer, h_mean), b_mean)
        log_sigma_sq = tf.log(tf.exp(tf.add(tf.matmul(input_layer, h_var), b_var)) + 0.0001)
        return mean, log_sigma_sq

    def _create_modalities_network(self, names, slices):
        for i in range(len(names)):
            self._create_partial_network(names[i], slices[i])

    def _create_mod_variational_network(self, names, sizes_mod):
        assert len(self.network_architecture[sizes_mod]) == len(names)
        sizes = self.network_architecture[sizes_mod]
        self.layers["final_means"] = []
        self.layers["final_sigmas"] = []
        for i in range(len(names)):
            mean, log_sigma_sq = self._create_variational_network(
                self.layers[names[i]][-1],
                sizes[i],
            )
            self.layers["final_means"].append(mean)
            self.layers["final_sigmas"].append(log_sigma_sq)
        global_mean = tf.concat(self.layers["final_means"], 1)
        global_sigma = tf.concat(self.layers["final_sigmas"], 1)
        self.layers["global_mean_reconstr"] = [global_mean]
        self.layers["global_sigma_reconstr"] = [global_sigma]
        return global_mean, global_sigma

    def _create_network(self):
        self.x_noiseless_sliced = self._slice_input(self.x_noiseless, "size_slices")
        slices = self._slice_input(self.x, "size_slices")
        self._create_modalities_network(["mod0", "mod1"], slices)

        self.output_mod = tf.concat([self.layers["mod0"][-1], self.layers["mod1"][-1]], 1)
        self.layers["concat"] = [self.output_mod]

        self._create_partial_network("enc_shared", self.output_mod)
        self.z_mean, self.z_log_sigma_sq = self._create_variational_network(
            self.layers["enc_shared"][-1],
            self.n_z,
        )

        if self.vae_mode:
            eps = tf.random_normal(tf.stack([tf.shape(self.x)[0], self.n_z]), 0, 1, dtype=tf.float64)
            self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        else:
            self.z = self.z_mean

        self._create_partial_network("dec_shared", self.z)

        slices_shared = self._slice_input(self.layers["dec_shared"][-1], "size_slices_shared")
        self._create_modalities_network(["mod0_2", "mod1_2"], slices_shared)

        self.x_reconstr, self.x_log_sigma_sq = self._create_mod_variational_network(
            ["mod0_2", "mod1_2"],
            "size_slices",
        )

    def _create_loss_optimizer(self):
        with tf.name_scope("Loss_Opt"):
            self.alpha = 1 - tf.minimum(self.n_epoch / 1000, 1)

            self.tmp_costs = []
            for i in range(len(self.layers["final_means"])):
                reconstr_loss = (
                    0.5
                    * tf.reduce_sum(
                        tf.square(self.x_noiseless_sliced[i] - self.layers["final_means"][i])
                        / tf.exp(self.layers["final_sigmas"][i]),
                        1,
                    )
                    + 0.5 * tf.reduce_sum(self.layers["final_sigmas"][i], 1)
                    + 0.5 * self.n_z / 2 * np.log(2 * math.pi)
                ) / self.network_architecture["size_slices"][i]
                self.tmp_costs.append(reconstr_loss)

            self.reconstr_loss = tf.reduce_mean(self.tmp_costs[0] + self.tmp_costs[1])

            self.latent_loss = -0.5 * tf.reduce_sum(
                1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq),
                1,
            )

            self.cost = tf.reduce_mean(
                self.reconstr_loss + tf.scalar_mul(self.alpha, self.latent_loss)
            )

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate
            ).minimize(self.cost)

            self.m_reconstr_loss = self.reconstr_loss
            self.m_latent_loss = tf.reduce_mean(self.latent_loss)

    def partial_fit(self, X, X_noiseless, epoch):
        opt, cost, recon, latent, x_rec, alpha = self.sess.run(
            (
                self.optimizer,
                self.cost,
                self.m_reconstr_loss,
                self.m_latent_loss,
                self.x_reconstr,
                self.alpha,
            ),
            feed_dict={
                self.x: X,
                self.x_noiseless: X_noiseless,
                self.n_epoch: epoch,
            },
        )
        return cost, recon, latent, x_rec, alpha

    def transform(self, X):
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        if z_mu is None:
            z_mu = np.random.normal(size=(1, self.n_z))
        return self.sess.run(self.x_reconstr, feed_dict={self.z: z_mu})

    def reconstruct(self, X_test):
        x_rec_mean, x_rec_log_sigma_sq = self.sess.run(
            (self.x_reconstr, self.x_log_sigma_sq),
            feed_dict={self.x: X_test, self.x_noiseless: X_test},
        )
        return x_rec_mean, x_rec_log_sigma_sq

    def save_checkpoint(self, checkpoint_prefix):
        return self.saver.save(self.sess, checkpoint_prefix)

    def load_checkpoint(self, checkpoint_prefix):
        self.saver.restore(self.sess, checkpoint_prefix)


def shuffle_data(x):
    y = list(x)
    np.random.shuffle(y)
    return np.asarray(y)


def train_whole(
    vae,
    input_data,
    checkpoint_prefix,
    batch_size=100,
    training_epochs=10,
    display_step=200,
):
    print("display_step:" + str(display_step))
    epoch_list = []
    avg_cost_list = []
    avg_recon_list = []
    avg_latent_list = []
    last_display_time = time.time()

    n_samples = input_data.shape[0]
    n_input = vae.n_input

    for epoch in tqdm(range(training_epochs)):
        avg_cost = 0.0
        avg_recon = 0.0
        avg_latent = 0.0
        total_batch = int(n_samples / batch_size)

        X_shuffled = shuffle_data(input_data)

        for i in range(total_batch):
            batch_xs_augmented = X_shuffled[batch_size * i : batch_size * i + batch_size]
            batch_xs = np.asarray(batch_xs_augmented[:, :n_input], dtype=np.float64)
            batch_xs_noiseless = np.asarray(batch_xs_augmented[:, n_input:], dtype=np.float64)

            cost, recon, latent, x_rec, alpha = vae.partial_fit(
                batch_xs,
                batch_xs_noiseless,
                epoch,
            )
            avg_cost += cost / n_samples * batch_size
            avg_recon += recon / n_samples * batch_size
            avg_latent += latent / n_samples * batch_size

        if epoch % display_step == 0:
            epoch_list.append(epoch)
            avg_cost_list.append(avg_cost)
            avg_recon_list.append(avg_recon)
            avg_latent_list.append(avg_latent)
            now = time.time()
            elapsed_s = now - last_display_time
            last_display_time = now
            print(
                "Epoch: %04d / %04d, Cost= %04f, Recon= %04f, Latent= %04f, Interval= %.3fs"
                % (epoch, training_epochs, avg_cost, avg_recon, avg_latent, elapsed_s)
            )

    vae.save_checkpoint(checkpoint_prefix)
    return epoch_list, avg_cost_list, avg_recon_list, avg_latent_list
