import math
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm


def _glorot_uniform():
    return tf.keras.initializers.GlorotUniform()


def _dense(units, activation=None, name=None):
    return tf.keras.layers.Dense(
        units,
        activation=activation,
        dtype=tf.float64,
        kernel_initializer=_glorot_uniform(),
        bias_initializer="zeros",
        name=name,
    )


LOG_SIGMA_EPS = 1e-4


class VariationalAutoencoder(tf.keras.Model):
    """TensorFlow 2 VAE that preserves the original model architecture and loss."""

    def __init__(
        self,
        network_architecture,
        transfer_fct=tf.nn.relu,
        learning_rate=0.001,
        batch_size=100,
        vae_mode=False,
        vae_mode_modalities=False,
    ):
        super().__init__(dtype=tf.float64, name="variational_autoencoder")
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
        self.size_slices = network_architecture["size_slices"]
        self.size_slices_shared = network_architecture["size_slices_shared"]

        self.mod_encoders = [
            self._make_mlp(network_architecture["mod0"], "mod0"),
            self._make_mlp(network_architecture["mod1"], "mod1"),
        ]
        self.enc_shared = self._make_mlp(network_architecture["enc_shared"], "enc_shared")

        self.z_mean_layer = _dense(self.n_z, name="z_mean")
        self.z_log_sigma_layer = _dense(self.n_z, name="z_log_sigma")

        self.dec_shared = self._make_mlp(network_architecture["dec_shared"], "dec_shared")
        self.mod_decoders = [
            self._make_mlp(network_architecture["mod0_2"], "mod0_2"),
            self._make_mlp(network_architecture["mod1_2"], "mod1_2"),
        ]
        self.output_mean_layers = [
            _dense(self.size_slices[0], name="mod0_mean"),
            _dense(self.size_slices[1], name="mod1_mean"),
        ]
        self.output_log_sigma_layers = [
            _dense(self.size_slices[0], name="mod0_log_sigma"),
            _dense(self.size_slices[1], name="mod1_log_sigma"),
        ]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self._build_variables()

    def _make_mlp(self, layer_sizes, name):
        return tf.keras.Sequential(
            [_dense(units, activation=self.transfer_fct, name=f"{name}_{i}") for i, units in enumerate(layer_sizes)],
            name=name,
        )

    def _build_variables(self):
        dummy = tf.zeros((1, self.n_input), dtype=tf.float64)
        self._forward(dummy, training=False)

    @staticmethod
    def _stable_log_sigma_sq(raw_values):
        # Numerically stable equivalent of the original TF1 expression:
        # log(exp(raw_values) + 1e-4)
        log_eps = tf.cast(math.log(LOG_SIGMA_EPS), dtype=tf.float64)
        return tf.reduce_logsumexp(
            tf.stack(
                [
                    raw_values,
                    tf.fill(tf.shape(raw_values), log_eps),
                ],
                axis=0,
            ),
            axis=0,
        )

    def _slice_input(self, input_tensor, sizes):
        return tf.split(input_tensor, sizes, axis=1)

    def _sample_latent(self, z_mean, z_log_sigma_sq, training):
        if training and self.vae_mode:
            eps = tf.random.normal(tf.shape(z_mean), 0.0, 1.0, dtype=tf.float64)
            return z_mean + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps
        return z_mean

    def _decode_from_latent(self, z, training=False):
        dec_shared = self.dec_shared(z, training=training)
        shared_slices = self._slice_input(dec_shared, self.size_slices_shared)

        final_means = []
        final_sigmas = []
        for decoder, mean_layer, sigma_layer, shared_slice in zip(
            self.mod_decoders,
            self.output_mean_layers,
            self.output_log_sigma_layers,
            shared_slices,
        ):
            decoded = decoder(shared_slice, training=training)
            mean = mean_layer(decoded)
            log_sigma_sq = self._stable_log_sigma_sq(sigma_layer(decoded))
            final_means.append(mean)
            final_sigmas.append(log_sigma_sq)

        return (
            tf.concat(final_means, axis=1),
            tf.concat(final_sigmas, axis=1),
            final_means,
            final_sigmas,
        )

    def _forward(self, x, training=False):
        slices = self._slice_input(x, self.size_slices)
        mod_outputs = [
            encoder(mod_slice, training=training)
            for encoder, mod_slice in zip(self.mod_encoders, slices)
        ]
        output_mod = tf.concat(mod_outputs, axis=1)
        enc_shared = self.enc_shared(output_mod, training=training)
        z_mean = self.z_mean_layer(enc_shared)
        z_log_sigma_sq = self._stable_log_sigma_sq(self.z_log_sigma_layer(enc_shared))
        z = self._sample_latent(z_mean, z_log_sigma_sq, training=training)
        x_reconstr, x_log_sigma_sq, final_means, final_sigmas = self._decode_from_latent(
            z,
            training=training,
        )
        return {
            "x_reconstr": x_reconstr,
            "x_log_sigma_sq": x_log_sigma_sq,
            "z_mean": z_mean,
            "z_log_sigma_sq": z_log_sigma_sq,
            "final_means": final_means,
            "final_sigmas": final_sigmas,
        }

    def call(self, inputs, training=False):
        return self._forward(inputs, training=training)["x_reconstr"]

    def compute_losses(self, x, x_noiseless, epoch, training):
        outputs = self._forward(x, training=training)
        x_noiseless_sliced = self._slice_input(x_noiseless, self.size_slices)

        epoch = tf.cast(epoch, tf.float64)
        alpha = 1.0 - tf.minimum(epoch / 1000.0, 1.0)

        tmp_costs = []
        for i, (target_slice, mean, log_sigma_sq) in enumerate(
            zip(x_noiseless_sliced, outputs["final_means"], outputs["final_sigmas"])
        ):
            reconstr_loss = (
                0.5
                * tf.reduce_sum(tf.square(target_slice - mean) / tf.exp(log_sigma_sq), axis=1)
                + 0.5 * tf.reduce_sum(log_sigma_sq, axis=1)
                + 0.5 * self.n_z / 2 * np.log(2 * math.pi)
            ) / self.network_architecture["size_slices"][i]
            tmp_costs.append(reconstr_loss)

        reconstr_loss = tf.reduce_mean(tmp_costs[0] + tmp_costs[1])
        latent_loss = -0.5 * tf.reduce_sum(
            1 + outputs["z_log_sigma_sq"] - tf.square(outputs["z_mean"]) - tf.exp(outputs["z_log_sigma_sq"]),
            axis=1,
        )
        latent_loss_mean = tf.reduce_mean(latent_loss)
        cost = tf.reduce_mean(reconstr_loss + alpha * latent_loss)

        return {
            "cost": cost,
            "recon": reconstr_loss,
            "latent": latent_loss_mean,
            "x_reconstr": outputs["x_reconstr"],
            "x_log_sigma_sq": outputs["x_log_sigma_sq"],
            "alpha": alpha,
            "z_mean": outputs["z_mean"],
        }

    def partial_fit(self, X, X_noiseless, epoch):
        x = tf.convert_to_tensor(X, dtype=tf.float64)
        x_noiseless = tf.convert_to_tensor(X_noiseless, dtype=tf.float64)
        with tf.GradientTape() as tape:
            losses = self.compute_losses(x, x_noiseless, epoch, training=True)
        grads = tape.gradient(losses["cost"], self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return (
            float(losses["cost"].numpy()),
            float(losses["recon"].numpy()),
            float(losses["latent"].numpy()),
            losses["x_reconstr"].numpy(),
            float(losses["alpha"].numpy()),
        )

    def transform(self, X):
        x = tf.convert_to_tensor(X, dtype=tf.float64)
        outputs = self.compute_losses(x, x, epoch=1000.0, training=False)
        return outputs["z_mean"].numpy()

    def generate(self, z_mu=None):
        if z_mu is None:
            z_mu = np.random.normal(size=(1, self.n_z))
        z = tf.convert_to_tensor(z_mu, dtype=tf.float64)
        x_reconstr, _, _, _ = self._decode_from_latent(z, training=False)
        return x_reconstr.numpy()

    def reconstruct(self, X_test):
        x = tf.convert_to_tensor(X_test, dtype=tf.float64)
        outputs = self._forward(x, training=False)
        return outputs["x_reconstr"].numpy(), outputs["x_log_sigma_sq"].numpy()

    def save_checkpoint(self, checkpoint_prefix):
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.write(checkpoint_prefix)
        return checkpoint_prefix

    def load_checkpoint(self, checkpoint_prefix):
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.restore(checkpoint_prefix).expect_partial()

    def cleanup(self):
        return None


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

    for epoch in tqdm(range(training_epochs)):
        avg_cost = 0.0
        avg_recon = 0.0
        avg_latent = 0.0
        total_batch = int(n_samples / batch_size)

        X_shuffled = shuffle_data(input_data)

        for i in range(total_batch):
            batch_xs_augmented = X_shuffled[batch_size * i : batch_size * i + batch_size]
            batch_xs_augmented = np.asarray(batch_xs_augmented)
            if batch_xs_augmented.shape[1] == vae.n_input * 2:
                batch_xs = batch_xs_augmented[:, : vae.n_input]
                batch_xs_noiseless = batch_xs_augmented[:, vae.n_input :]
            else:
                batch_xs = batch_xs_augmented
                batch_xs_noiseless = batch_xs_augmented

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

    checkpoint_prefix = Path(checkpoint_prefix)
    checkpoint_prefix.parent.mkdir(parents=True, exist_ok=True)
    vae.save_checkpoint(str(checkpoint_prefix))
    return epoch_list, avg_cost_list, avg_recon_list, avg_latent_list
