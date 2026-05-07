import math
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm


class XavierUniformLike(tf.keras.initializers.Initializer):
    def __init__(self, constant=1.0):
        self.constant = constant

    def __call__(self, shape, dtype=None):
        dtype = tf.as_dtype(dtype or tf.float64)
        fan_in, fan_out = shape
        limit = self.constant * np.sqrt(1.0 / (fan_in + fan_out))
        return tf.random.uniform(
            shape,
            minval=-limit,
            maxval=limit,
            dtype=dtype,
        )

    def get_config(self):
        return {"constant": self.constant}


class DenseStack(tf.keras.layers.Layer):
    def __init__(self, layer_sizes, transfer_fct, name):
        super().__init__(name=name, dtype=tf.float64)
        self.hidden_layers = [
            tf.keras.layers.Dense(
                layer_size,
                activation=transfer_fct,
                kernel_initializer=XavierUniformLike(),
                bias_initializer="zeros",
                dtype=tf.float64,
                name=f"{name}_dense_{index}",
            )
            for index, layer_size in enumerate(layer_sizes)
        ]

    def call(self, inputs):
        layer = inputs
        for dense in self.hidden_layers:
            layer = dense(layer)
        return layer


class VariationalProjection(tf.keras.layers.Layer):
    def __init__(self, latent_size, name):
        super().__init__(name=name, dtype=tf.float64)
        self.mean_layer = tf.keras.layers.Dense(
            latent_size,
            activation=None,
            kernel_initializer=XavierUniformLike(),
            bias_initializer="zeros",
            dtype=tf.float64,
            name=f"{name}_mean",
        )
        self.var_layer = tf.keras.layers.Dense(
            latent_size,
            activation=None,
            kernel_initializer=XavierUniformLike(),
            bias_initializer="zeros",
            dtype=tf.float64,
            name=f"{name}_var",
        )

    def call(self, inputs):
        mean = self.mean_layer(inputs)
        raw_log_sigma_sq = self.var_layer(inputs)
        log_sigma_sq = tf.math.log(tf.math.exp(raw_log_sigma_sq) + tf.constant(0.0001, tf.float64))
        return mean, log_sigma_sq


class NativeVariationalAutoencoder(tf.keras.Model):
    def __init__(self, network_architecture, transfer_fct):
        super().__init__(name="native_variational_autoencoder", dtype=tf.float64)
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.n_input = network_architecture["n_input"]
        self.n_z = network_architecture["n_z"]

        self.mod0 = DenseStack(network_architecture["mod0"], transfer_fct, "mod0")
        self.mod1 = DenseStack(network_architecture["mod1"], transfer_fct, "mod1")
        self.enc_shared = DenseStack(network_architecture["enc_shared"], transfer_fct, "enc_shared")
        self.latent_projection = VariationalProjection(self.n_z, "latent_projection")
        self.dec_shared = DenseStack(network_architecture["dec_shared"], transfer_fct, "dec_shared")
        self.mod0_2 = DenseStack(network_architecture["mod0_2"], transfer_fct, "mod0_2")
        self.mod1_2 = DenseStack(network_architecture["mod1_2"], transfer_fct, "mod1_2")
        self.mod0_projection = VariationalProjection(network_architecture["size_slices"][0], "mod0_projection")
        self.mod1_projection = VariationalProjection(network_architecture["size_slices"][1], "mod1_projection")

    def _slice_input(self, input_layer, size_mod):
        return tf.split(input_layer, self.network_architecture[size_mod], axis=1)

    def encode(self, inputs, sample_latent):
        mod0_input, mod1_input = self._slice_input(inputs, "size_slices")
        mod0_output = self.mod0(mod0_input)
        mod1_output = self.mod1(mod1_input)

        encoder_input = tf.concat([mod0_output, mod1_output], axis=1)
        encoder_output = self.enc_shared(encoder_input)
        z_mean, z_log_sigma_sq = self.latent_projection(encoder_output)

        if sample_latent:
            eps = tf.random.normal(
                tf.stack([tf.shape(inputs)[0], self.n_z]),
                mean=0.0,
                stddev=1.0,
                dtype=tf.float64,
            )
            z = z_mean + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps
        else:
            z = z_mean

        return z, z_mean, z_log_sigma_sq

    def decode(self, latent):
        decoder_output = self.dec_shared(latent)
        mod0_shared, mod1_shared = self._slice_input(decoder_output, "size_slices_shared")

        mod0_output = self.mod0_2(mod0_shared)
        mod1_output = self.mod1_2(mod1_shared)

        mod0_mean, mod0_log_sigma_sq = self.mod0_projection(mod0_output)
        mod1_mean, mod1_log_sigma_sq = self.mod1_projection(mod1_output)

        return {
            "final_means": [mod0_mean, mod1_mean],
            "final_sigmas": [mod0_log_sigma_sq, mod1_log_sigma_sq],
            "x_reconstr": tf.concat([mod0_mean, mod1_mean], axis=1),
            "x_log_sigma_sq": tf.concat([mod0_log_sigma_sq, mod1_log_sigma_sq], axis=1),
        }

    def call(self, inputs, sample_latent=False):
        z, z_mean, z_log_sigma_sq = self.encode(inputs, sample_latent=sample_latent)
        decoded = self.decode(z)
        decoded.update(
            {
                "z": z,
                "z_mean": z_mean,
                "z_log_sigma_sq": z_log_sigma_sq,
            }
        )
        return decoded


class VariationalAutoencoder(object):
    """TF2-native VAE wrapped with the current training/inference interface."""

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
        self.size_slices = tuple(network_architecture["size_slices"])
        self.recon_constant = tf.constant(
            0.5 * self.n_z / 2 * np.log(2 * math.pi),
            dtype=tf.float64,
        )

        self.model = NativeVariationalAutoencoder(network_architecture, transfer_fct)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        )
        self.profiler = None

        dummy_input = tf.zeros((1, self.n_input), dtype=tf.float64)
        self.model(dummy_input, sample_latent=False)
        self.optimizer.build(self.model.trainable_variables)
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self._train_step.get_concrete_function(
            tf.TensorSpec(shape=[None, self.n_input], dtype=tf.float64),
            tf.TensorSpec(shape=[None, self.n_input], dtype=tf.float64),
            tf.TensorSpec(shape=[], dtype=tf.float64),
        )
        self._transform.get_concrete_function(
            tf.TensorSpec(shape=[None, self.n_input], dtype=tf.float64)
        )
        self._decode.get_concrete_function(
            tf.TensorSpec(shape=[None, self.n_z], dtype=tf.float64)
        )
        self._reconstruct.get_concrete_function(
            tf.TensorSpec(shape=[None, self.n_input], dtype=tf.float64)
        )

    def cleanup(self):
        self.profiler = None

    def _compute_losses(self, outputs, x_noiseless, epoch):
        x_noiseless_sliced = tf.split(x_noiseless, self.size_slices, axis=1)
        alpha = tf.constant(1.0, tf.float64) - tf.minimum(
            epoch / tf.constant(1000.0, tf.float64),
            tf.constant(1.0, tf.float64),
        )

        tmp_costs = []
        for index, size in enumerate(self.size_slices):
            reconstr_loss = (
                0.5
                * tf.reduce_sum(
                    tf.square(x_noiseless_sliced[index] - outputs["final_means"][index])
                    / tf.exp(outputs["final_sigmas"][index]),
                    axis=1,
                )
                + 0.5 * tf.reduce_sum(outputs["final_sigmas"][index], axis=1)
                + self.recon_constant
            ) / tf.constant(float(size), dtype=tf.float64)
            tmp_costs.append(reconstr_loss)

        reconstr_loss = tf.reduce_mean(tmp_costs[0] + tmp_costs[1])
        latent_loss = -0.5 * tf.reduce_sum(
            1
            + outputs["z_log_sigma_sq"]
            - tf.square(outputs["z_mean"])
            - tf.exp(outputs["z_log_sigma_sq"]),
            axis=1,
        )
        cost = tf.reduce_mean(reconstr_loss + alpha * latent_loss)
        mean_latent_loss = tf.reduce_mean(latent_loss)
        return cost, reconstr_loss, mean_latent_loss, alpha

    @tf.function(reduce_retracing=True)
    def _train_step(self, x, x_noiseless, epoch):
        with tf.GradientTape() as tape:
            outputs = self.model(x, sample_latent=self.vae_mode)
            cost, reconstr_loss, latent_loss, alpha = self._compute_losses(
                outputs,
                x_noiseless,
                epoch,
            )

        gradients = tape.gradient(cost, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return cost, reconstr_loss, latent_loss, outputs["x_reconstr"], alpha

    @tf.function(reduce_retracing=True)
    def _transform(self, x):
        _, z_mean, _ = self.model.encode(x, sample_latent=False)
        return z_mean

    @tf.function(reduce_retracing=True)
    def _decode(self, latent):
        decoded = self.model.decode(latent)
        return decoded["x_reconstr"]

    @tf.function(reduce_retracing=True)
    def _reconstruct(self, x_test):
        outputs = self.model(x_test, sample_latent=self.vae_mode)
        return outputs["x_reconstr"], outputs["x_log_sigma_sq"]

    def partial_fit(self, X, X_noiseless, epoch):
        cost, recon, latent, x_rec, alpha = self._train_step(
            tf.convert_to_tensor(X, dtype=tf.float64),
            tf.convert_to_tensor(X_noiseless, dtype=tf.float64),
            tf.convert_to_tensor(float(epoch), dtype=tf.float64),
        )
        return (
            float(cost.numpy()),
            float(recon.numpy()),
            float(latent.numpy()),
            x_rec.numpy(),
            float(alpha.numpy()),
        )

    def transform(self, X):
        return self._transform(tf.convert_to_tensor(X, dtype=tf.float64)).numpy()

    def generate(self, z_mu=None):
        if z_mu is None:
            z_mu = np.random.normal(size=(1, self.n_z))
        return self._decode(tf.convert_to_tensor(z_mu, dtype=tf.float64)).numpy()

    def reconstruct(self, X_test):
        x_rec_mean, x_rec_log_sigma_sq = self._reconstruct(
            tf.convert_to_tensor(X_test, dtype=tf.float64)
        )
        return x_rec_mean.numpy(), x_rec_log_sigma_sq.numpy()

    def save_checkpoint(self, checkpoint_prefix):
        checkpoint_path = self.checkpoint.write(checkpoint_prefix)
        checkpoint_prefix_path = Path(checkpoint_prefix)
        checkpoint_state_path = checkpoint_prefix_path.parent / "checkpoint"
        checkpoint_state_path.write_text(
            'model_checkpoint_path: "{}"\nall_model_checkpoint_paths: "{}"\n'.format(
                checkpoint_prefix_path.name,
                checkpoint_prefix_path.name,
            )
        )
        return checkpoint_path

    def load_checkpoint(self, checkpoint_prefix):
        self.checkpoint.read(checkpoint_prefix).assert_existing_objects_matched()


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
