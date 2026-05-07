import math
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tqdm.auto import tqdm

TF_DTYPE = tf.float64
LOG_SIGMA_EPS = 1e-4


def xavier_limit(fan_in, fan_out, constant=1.0):
    return constant * np.sqrt(1.0 / (fan_in + fan_out))


def xavier_tensor(shape, constant=1.0):
    fan_in, fan_out = shape
    limit = xavier_limit(fan_in, fan_out, constant=constant)
    return tf.random.uniform(
        shape,
        minval=-limit,
        maxval=limit,
        dtype=TF_DTYPE,
    )


class LegacyReferenceGraph:
    """
    Small TF1-style reference graph used only to reproduce seeded initialization
    and latent epsilon sampling from the old implementation.
    """

    def __init__(self, network_architecture, vae_mode=False, seed=None):
        self.network_architecture = network_architecture
        self.vae_mode = vae_mode
        self.seed = seed
        self.n_input = network_architecture["n_input"]
        self.n_z = network_architecture["n_z"]

        self.graph = tf1.Graph()
        with self.graph.as_default():
            if seed is not None:
                tf1.set_random_seed(seed)

            self.x = tf1.placeholder(TF_DTYPE, [None, self.n_input], name="InputData")
            self.x_noiseless = tf1.placeholder(TF_DTYPE, [None, self.n_input], name="NoiselessData")
            self.layers = {}
            self.eps = None

            self._create_network()
            self.init_op = tf1.global_variables_initializer()

        self.sess = tf1.Session(graph=self.graph)
        self.sess.run(self.init_op)

    def close(self):
        try:
            self.sess.close()
        except Exception:
            pass

    def trainable_values(self):
        variables = self.graph.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES)
        return self.sess.run(variables)

    def sample_eps(self, batch_size):
        if self.eps is None:
            raise ValueError("Legacy epsilon sampling requested with vae_mode=False.")
        dummy_x = np.zeros((batch_size, self.n_input), dtype=np.float64)
        return self.sess.run(self.eps, feed_dict={self.x: dummy_x})

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
            last_width = int(input_layer.get_shape()[1])
            for layer_size in self.network_architecture[name]:
                limit = xavier_limit(last_width, layer_size)
                h = tf.Variable(
                    tf1.random_uniform(
                        (last_width, layer_size),
                        minval=-limit,
                        maxval=limit,
                        dtype=TF_DTYPE,
                    )
                )
                b = tf.Variable(tf.zeros([layer_size], dtype=TF_DTYPE))
                layer = tf.nn.relu(tf.add(tf.matmul(self.layers[name][-1], h), b))
                self.layers[name].append(layer)
                last_width = layer_size

    def _create_variational_network(self, input_layer, latent_size):
        input_layer_size = int(input_layer.get_shape()[1])
        limit = xavier_limit(input_layer_size, latent_size)
        h_mean = tf.Variable(
            tf1.random_uniform(
                (input_layer_size, latent_size),
                minval=-limit,
                maxval=limit,
                dtype=TF_DTYPE,
            )
        )
        h_var = tf.Variable(
            tf1.random_uniform(
                (input_layer_size, latent_size),
                minval=-limit,
                maxval=limit,
                dtype=TF_DTYPE,
            )
        )
        b_mean = tf.Variable(tf.zeros([latent_size], dtype=TF_DTYPE))
        b_var = tf.Variable(tf.zeros([latent_size], dtype=TF_DTYPE))
        mean = tf.add(tf.matmul(input_layer, h_mean), b_mean)
        log_sigma_sq = tf.math.log(tf.exp(tf.add(tf.matmul(input_layer, h_var), b_var)) + LOG_SIGMA_EPS)
        return mean, log_sigma_sq

    def _create_modalities_network(self, names, slices):
        for i in range(len(names)):
            self._create_partial_network(names[i], slices[i])

    def _create_mod_variational_network(self, names, sizes_mod):
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

    def _create_network(self):
        slices = self._slice_input(self.x, "size_slices")
        self._create_modalities_network(["mod0", "mod1"], slices)

        output_mod = tf.concat([self.layers["mod0"][-1], self.layers["mod1"][-1]], 1)
        self.layers["concat"] = [output_mod]

        self._create_partial_network("enc_shared", output_mod)
        self.z_mean, self.z_log_sigma_sq = self._create_variational_network(
            self.layers["enc_shared"][-1],
            self.n_z,
        )

        if self.vae_mode:
            self.eps = tf1.random_normal(
                tf.stack([tf.shape(self.x)[0], self.n_z]),
                0,
                1,
                dtype=TF_DTYPE,
            )
            z = self.z_mean + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * self.eps
        else:
            z = self.z_mean

        self._create_partial_network("dec_shared", z)
        slices_shared = self._slice_input(self.layers["dec_shared"][-1], "size_slices_shared")
        self._create_modalities_network(["mod0_2", "mod1_2"], slices_shared)
        self._create_mod_variational_network(["mod0_2", "mod1_2"], "size_slices")


class VariationalAutoencoder(tf.Module):
    """Native TF2 VAE with legacy-seeded parity hooks for the old implementation."""

    def __init__(
        self,
        network_architecture,
        transfer_fct=tf.nn.relu,
        learning_rate=0.001,
        batch_size=100,
        vae_mode=False,
        vae_mode_modalities=False,
        seed=None,
    ):
        super().__init__(name="variational_autoencoder")
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.vae_mode = vae_mode
        self.vae_mode_modalities = vae_mode_modalities
        self.seed = seed

        self.n_mc = 4
        self.n_vis = 4

        self.n_input = network_architecture["n_input"]
        self.n_z = network_architecture["n_z"]
        self.size_slices = list(network_architecture["size_slices"])
        self.size_slices_shared = list(network_architecture["size_slices_shared"])

        self._legacy_reference = None
        self._legacy_init_iter = None
        if seed is not None:
            self._legacy_reference = LegacyReferenceGraph(
                network_architecture,
                vae_mode=vae_mode,
                seed=seed,
            )
            self._legacy_init_iter = iter(self._legacy_reference.trainable_values())

        self._trainable_vars = []

        self.mod0_kernels, self.mod0_biases = self._create_dense_stack_vars(13, network_architecture["mod0"], "mod0")
        self.mod1_kernels, self.mod1_biases = self._create_dense_stack_vars(80, network_architecture["mod1"], "mod1")
        self.enc_shared_kernels, self.enc_shared_biases = self._create_dense_stack_vars(
            148,
            network_architecture["enc_shared"],
            "enc_shared",
        )
        self.z_params = self._create_variational_vars(350, self.n_z, "z")
        self.dec_shared_kernels, self.dec_shared_biases = self._create_dense_stack_vars(
            self.n_z,
            network_architecture["dec_shared"],
            "dec_shared",
        )
        self.mod0_2_kernels, self.mod0_2_biases = self._create_dense_stack_vars(
            48,
            network_architecture["mod0_2"],
            "mod0_2",
        )
        self.mod1_2_kernels, self.mod1_2_biases = self._create_dense_stack_vars(
            100,
            network_architecture["mod1_2"],
            "mod1_2",
        )
        self.out0_params = self._create_variational_vars(13, self.size_slices[0], "out0")
        self.out1_params = self._create_variational_vars(80, self.size_slices[1], "out1")

        try:
            self.optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
            )
        except (AttributeError, ImportError):
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
            )

        if self._legacy_reference is not None and not self.vae_mode:
            self._legacy_reference.close()
            self._legacy_reference = None

    @property
    def trainable_variables(self):
        return self._trainable_vars

    def _next_init_value(self, shape, is_bias):
        if self._legacy_init_iter is not None:
            value = next(self._legacy_init_iter)
            if tuple(value.shape) != tuple(shape):
                raise ValueError(f"Legacy init shape mismatch: expected {shape}, got {value.shape}")
            return value
        if is_bias:
            return np.zeros(shape, dtype=np.float64)
        fan_in, fan_out = shape
        limit = xavier_limit(fan_in, fan_out)
        return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=TF_DTYPE).numpy()

    def _create_variable(self, shape, name, is_bias=False):
        init_value = self._next_init_value(shape, is_bias=is_bias)
        var = tf.Variable(init_value, dtype=TF_DTYPE, trainable=True, name=name)
        self._trainable_vars.append(var)
        return var

    def _create_dense_stack_vars(self, input_dim, layer_sizes, prefix):
        kernels = []
        biases = []
        in_dim = input_dim
        for index, layer_size in enumerate(layer_sizes):
            kernels.append(self._create_variable((in_dim, layer_size), f"{prefix}_kernel_{index}"))
            biases.append(self._create_variable((layer_size,), f"{prefix}_bias_{index}", is_bias=True))
            in_dim = layer_size
        return kernels, biases

    def _create_variational_vars(self, input_dim, latent_size, prefix):
        mean_kernel = self._create_variable((input_dim, latent_size), f"{prefix}_mean_kernel")
        log_kernel = self._create_variable((input_dim, latent_size), f"{prefix}_log_kernel")
        mean_bias = self._create_variable((latent_size,), f"{prefix}_mean_bias", is_bias=True)
        log_bias = self._create_variable((latent_size,), f"{prefix}_log_bias", is_bias=True)
        return mean_kernel, log_kernel, mean_bias, log_bias

    def _slice_input(self, input_tensor, sizes):
        return tf.split(input_tensor, sizes, axis=1)

    def _forward_dense_stack(self, inputs, kernels, biases):
        layer = inputs
        for kernel, bias in zip(kernels, biases):
            layer = self.transfer_fct(tf.matmul(layer, kernel) + bias)
        return layer

    def _forward_variational(self, inputs, params):
        mean_kernel, log_kernel, mean_bias, log_bias = params
        mean = tf.matmul(inputs, mean_kernel) + mean_bias
        raw_log_sigma = tf.matmul(inputs, log_kernel) + log_bias
        log_sigma_sq = tf.math.log(tf.exp(raw_log_sigma) + tf.cast(LOG_SIGMA_EPS, TF_DTYPE))
        return mean, log_sigma_sq

    def _decode_from_latent(self, z):
        dec_shared = self._forward_dense_stack(z, self.dec_shared_kernels, self.dec_shared_biases)
        mod0_shared, mod1_shared = self._slice_input(dec_shared, self.size_slices_shared)

        mod0_decoded = self._forward_dense_stack(mod0_shared, self.mod0_2_kernels, self.mod0_2_biases)
        mod1_decoded = self._forward_dense_stack(mod1_shared, self.mod1_2_kernels, self.mod1_2_biases)

        mod0_mean, mod0_log_sigma_sq = self._forward_variational(mod0_decoded, self.out0_params)
        mod1_mean, mod1_log_sigma_sq = self._forward_variational(mod1_decoded, self.out1_params)

        return {
            "final_means": [mod0_mean, mod1_mean],
            "final_sigmas": [mod0_log_sigma_sq, mod1_log_sigma_sq],
            "x_reconstr": tf.concat([mod0_mean, mod1_mean], axis=1),
            "x_log_sigma_sq": tf.concat([mod0_log_sigma_sq, mod1_log_sigma_sq], axis=1),
        }

    def _forward(self, x, training=False, epsilon=None):
        mod0_input, mod1_input = self._slice_input(x, self.size_slices)
        mod0_output = self._forward_dense_stack(mod0_input, self.mod0_kernels, self.mod0_biases)
        mod1_output = self._forward_dense_stack(mod1_input, self.mod1_kernels, self.mod1_biases)

        output_mod = tf.concat([mod0_output, mod1_output], axis=1)
        enc_shared = self._forward_dense_stack(output_mod, self.enc_shared_kernels, self.enc_shared_biases)
        z_mean, z_log_sigma_sq = self._forward_variational(enc_shared, self.z_params)

        if training and self.vae_mode:
            if epsilon is None:
                epsilon = tf.random.normal(tf.shape(z_mean), 0.0, 1.0, dtype=TF_DTYPE)
            z = z_mean + tf.sqrt(tf.exp(z_log_sigma_sq)) * epsilon
        else:
            z = z_mean

        decoded = self._decode_from_latent(z)
        decoded.update(
            {
                "z_mean": z_mean,
                "z_log_sigma_sq": z_log_sigma_sq,
            }
        )
        return decoded

    def compute_losses(self, x, x_noiseless, epoch, training, epsilon=None):
        outputs = self._forward(x, training=training, epsilon=epsilon)
        x_noiseless_sliced = self._slice_input(x_noiseless, self.size_slices)

        epoch = tf.cast(epoch, TF_DTYPE)
        alpha = tf.cast(1.0, TF_DTYPE) - tf.minimum(epoch / tf.cast(1000.0, TF_DTYPE), tf.cast(1.0, TF_DTYPE))

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

    @tf.function(reduce_retracing=True)
    def _train_step(self, x, x_noiseless, epoch, epsilon):
        with tf.GradientTape() as tape:
            losses = self.compute_losses(x, x_noiseless, epoch, training=True, epsilon=epsilon)
        grads = tape.gradient(losses["cost"], self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return losses

    def _legacy_epsilon(self, batch_size):
        if self._legacy_reference is not None and self.vae_mode:
            return self._legacy_reference.sample_eps(batch_size)
        return np.random.normal(size=(batch_size, self.n_z))

    def partial_fit(self, X, X_noiseless, epoch):
        x = tf.convert_to_tensor(X, dtype=TF_DTYPE)
        x_noiseless = tf.convert_to_tensor(X_noiseless, dtype=TF_DTYPE)
        epoch = tf.convert_to_tensor(epoch, dtype=TF_DTYPE)
        epsilon = tf.convert_to_tensor(self._legacy_epsilon(X.shape[0]), dtype=TF_DTYPE)
        losses = self._train_step(x, x_noiseless, epoch, epsilon)
        return (
            float(losses["cost"].numpy()),
            float(losses["recon"].numpy()),
            float(losses["latent"].numpy()),
            losses["x_reconstr"].numpy(),
            float(losses["alpha"].numpy()),
        )

    def transform(self, X):
        x = tf.convert_to_tensor(X, dtype=TF_DTYPE)
        outputs = self.compute_losses(x, x, epoch=1000.0, training=False)
        return outputs["z_mean"].numpy()

    def generate(self, z_mu=None):
        if z_mu is None:
            z_mu = np.random.normal(size=(1, self.n_z))
        z = tf.convert_to_tensor(z_mu, dtype=TF_DTYPE)
        outputs = self._decode_from_latent(z)
        return outputs["x_reconstr"].numpy()

    def reconstruct(self, X_test):
        x = tf.convert_to_tensor(X_test, dtype=TF_DTYPE)
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
        if self._legacy_reference is not None:
            self._legacy_reference.close()
            self._legacy_reference = None


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
