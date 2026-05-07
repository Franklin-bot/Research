import math
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

TF_DTYPE = tf.float64
LOG_SIGMA_EPS = 1e-4


def xavier_limit(fan_in, fan_out, constant=1.0):
    return constant * np.sqrt(1.0 / (fan_in + fan_out))


class TF1StyleAdam(tf.Module):
    """Small TF2 optimizer that mirrors TF1 Adam update semantics."""

    def __init__(
        self,
        variables,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
    ):
        super().__init__(name="tf1_style_adam")
        self.learning_rate = tf.constant(learning_rate, dtype=TF_DTYPE)
        self.beta_1 = tf.constant(beta_1, dtype=TF_DTYPE)
        self.beta_2 = tf.constant(beta_2, dtype=TF_DTYPE)
        self.epsilon = tf.constant(epsilon, dtype=TF_DTYPE)
        self.one = tf.constant(1.0, dtype=TF_DTYPE)

        self.beta1_power = tf.Variable(
            beta_1,
            dtype=TF_DTYPE,
            trainable=False,
            name="beta1_power",
        )
        self.beta2_power = tf.Variable(
            beta_2,
            dtype=TF_DTYPE,
            trainable=False,
            name="beta2_power",
        )
        self.m_slots = []
        self.v_slots = []
        for index, variable in enumerate(variables):
            self.m_slots.append(
                tf.Variable(
                    tf.zeros_like(variable),
                    dtype=TF_DTYPE,
                    trainable=False,
                    name=f"m_{index}",
                )
            )
            self.v_slots.append(
                tf.Variable(
                    tf.zeros_like(variable),
                    dtype=TF_DTYPE,
                    trainable=False,
                    name=f"v_{index}",
                )
            )

    def apply_gradients(self, gradients, variables):
        lr_t = self.learning_rate * tf.sqrt(self.one - self.beta2_power) / (
            self.one - self.beta1_power
        )

        for gradient, variable, m_slot, v_slot in zip(
            gradients,
            variables,
            self.m_slots,
            self.v_slots,
        ):
            if gradient is None:
                continue

            gradient = tf.cast(gradient, TF_DTYPE)
            new_m = m_slot.assign(
                self.beta_1 * m_slot + (self.one - self.beta_1) * gradient
            )
            new_v = v_slot.assign(
                self.beta_2 * v_slot
                + (self.one - self.beta_2) * tf.square(gradient)
            )
            variable.assign_sub(
                lr_t * new_m / (tf.sqrt(new_v) + self.epsilon)
            )

        self.beta1_power.assign(self.beta1_power * self.beta_1)
        self.beta2_power.assign(self.beta2_power * self.beta_2)


class VariationalAutoencoder(tf.Module):
    """Native TF2 VAE with manual variables and TF-owned randomness."""

    def __init__(
        self,
        network_architecture,
        transfer_fct=tf.nn.relu,
        learning_rate=0.001,
        batch_size=100,
        vae_mode=False,
        vae_mode_modalities=False,
        seed=None,
        strategy=None,
    ):
        super().__init__(name="variational_autoencoder")
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.vae_mode = vae_mode
        self.vae_mode_modalities = vae_mode_modalities
        self.seed = None if seed is None else int(seed)
        self.strategy = strategy
        self.num_replicas_in_sync = (
            strategy.num_replicas_in_sync if strategy is not None else 1
        )
        self._step_seed_counter = 0

        self.n_mc = 4
        self.n_vis = 4

        self.n_input = network_architecture["n_input"]
        self.n_z = network_architecture["n_z"]
        self.size_slices = list(network_architecture["size_slices"])
        self.size_slices_shared = list(network_architecture["size_slices_shared"])

        if self.seed is None:
            self._rng = tf.random.Generator.from_non_deterministic_state()
        else:
            self._rng = tf.random.Generator.from_seed(self.seed)

        self._trainable_vars = []

        self.mod0_kernels, self.mod0_biases = self._create_dense_stack_vars(
            self.size_slices[0],
            network_architecture["mod0"],
            "mod0",
        )
        self.mod1_kernels, self.mod1_biases = self._create_dense_stack_vars(
            self.size_slices[1],
            network_architecture["mod1"],
            "mod1",
        )
        self.enc_shared_kernels, self.enc_shared_biases = self._create_dense_stack_vars(
            network_architecture["mod0"][-1] + network_architecture["mod1"][-1],
            network_architecture["enc_shared"],
            "enc_shared",
        )
        self.z_params = self._create_variational_vars(
            network_architecture["enc_shared"][-1],
            self.n_z,
            "z",
        )
        self.dec_shared_kernels, self.dec_shared_biases = self._create_dense_stack_vars(
            self.n_z,
            network_architecture["dec_shared"],
            "dec_shared",
        )
        self.mod0_2_kernels, self.mod0_2_biases = self._create_dense_stack_vars(
            self.size_slices_shared[0],
            network_architecture["mod0_2"],
            "mod0_2",
        )
        self.mod1_2_kernels, self.mod1_2_biases = self._create_dense_stack_vars(
            self.size_slices_shared[1],
            network_architecture["mod1_2"],
            "mod1_2",
        )
        self.out0_params = self._create_variational_vars(
            network_architecture["mod0_2"][-1],
            self.size_slices[0],
            "out0",
        )
        self.out1_params = self._create_variational_vars(
            network_architecture["mod1_2"][-1],
            self.size_slices[1],
            "out1",
        )

        self.optimizer = TF1StyleAdam(
            self.trainable_variables,
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        )

    @property
    def trainable_variables(self):
        return self._trainable_vars

    def _random_uniform(self, shape, minval, maxval):
        return self._rng.uniform(
            shape=shape,
            minval=tf.cast(minval, TF_DTYPE),
            maxval=tf.cast(maxval, TF_DTYPE),
            dtype=TF_DTYPE,
        )

    def _random_normal(self, shape):
        return self._rng.normal(
            shape=shape,
            mean=0.0,
            stddev=1.0,
            dtype=TF_DTYPE,
        )

    def _next_step_seed(self):
        if self.seed is None:
            return None
        seed_0 = int(self.seed % (2**31 - 1))
        seed_1 = int(self._step_seed_counter % (2**31 - 1))
        self._step_seed_counter += 1
        return tf.convert_to_tensor([seed_0, seed_1], dtype=tf.int32)

    def _sample_training_epsilon(self, shape, step_seed=None):
        if step_seed is None:
            return self._random_normal(shape)

        replica_context = tf.distribute.get_replica_context()
        replica_id = tf.constant(0, dtype=tf.int32)
        if replica_context is not None:
            replica_id = tf.cast(
                replica_context.replica_id_in_sync_group,
                tf.int32,
            )

        stateless_seed = tf.stack(
            [
                step_seed[0],
                step_seed[1] * tf.cast(self.num_replicas_in_sync, tf.int32)
                + replica_id,
            ]
        )
        return tf.random.stateless_normal(
            shape,
            seed=stateless_seed,
            dtype=TF_DTYPE,
        )

    def _create_variable(self, shape, name, is_bias=False):
        if is_bias:
            init_value = tf.zeros(shape, dtype=TF_DTYPE)
        else:
            fan_in, fan_out = shape
            limit = xavier_limit(fan_in, fan_out)
            init_value = self._random_uniform(shape, -limit, limit)

        variable = tf.Variable(
            init_value,
            dtype=TF_DTYPE,
            trainable=True,
            name=name,
        )
        self._trainable_vars.append(variable)
        return variable

    def _create_dense_stack_vars(self, input_dim, layer_sizes, prefix):
        kernels = []
        biases = []
        in_dim = input_dim
        for index, layer_size in enumerate(layer_sizes):
            kernels.append(
                self._create_variable(
                    (in_dim, layer_size),
                    f"{prefix}_kernel_{index}",
                )
            )
            biases.append(
                self._create_variable(
                    (layer_size,),
                    f"{prefix}_bias_{index}",
                    is_bias=True,
                )
            )
            in_dim = layer_size
        return kernels, biases

    def _create_variational_vars(self, input_dim, latent_size, prefix):
        mean_kernel = self._create_variable(
            (input_dim, latent_size),
            f"{prefix}_mean_kernel",
        )
        log_kernel = self._create_variable(
            (input_dim, latent_size),
            f"{prefix}_log_kernel",
        )
        mean_bias = self._create_variable(
            (latent_size,),
            f"{prefix}_mean_bias",
            is_bias=True,
        )
        log_bias = self._create_variable(
            (latent_size,),
            f"{prefix}_log_bias",
            is_bias=True,
        )
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
        log_sigma_sq = tf.math.log(
            tf.exp(raw_log_sigma) + tf.cast(LOG_SIGMA_EPS, TF_DTYPE)
        )
        return mean, log_sigma_sq

    def _decode_from_latent(self, z):
        dec_shared = self._forward_dense_stack(
            z,
            self.dec_shared_kernels,
            self.dec_shared_biases,
        )
        mod0_shared, mod1_shared = self._slice_input(
            dec_shared,
            self.size_slices_shared,
        )

        mod0_decoded = self._forward_dense_stack(
            mod0_shared,
            self.mod0_2_kernels,
            self.mod0_2_biases,
        )
        mod1_decoded = self._forward_dense_stack(
            mod1_shared,
            self.mod1_2_kernels,
            self.mod1_2_biases,
        )

        mod0_mean, mod0_log_sigma_sq = self._forward_variational(
            mod0_decoded,
            self.out0_params,
        )
        mod1_mean, mod1_log_sigma_sq = self._forward_variational(
            mod1_decoded,
            self.out1_params,
        )

        return {
            "final_means": [mod0_mean, mod1_mean],
            "final_sigmas": [mod0_log_sigma_sq, mod1_log_sigma_sq],
            "x_reconstr": tf.concat([mod0_mean, mod1_mean], axis=1),
            "x_log_sigma_sq": tf.concat(
                [mod0_log_sigma_sq, mod1_log_sigma_sq],
                axis=1,
            ),
        }

    def _forward(self, x, training=False, epsilon=None, step_seed=None):
        mod0_input, mod1_input = self._slice_input(x, self.size_slices)
        mod0_output = self._forward_dense_stack(
            mod0_input,
            self.mod0_kernels,
            self.mod0_biases,
        )
        mod1_output = self._forward_dense_stack(
            mod1_input,
            self.mod1_kernels,
            self.mod1_biases,
        )

        output_mod = tf.concat([mod0_output, mod1_output], axis=1)
        enc_shared = self._forward_dense_stack(
            output_mod,
            self.enc_shared_kernels,
            self.enc_shared_biases,
        )
        z_mean, z_log_sigma_sq = self._forward_variational(
            enc_shared,
            self.z_params,
        )

        if training and self.vae_mode:
            if epsilon is None:
                epsilon = self._sample_training_epsilon(
                    tf.shape(z_mean),
                    step_seed=step_seed,
                )
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

    def compute_losses(
        self,
        x,
        x_noiseless,
        epoch,
        training,
        epsilon=None,
        step_seed=None,
    ):
        outputs = self._forward(
            x,
            training=training,
            epsilon=epsilon,
            step_seed=step_seed,
        )
        x_noiseless_sliced = self._slice_input(x_noiseless, self.size_slices)

        epoch = tf.cast(epoch, TF_DTYPE)
        one = tf.cast(1.0, TF_DTYPE)
        alpha = one - tf.minimum(
            epoch / tf.cast(1000.0, TF_DTYPE),
            one,
        )

        tmp_costs = []
        for index, (target_slice, mean, log_sigma_sq) in enumerate(
            zip(
                x_noiseless_sliced,
                outputs["final_means"],
                outputs["final_sigmas"],
            )
        ):
            reconstr_loss = (
                0.5
                * tf.reduce_sum(
                    tf.square(target_slice - mean) / tf.exp(log_sigma_sq),
                    axis=1,
                )
                + 0.5 * tf.reduce_sum(log_sigma_sq, axis=1)
                + 0.5 * self.n_z / 2 * np.log(2 * math.pi)
            ) / self.network_architecture["size_slices"][index]
            tmp_costs.append(reconstr_loss)

        reconstr_loss = tf.reduce_mean(tmp_costs[0] + tmp_costs[1])
        latent_loss = -0.5 * tf.reduce_sum(
            1
            + outputs["z_log_sigma_sq"]
            - tf.square(outputs["z_mean"])
            - tf.exp(outputs["z_log_sigma_sq"]),
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
    def _train_step(self, x, x_noiseless, epoch, step_seed):
        with tf.GradientTape() as tape:
            losses = self.compute_losses(
                x,
                x_noiseless,
                epoch,
                training=True,
                step_seed=step_seed,
            )
        gradients = tape.gradient(losses["cost"], self.trainable_variables)
        self.optimizer.apply_gradients(gradients, self.trainable_variables)
        return losses

    @tf.function(reduce_retracing=True)
    def _distributed_train_step(self, distributed_batch, epoch, step_seed):
        def replica_step(replica_batch):
            if tf.shape(replica_batch)[1] == self.n_input * 2:
                batch_xs = replica_batch[:, : self.n_input]
                batch_xs_noiseless = replica_batch[:, self.n_input :]
            else:
                batch_xs = replica_batch
                batch_xs_noiseless = replica_batch

            with tf.GradientTape() as tape:
                losses = self.compute_losses(
                    batch_xs,
                    batch_xs_noiseless,
                    epoch,
                    training=True,
                    step_seed=step_seed,
                )

            gradients = tape.gradient(losses["cost"], self.trainable_variables)
            replica_context = tf.distribute.get_replica_context()
            gradients = [
                replica_context.all_reduce(tf.distribute.ReduceOp.MEAN, gradient)
                if gradient is not None
                else None
                for gradient in gradients
            ]
            self.optimizer.apply_gradients(gradients, self.trainable_variables)
            return {
                "cost": losses["cost"],
                "recon": losses["recon"],
                "latent": losses["latent"],
                "alpha": losses["alpha"],
            }

        per_replica_losses = self.strategy.run(replica_step, args=(distributed_batch,))
        return {
            name: self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                value,
                axis=None,
            )
            for name, value in per_replica_losses.items()
        }

    def partial_fit(self, X, X_noiseless, epoch):
        x = tf.convert_to_tensor(X, dtype=TF_DTYPE)
        x_noiseless = tf.convert_to_tensor(X_noiseless, dtype=TF_DTYPE)
        epoch = tf.convert_to_tensor(epoch, dtype=TF_DTYPE)
        step_seed = self._next_step_seed()
        losses = self._train_step(x, x_noiseless, epoch, step_seed)
        return (
            float(losses["cost"].numpy()),
            float(losses["recon"].numpy()),
            float(losses["latent"].numpy()),
            losses["x_reconstr"].numpy(),
            float(losses["alpha"].numpy()),
        )

    def distributed_partial_fit(self, distributed_batch, epoch):
        epoch = tf.convert_to_tensor(epoch, dtype=TF_DTYPE)
        step_seed = self._next_step_seed()
        losses = self._distributed_train_step(distributed_batch, epoch, step_seed)
        return (
            float(losses["cost"].numpy()),
            float(losses["recon"].numpy()),
            float(losses["latent"].numpy()),
            None,
            float(losses["alpha"].numpy()),
        )

    def transform(self, X):
        x = tf.convert_to_tensor(X, dtype=TF_DTYPE)
        outputs = self.compute_losses(x, x, epoch=1000.0, training=False)
        return outputs["z_mean"].numpy()

    def generate(self, z_mu=None):
        if z_mu is None:
            z = self._random_normal((1, self.n_z))
        else:
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
    use_distributed = vae.strategy is not None and vae.num_replicas_in_sync > 1

    if use_distributed and batch_size % vae.num_replicas_in_sync != 0:
        raise ValueError(
            "Global batch_size must be divisible by the number of replicas. "
            f"Got batch_size={batch_size}, replicas={vae.num_replicas_in_sync}."
        )

    for epoch in tqdm(range(training_epochs)):
        avg_cost = 0.0
        avg_recon = 0.0
        avg_latent = 0.0
        total_batch = int(n_samples / batch_size)

        X_shuffled = shuffle_data(input_data)

        if use_distributed:
            trimmed = np.asarray(X_shuffled[: total_batch * batch_size], dtype=np.float64)
            dataset = tf.data.Dataset.from_tensor_slices(trimmed).batch(
                batch_size,
                drop_remainder=True,
            )
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            distributed_dataset = vae.strategy.experimental_distribute_dataset(dataset)

            for batch_xs_augmented in distributed_dataset:
                cost, recon, latent, x_rec, alpha = vae.distributed_partial_fit(
                    batch_xs_augmented,
                    epoch,
                )
                avg_cost += cost / n_samples * batch_size
                avg_recon += recon / n_samples * batch_size
                avg_latent += latent / n_samples * batch_size
        else:
            for i in range(total_batch):
                batch_xs_augmented = X_shuffled[
                    batch_size * i : batch_size * i + batch_size
                ]
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
