import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment

from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops

class PrunedConv(tf.keras.layers.Layer):
    def __init__(self, config, weights, **kwargs):
        super(PrunedConv, self).__init__(**kwargs)
        self.strides = config["strides"]
        self.kernel_size = config["kernel_size"]
        self.padding = config["padding"]
        k, b, m = weights
        self.mask = m
        self.bias = tf.Variable(b, dtype=tf.float32, trainable=True)
        self.depth_multiplier = int(np.sum(m[0, 0, 0]))
        print("Number of outputs per channel", self.depth_multiplier)
        if not np.all(np.sum(m, axis=-1) == self.depth_multiplier):
            raise ValueError
        dw_kernel = np.zeros((config["kernel_size"][0], config["kernel_size"][1], k.shape[2], self.depth_multiplier))
        for i in range(m.shape[2]):
            l = 0
            for j in range(m.shape[3]):
                if m[0, 0, i, j] == 0:
                    continue
                dw_kernel[:, :, i, l] = k[:, :, i, j]
                l += 1
        self.kernel = tf.Variable(dw_kernel, trainable=True, dtype=tf.float32)
        self.strides = (1,) + config["strides"] + (1,)
        self.kernel_size = config["kernel_size"]
        self.padding = "SAME" if config["padding"] == "same" else "VALID"
        self.activation = config["activation"]

    def call(self, x):
        
        x = tf.nn.depthwise_conv2d(x, filter=self.kernel, strides=self.strides, padding=self.padding, data_format="NHWC")
        
        s = tf.shape(x)
        batch_size = s[0]
        height = s[1]
        width = s[2]

        scatter_indices = [0 for _ in range(self.mask.shape[2]*self.depth_multiplier)]
        counts = [ 0 for _ in range(self.mask.shape[2])]
        for i in range(self.mask.shape[2]):
            for j in range(self.mask.shape[3]):
                if self.mask[0, 0, i, j] == 1:
                    scatter_indices[i * self.depth_multiplier + counts[i]] = j
                    counts[i] += 1
               
        a, b, c, d = tf.meshgrid(tf.range(batch_size), tf.range(height), tf.range(width), scatter_indices, indexing="ij")
        indices = tf.stack([a, b, c, d], axis=-1) # has shape (batch_size, height, width, self.mask.shape[2]*self.depth_multiplier, 4)
        shape = [batch_size, height, width, self.mask.shape[3]] # shape of the output
        output = tf.scatter_nd(indices, x, shape)
        output = tf.nn.bias_add(output, self.bias)
        if self.activation is not None:
            output = getattr(tf.nn, self.activation)(output)

        return output

        # The version below, which "manually" adds the outputs, is slower
        outputs = [ tf.zeros(tf.shape(x)[:-1]) for _ in range(self.mask.shape[3]) ]
        counts = [ 0 for _ in range(self.mask.shape[2])]
        for j in range(self.mask.shape[3]):
            for i in range(self.mask.shape[2]):
                if self.mask[0, 0, i, j] == 1:
                    print("node", self.mask.shape[3], j, i)
                    outputs[j] += x[:, :, :, i * self.depth_multiplier + counts[i]]
                    counts[i] += 1
            continue
        output = tf.stack(outputs, axis=-1)
        output = tf.nn.bias_add(output, self.bias)
        if self.activation is not None:
            output = getattr(tf.nn, self.activation)(output)
        return output

class PrunableConv(tf.keras.layers.Conv2D):
    def __init__(self, auto_prune_settings, *args, binary_init=False, **kwargs):
        super(PrunableConv, self).__init__(*args, **kwargs)
        self.auto_prune_settings = auto_prune_settings
        self.binary_init = binary_init
        if auto_prune_settings is not None:
            final_value = auto_prune_settings["final_value"]
            initial_value = auto_prune_settings["initial_value"]
            num_steps = auto_prune_settings["num_steps"]
            schedule_type = auto_prune_settings.get("schedule_type", "linear")
            if schedule_type == "linear":
                schedule = [0 for _ in range(num_steps)]
                i = 0
                while sum(schedule) < initial_value - final_value:
                    schedule[i%len(schedule)] += 1
                    i += 1
            if schedule_type == "geometric":
                alpha = (float(final_value)/initial_value)**(1/(num_steps))
                schedule = []
                for t in range(num_steps):
                    schedule.append( round((1-alpha) * alpha**t * initial_value) )
                i = 0
                while sum(schedule) < initial_value - final_value: # The sum can be different because of rounding
                    schedule[i%len(schedule)] += 1
                    i += 1
                i = len(schedule)-1
                while sum(schedule) > initial_value - final_value: # The sum can be different because of rounding
                    schedule[i%len(schedule)] -= 1
                    i -= 1
            self.pruning_schedule = schedule
            self.current_pruning_step = 0
            self.pruning_type = auto_prune_settings["pruning_type"]
            self.pruning_criteria = auto_prune_settings.get("criteria", "magnitude")
            print(self.name, "pruning schedule", schedule)
    def build(self, input_shape):
        super(PrunableConv, self).build(input_shape)
        self.initial_weights = self.get_weights()

        if self.binary_init:
            w = self.initial_weights[0]
            other = self.initial_weights[1:]
            std = np.std(w)
            w[w<0] = -std
            w[w>0] = std
            self.set_weights([w] + other)

        self.is_resettable = True
        self.is_prunable = True
        self.mask = self.add_weight(shape=self.kernel.shape, trainable=False, name=self.name + "_mask", dtype=tf.float32, initializer="ones")
    
    def call(self, inputs):
        k = self.kernel * self.mask
        # Check if the input_shape in call() is different from that in build().
        # If they are different, recreate the _convolution_op to avoid the stateful
        # behavior.
        call_input_shape = inputs.get_shape()
        recreate_conv_op = (call_input_shape[1:] != self._build_conv_op_input_shape[1:])

        if recreate_conv_op:
            self._convolution_op = nn_ops.Convolution( call_input_shape, filter_shape=self.kernel.shape, dilation_rate=self.dilation_rate, strides=self.strides, padding=self._padding_op, data_format=self._conv_op_data_format)

        # Apply causal padding to inputs for Conv1D.
        if self.padding == 'causal' and self.__class__.__name__ == 'Conv1D':
            inputs = array_ops.pad(inputs, self._compute_causal_padding())

        outputs = self._convolution_op(inputs, k)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
    def get_mask(self):
        return self.get_weights()[-1]
    def reset_mask_to_ones(self):
        self.set_weights(self.get_weights() + [np.ones_like(self.get_mask())])
    def reset_weights_to_init(self):
        self.set_weights(self.initial_weights + [self.get_mask()])
    def reset_weights_to_random_init(self):
        self.set_weights([np.random.permutation(w.flat).reshape(w.shape) for w in self.initial_weights] + [self.get_mask()])
    def prune_n_outputs(self, n=1, criteria="magnitude"):
        w = self.get_weights()
        k = w[0]
        m = w[-1]
        other = w[1:-1]
        if criteria == "min_matching":
            for i in range(n):
                cost = np.sum(np.sum(np.abs(k), axis=0), axis=0)
                cost[m[0, 0, :, :] == 0] = 1000
                row_ind, col_ind = linear_sum_assignment(cost, maximize=False)
                np.put_along_axis(m, col_ind[None, None, :, None], 0, axis=-1)
        else:
            if criteria == "magnitude":
                crit = np.sum(np.sum(np.abs(k), axis=0), axis=0)
            elif criteria == "random":
                crit = np.random.rand(m.shape[2], m.shape[3])
            crit[m[0, 0, :, :] == 0] = float("inf")
            ind = np.argpartition(crit, kth=n, axis=1)[:, :n]
            np.put_along_axis(m, ind[None, None], 0, axis=-1)
        self.set_weights([k] + other + [m])
        print(self.name, "sparsity is now", 1 - np.sum(m)/m.size)
    def prune_n_connexions(self, n, criteria="magnitude"):
        w = self.get_weights()
        k = w[0]
        m = w[-1]
        other = w[1:-1]
        if criteria == "magnitude":
            crit = np.sum(np.sum(np.abs(k), axis=0), axis=0)
        elif criteria == "random":
            crit = np.random.rand(m.shape[2], m.shape[3])
        crit[m[0, 0, :, :] == 0] = float("inf")
        m_flat = np.reshape(m, (m.shape[0], m.shape[1], -1,))
        m_flat[:, :, np.argpartition(crit.flat, kth=n)[:n]] = 0
        m = m_flat.reshape(m.shape)
        self.set_weights([k] + other + [m])
        print(self.name, "sparsity is now", 1 - np.sum(m)/m.size)
    def prune_n_params(self, n, criteria="magnitude"):
        w = self.get_weights()
        k = w[0]
        m = w[-1]
        other = w[1:-1]
        if criteria == "magnitude":
            crit = np.abs(k)
        elif criteria == "random":
            crit = np.random.rand(m.shape[0], m.shape[1], m.shape[2], m.shape[3])
        crit[m == 0] = float("inf")
        m_flat = np.reshape(m, (-1,))
        m_flat[np.argpartition(crit.flat, kth=n)[:n]] = 0
        m = m_flat.reshape(m.shape)
        self.set_weights([k] + other + [m])
        print(self.name, "sparsity is now", 1 - np.sum(m)/m.size)
    def prune_n_depthwise(self, n, criteria="magnitude"):
        w = self.get_weights()
        k = w[0]
        m = w[-1]
        other = w[1:-1]
        if criteria == "magnitude":
            crit = np.sum(np.sum(np.abs(k), axis=-1), axis=-1)
        elif criteria == "random":
            crit = np.random.rand(m.shape[0], m.shape[1])
        crit[m[:, :, 0, 0] == 0] = float("inf")
        m_flat = np.reshape(m, (-1, m.shape[2], m.shape[3]))
        m_flat[np.argpartition(crit.flat, kth=n)[:n], :, :] = 0
        m = m_flat.reshape(m.shape)
        self.set_weights([k] + other + [m])
        print(self.name, "sparsity is now", 1 - np.sum(m)/m.size)
    def get_pruned_layer(self):
        pruned_conv = PrunedConv(self.get_config(), self.get_weights())
        return pruned_conv
    def pruning_step(self):
        if self.auto_prune_settings is None: return
        if self.current_pruning_step < len(self.pruning_schedule):
            if self.pruning_schedule[self.current_pruning_step] > 0:
                if self.pruning_type == "outputs":
                    self.prune_n_outputs(self.pruning_schedule[self.current_pruning_step], criteria=self.pruning_criteria)
                if self.pruning_type == "connexions":
                    self.prune_n_connexions(self.pruning_schedule[self.current_pruning_step], criteria=self.pruning_criteria)
                if self.pruning_type == "depthwise":
                    self.prune_n_depthwise(self.pruning_schedule[self.current_pruning_step], criteria=self.pruning_criteria)
                if self.pruning_type == "params":
                    self.prune_n_params(self.pruning_schedule[self.current_pruning_step], criteria=self.pruning_criteria)
            print(self.name, "pruning", self.pruning_schedule[self.current_pruning_step], "outputs per channel")
        self.current_pruning_step += 1

class ResettableBN(tf.keras.layers.BatchNormalization):
    def build(self, input_shape):
        super(ResettableBN, self).build(input_shape)
        self.initial_weights = self.get_weights()
        self.is_resettable = True
    def reset_weights_to_init(self):
        self.set_weights(self.initial_weights)
    def reset_weights_to_random_init(self):
        self.set_weights([np.random.permutation(w.flat).reshape(w.shape) for w in self.initial_weights])

class ResettableConv(tf.keras.layers.Conv2D):
    def build(self, input_shape):
        super(ResettableConv, self).build(input_shape)
        self.initial_weights = self.get_weights()
        self.is_resettable = True
    def reset_weights_to_init(self):
        self.set_weights(self.initial_weights)
    def reset_weights_to_random_init(self):
        self.set_weights([np.random.permutation(w.flat).reshape(w.shape) for w in self.initial_weights])

class ResettableDense(tf.keras.layers.Dense):
    def build(self, input_shape):
        super(ResettableDense, self).build(input_shape)
        self.initial_weights = self.get_weights()
        self.is_resettable = True
    def reset_weights_to_init(self):
        self.set_weights(self.initial_weights)
    def reset_weights_to_random_init(self):
        self.set_weights([np.random.permutation(w.flat).reshape(w.shape) for w in self.initial_weights])