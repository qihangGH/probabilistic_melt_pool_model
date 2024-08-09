import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from functools import reduce

tfk = tf.keras
tfkl = tfk.layers
tfd = tfp.distributions
tfpl = tfp.layers


class DenseVar(tfpl.DenseVariational):
    def call(self, inputs):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        inputs = tf.cast(inputs, dtype, name='inputs')

        q = self._posterior(inputs)
        r = self._prior(inputs)
        self.add_loss(self._kl_divergence_fn(q, r))

        w = tf.convert_to_tensor(value=q)
        self.prior_log_prob = r.log_prob(w)
        self.posterior_log_prob = q.log_prob(w)

        prev_units = self.input_spec.axes[-1]
        if self.use_bias:
            split_sizes = [prev_units * self.units, self.units]
            kernel, bias = tf.split(w, split_sizes, axis=-1)
        else:
            kernel, bias = w, None

        kernel = tf.reshape(kernel, shape=tf.concat([
            tf.shape(kernel)[:-1],
            [prev_units, self.units],
        ], axis=0))
        outputs = tf.matmul(inputs, kernel)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs


class Conv2DVar(tfpl.Convolution2DReparameterization):
    def _apply_variational_kernel(self, inputs):
        outputs = super()._apply_variational_kernel(inputs)
        self.prior_log_prob = self.kernel_prior.log_prob(self.kernel_posterior_tensor)
        self.posterior_log_prob = self.kernel_posterior.log_prob(self.kernel_posterior_tensor)
        return outputs


Convolution2DReparameterization = Conv2DVar
DenseVariational = DenseVar
# Convolution2DReparameterization = tfpl.Convolution2DReparameterization
# DenseVariational = tfpl.DenseVariational


class ConvBlock(tfk.Model):
    def __init__(
            self,
            is_bayesian,
            filters,
            activation,
            kl_weight=None,
            dropout_rate=None,
            weight_decay=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.is_bayesian = is_bayesian
        self.dropout_rate = dropout_rate
        if is_bayesian:
            assert dropout_rate is not None or kl_weight is not None, \
                'Please specify `kl_weight` or `dropout_rate` for Bayesian layers.'
            if dropout_rate is None:
                divergence_fn = lambda q, p, ignore: tfd.kl_divergence(q, p) * kl_weight
                self.conv = Convolution2DReparameterization(
                    filters, kernel_size=3, padding='same',
                    kernel_prior_fn=trainable_prior_fn,
                    kernel_divergence_fn=divergence_fn
                )
            else:
                # assert weight_decay is not None, 'please specify `weight_decay` for MC dropout'
                self.conv = tfkl.Conv2D(filters, kernel_size=3, padding='same',
                                        activation=activation, kernel_regularizer=weight_decay)
                self.mc_dropout = MCDropout(rate=dropout_rate)
        else:
            self.conv = tfkl.Conv2D(filters, kernel_size=3, padding='same', kernel_regularizer=weight_decay)
        self.batch_norm = tfkl.BatchNormalization(axis=3)
        self.activation = tfkl.Activation(activation)

    def call(self, x):
        if self.is_bayesian is True and self.dropout_rate is not None:
            return self.mc_dropout(self.conv(x))
        return self.activation(self.batch_norm(self.conv(x)))


class DenseBlock(tfk.Model):
    def __init__(
            self,
            is_bayesian,
            units,
            activation,
            kl_weight=None,
            dropout_rate=None,
            weight_decay=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.is_bayesian = is_bayesian
        self.dropout_rate = dropout_rate
        self.dense = tfkl.Dense(units, kernel_regularizer=weight_decay)
        if is_bayesian:
            assert dropout_rate is not None or kl_weight is not None, \
                'Please specify `kl_weight` or `dropout_rate` for Bayesian layers.'
            if dropout_rate is None:
                self.dense = DenseVariational(
                    units, make_posterior_fn=posterior_mean_field,
                    make_prior_fn=prior_trainable,
                    kl_weight=kl_weight,
                    kl_use_exact=True
                )
            else:
                self.mc_dropout = MCDropout(rate=dropout_rate)
                self.dense = tfkl.Dense(units, kernel_regularizer=weight_decay)
        self.batch_norm = tfkl.BatchNormalization()
        self.activation = tfkl.Activation(activation)

    def call(self, x):
        if self.is_bayesian is True and self.dropout_rate is not None:
            return self.mc_dropout(self.activation(self.dense(x)))
        return self.activation(self.batch_norm(self.dense(x)))


class MCDropout(tfkl.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


class MPSigRegressor(tfk.Model):
    def __init__(
            self,
            is_bayesian,
            blocks_feature_extractor,
            filters_feature_extractor,
            units_before_concat,
            units_after_concat,
            activation,
            kl_weight=None,
            output_dist=None,
            scales=1,
            pv_units=None,
            fusion_opt='concat',
            dropout_rate=None,
            preprocess_pvt_mode=None,
            weight_decay=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.is_bayesian = is_bayesian
        if is_bayesian:
            assert dropout_rate is not None or kl_weight is not None, \
                'Please specify `kl_weight` or `dropout_rate` for Bayesian layers.'
            assert output_dist is not None, \
                'Please specify an output distribution to `output_dist` as `gaussian` or `student`.'
            if output_dist == 'gaussian':
                transform_fn = transform_to_Gaussian
            elif output_dist == 'student':
                transform_fn = transform_to_StudentT
            else:
                raise ValueError('`output_dist` should be `gaussian` or `student`')
        # `self.param_regressor` must be placed before `self.out`
        # because model.layers[-2] is used in training
        self.param_regressor = ParamRegressor(
            is_bayesian,
            blocks_feature_extractor,
            filters_feature_extractor,
            units_before_concat,
            units_after_concat,
            activation,
            kl_weight,
            output_dist,
            scales,
            pv_units,
            fusion_opt,
            dropout_rate,
            preprocess_pvt_mode,
            weight_decay,
            **kwargs
        )
        if is_bayesian:
            self.out = tfpl.DistributionLambda(transform_fn)

    def call(self, inputs, training=None, mask=None):
        params = self.param_regressor(inputs)
        if self.is_bayesian is True:
            params = self.out(params)
        return params


class NeighborFeatureExtractor(tfk.Model):
    def __init__(
            self,
            is_bayesian,
            num_blocks,
            filters,
            activation,
            kl_weight=None,
            dropout_rate=None,
            weight_decay=None,
            **kwargs
    ):
        """
        Preprocessing of p, v, t, data
        :param is_bayesian:
        :param num_blocks:
        :param filters:
        :param activation:
        :param kl_weight:
        :param weight_pv:
        :param dropout_rate:
        :param kwargs:
        """
        super().__init__(**kwargs)
        if is_bayesian:
            assert dropout_rate is not None or kl_weight is not None, \
                'Please specify `kl_weight` or `dropout_rate` for Bayesian layers.'
        self.conv_blocks = [
            [ConvBlock(is_bayesian, i * filters, activation, kl_weight, dropout_rate, weight_decay) for _ in range(2)]
            for i in [2 ** j for j in range(num_blocks)]]
        self.conv_block1 = ConvBlock(is_bayesian, filters, activation, kl_weight, dropout_rate, weight_decay)
        self.max_pooling = [tfkl.MaxPooling2D(pool_size=(2, 2)) for _ in range(num_blocks)]

    def call(self, x):
        for blocks, max_pooling in zip(self.conv_blocks, self.max_pooling):
            for block in blocks:
                x = block(x)
            x = max_pooling(x)
        x = self.conv_block1(x)
        return x


class NToWeightMap(tfkl.Layer):
    def __init__(self, blocks, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv_batch_norm_blocks = [
            ConvBlock(False, i * filters, 'relu') for _ in range(2)
            for i in [2 ** j for j in range(blocks)]
        ]


class ParamRegressor(tfk.Model):
    def __init__(
            self,
            is_bayesian,
            blocks_feature_extractor,
            filters_feature_extractor,
            units_before_concat,
            units_after_concat,
            activation,
            kl_weight=None,
            output_dist=None,
            scales=1,
            pv_units=None,
            fusion_opt='concat',
            dropout_rate=None,
            preprocess_pvt_mode=None,
            weight_decay=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.is_bayesian = is_bayesian
        self.units_after_concat = units_after_concat
        self.scales = scales
        self.fusion_opt = fusion_opt
        assert fusion_opt in ['add', 'concat'], 'fusion option {} is not defined'.format(fusion_opt)
        if fusion_opt == 'add':
            assert pv_units is not None, \
                'In `add` fusion mode, please specify `pv_units` to extract p, v data\'s features'
        if is_bayesian:
            assert dropout_rate is not None or kl_weight is not None, \
                'Please specify `kl_weight` or `dropout_rate` for Bayesian layers.'
            assert output_dist is not None, \
                'Please specify an output distribution to `output_dist` as `gaussian` or `student`.'
        # preprocess pvt patches
        self.preprocess_pvt = PreprocessNeighPVT(mode=preprocess_pvt_mode)
        # extract the features of pvt patches
        self.neighbor_feature_extractor = NeighborFeatureExtractor(
            is_bayesian,
            blocks_feature_extractor,
            filters_feature_extractor,
            activation,
            kl_weight,
            dropout_rate,
            weight_decay
        )
        # extract the features of p, v data
        self.pv_units = pv_units
        if pv_units is not None:
            self.pv_feature_extractor = PVFeatureExtrator(is_bayesian, pv_units, activation,
                                                          kl_weight, dropout_rate, output_dist, weight_decay)
        # fusion option and output
        if is_bayesian:
            if output_dist == 'gaussian':
                self.num_params = 2
                self.calc_dist_params = tfkl.Lambda(self.calc_gaussian_params)
            elif output_dist == 'student':
                self.num_params = 4
                self.calc_dist_params = tfkl.Lambda(self.calc_stu_params)
            else:
                raise ValueError('`output_dist` should be `gaussian` or `student`')
            if dropout_rate is None:
                if self.fusion_opt == 'concat':
                    self.concat_feature_to_params = DenseVariational(
                        self.num_params, make_posterior_fn=posterior_mean_field,
                        make_prior_fn=prior_trainable,
                        kl_weight=kl_weight,
                        kl_use_exact=True,
                        activation='relu'
                    )
                elif self.fusion_opt == 'add':
                    self.dense_before_add = [
                        DenseVariational(
                            self.num_params, make_posterior_fn=posterior_mean_field,
                            make_prior_fn=prior_trainable,
                            kl_weight=kl_weight,
                            kl_use_exact=True,
                            activation='relu'
                        ) for _ in range(self.scales)
                    ]
            else:
                if self.fusion_opt == 'concat':
                    self.concat_feature_to_params = tfkl.Dense(self.num_params, activation='relu',
                                                               kernel_regularizer=weight_decay)
                elif self.fusion_opt == 'add':
                    self.dense_before_add = [
                        DenseBlock(
                            True, self.num_params, activation='relu',
                            dropout_rate=dropout_rate, weight_decay=weight_decay
                        ) for _ in range(self.scales)
                    ]
        else:
            if self.fusion_opt == 'concat':
                self.concat_feature_to_params = tfkl.Dense(1, activation='relu')
            elif self.fusion_opt == 'add':
                self.dense_before_add = [
                    tfkl.Dense(1, activation='relu') for _ in range(self.scales)
                ]

        self.flatten = tfkl.Flatten()
        if self.fusion_opt == 'concat':
            self.dense_before_concat = \
                [DenseBlock(is_bayesian, units_before_concat, activation, kl_weight, dropout_rate, weight_decay)
                 for _ in range(self.scales)]
            self.concat = tfkl.Concatenate()
            if isinstance(units_after_concat, int):
                self.dense_after_concat = DenseBlock(is_bayesian, units_after_concat,
                                                     activation, kl_weight, dropout_rate, weight_decay)
                self.batch_norm_concat = tfkl.BatchNormalization()
            else:
                self.dense_after_concat = [DenseBlock(is_bayesian, units, activation, kl_weight, dropout_rate,
                                                      weight_decay) for units in units_after_concat]

                self.batch_norm_concat = [tfkl.BatchNormalization() for _ in range(len(units_after_concat))]

    def call(self, inputs, training=None, mask=None):
        # step 1: get features of pvt patches
        # preprocess (multiscale) pvt patches
        preprocessed_pvt = [self.preprocess_pvt(inp) for inp in inputs[1:]]
        # extract the features of pvt patches
        # Note that multiscale patches share the same feature extractor
        pvt_features = [self.flatten(self.neighbor_feature_extractor(pvt))
                        for pvt in preprocessed_pvt]
        # step 2: get features of p, v data
        if self.pv_units is None:
            pv_features = inputs[0]
        else:
            pv_features = self.pv_feature_extractor(inputs[0])
        # step 3: fusion the features
        if self.fusion_opt == 'concat':
            pvt_features = [dense(pvt) for dense, pvt in zip(self.dense_before_concat, pvt_features)]
            concat_features = self.concat([pv_features] + pvt_features)
            if isinstance(self.units_after_concat, int):
                concat_features = self.dense_after_concat(self.batch_norm_concat(concat_features))
            else:
                for _dense, _bn in zip(self.dense_after_concat, self.batch_norm_concat):
                    concat_features = _dense(_bn(concat_features))
            out = self.concat_feature_to_params(concat_features)
        elif self.fusion_opt == 'add':
            pvt_features = [dense(pvt) for dense, pvt in zip(self.dense_before_add, pvt_features)]
            out = tf.add_n([pv_features] + pvt_features)
        # step 4: output results of a deterministic model
        # or the parameters of probability distribution
        if self.is_bayesian is True:
            out = self.calc_dist_params(out)
        return out

    @staticmethod
    def calc_gaussian_params(params):
        ms = params[..., 0]
        stds = 1e-3 + 0.01 * params[..., 1]
        return tf.stack([ms, stds], axis=-1)

    @staticmethod
    def calc_stu_params(params):
        ms = params[..., 0]
        betas = 1e-3 + 0.01 * params[..., 1]
        # betas = 1 + 1e-3 + 0.01 * params[..., 1]
        a_s = 1 + 1e-3 + 0.01 * params[..., 2]
        bs = 1e-3 + 0.01 * params[..., 3]
        return tf.stack([ms, betas, a_s, bs], axis=-1)


class PreprocessNeighPVT(tfkl.Layer):
    """preprocess a neighbor patch"""
    def __init__(
            self,
            mode=None,
            **kwargs
    ):
        self.mode = mode
        map_fn = {
            'only_p': tfkl.Lambda(lambda x: x[..., :1]),
            'only_v': tfkl.Lambda(lambda x: x[..., 1:2]),
            'only_t': tfkl.Lambda(lambda x: x[..., 2:]),
            'only_pv': tfkl.Lambda(lambda x: x[..., :2]),
            'only_pt': tfkl.Lambda(lambda x: tf.stack([x[..., 0], x[..., 2]], axis=-1)),
            'only_vt': tfkl.Lambda(lambda x: x[..., 1:]),
            't_dot_energy_without_weight': tfkl.Lambda(t_dot_energy_without_weight(
                exp_rate=0.0004, imposed_exp_rate=0.0004, lower_bound=10/255)),
            't_dot_pv_without_weight': tfkl.Lambda(t_dot_pv_without_weight(
                exp_rate=0.0004, imposed_exp_rate=0.0004, lower_bound=10/255)),
            't_dot_pv_with_learnable_exp_decay': TDotPVWithLearnableExpDecay(
                scale=0.001, imposed_exp_rate=0.0004, lower_bound=10/255),
            't_with_learnable_exp_decay': TWithLearnableExpDecay(
                scale=0.01, imposed_exp_rate=0.001, lower_bound=10/255),
            'td_dot_pv_without_weight': tfkl.Lambda(td_dot_pv_without_weight(
                t_exp_rate=0.0004, imposed_t_exp_rate=0.0004, d_exp_rate=0.04, lower_bound=10/255)),
            'td_dot_energy_without_weight': tfkl.Lambda(td_dot_energy_without_weight(
                t_exp_rate=0.0004, imposed_t_exp_rate=0.0004, d_exp_rate=0.04, lower_bound=10/255))
        }
        if mode is not None:
            assert mode in map_fn.keys(), 'mode {} is not defined'.format(mode)
            self.preprocess_layer = map_fn[mode]
        super().__init__(**kwargs)

    def call(self, x, **kwargs):
        if self.mode is None:
            return x
        return self.preprocess_layer(x)


def t_dot_energy_without_weight(exp_rate=0.0004, imposed_exp_rate=0.0004, lower_bound=10/255):
    def _t_dot_energy_without_weight(x):
        t = tf.clip_by_value(
            x[..., 2] ** (exp_rate / imposed_exp_rate),
            clip_value_max=1.,
            # fixme
            clip_value_min=0.
        )
        e_map = (x[..., 0] + 1) / (x[..., 1] + 1) * t
        return e_map[..., None]
    return _t_dot_energy_without_weight


def t_dot_pv_without_weight(exp_rate=0.0004, imposed_exp_rate=0.0004, lower_bound=10/255):
    def _t_dot_pv_without_weight(x):
        t = tf.clip_by_value(
            x[..., 2] ** (exp_rate / imposed_exp_rate),
            clip_value_max=1.,
            clip_value_min=lower_bound
        )
        return tf.stack([x[..., 0] * t, x[..., 1] * t], axis=-1)
    return _t_dot_pv_without_weight


def td_dot_pv_without_weight(t_exp_rate=0.0004, imposed_t_exp_rate=0.0004, d_exp_rate=0.04, lower_bound=10/255):
    def _td_dot_pv_without_weight(x):
        t = tf.clip_by_value(
            x[..., 2] ** (t_exp_rate / imposed_t_exp_rate),
            clip_value_max=1.,
            clip_value_min=lower_bound
        )
        # FIXME: optimization is needed here, because `d` only needs to be calculated once
        size = tf.shape(x)[1]
        ii, jj = tf.meshgrid(tf.range(size), tf.range(size), indexing='ij')
        center_index = tf.cast((size - 1) / 2, tf.int32)
        ii -= ii[center_index, center_index]
        jj -= jj[center_index, center_index]
        d = tf.clip_by_value(
            tf.exp(-d_exp_rate * tf.sqrt(tf.cast(ii ** 2 + jj ** 2, dtype=tf.float32))),
            clip_value_max=1.,
            clip_value_min=lower_bound
        )
        return tf.stack([x[..., 0] * t, x[..., 1] * t, x[..., 0] * d, x[..., 1] * d], axis=-1)
    return _td_dot_pv_without_weight


def td_dot_energy_without_weight(t_exp_rate=0.0004, imposed_t_exp_rate=0.0004, d_exp_rate=0.04, lower_bound=10/255):
    def _td_dot_energy_without_weight(x):
        t = tf.clip_by_value(
            x[..., 2] ** (t_exp_rate / imposed_t_exp_rate),
            clip_value_max=1.,
            clip_value_min=0.
        )
        # FIXME: optimization is needed here, because `d` only needs to be calculated once
        size = tf.shape(x)[1]
        ii, jj = tf.meshgrid(tf.range(size), tf.range(size), indexing='ij')
        center_index = tf.cast((size - 1) / 2, tf.int32)
        ii -= ii[center_index, center_index]
        jj -= jj[center_index, center_index]
        d = tf.clip_by_value(
            tf.exp(-d_exp_rate * tf.sqrt(tf.cast(ii ** 2 + jj ** 2, dtype=tf.float32))),
            clip_value_max=1.,
            clip_value_min=lower_bound
        )
        e_map = (x[..., 0] + 1) / (x[..., 1] + 1) * t * d
        return e_map[..., None]
    return _td_dot_energy_without_weight


class TDotPVWithLearnableExpDecay(tfkl.Layer):
    def __init__(self, scale=0.001, imposed_exp_rate=0.0004, lower_bound=10/255, **kwargs):
        # `exp_rate` will be restricted within (0, scale)
        self.scale = scale
        # the input data has already been processed with `imposed_exp_rate`,
        # i.e., t_input = tf.exp(-imposed_exp_rate * (t_original_last - t_original))
        # and note that if t_input < 10/255, t_input = 10/255
        self.imposed_exp_rate = imposed_exp_rate
        self.lower_bound = lower_bound
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.exp_rate = self.add_weight(
            name='exp_rate',
            shape=(1, ),
            initializer=tfk.initializers.constant(0.),
            trainable=True
        )

    def call(self, x):
        # stretch the sigmoid function so that it is less sensitive
        # to the change of `self.exp_rate`
        exp_rate = self.scale * tf.sigmoid(0.1 * self.exp_rate)
        # tf.exp(-imposed_exp_rate * t_span) ** (exp_rate / imposed_exp_rate)
        # --> tf.exp(-exp_rate * t_span)
        t = tf.clip_by_value(
            x[..., 2] ** (exp_rate / self.imposed_exp_rate),
            clip_value_max=1.,
            clip_value_min=self.lower_bound,
        )
        return tf.stack([x[..., 0] * t, x[..., 1] * t], axis=-1)

    def compute_output_shape(self, input_shape):
        # [B, H, W, 3] --> [B, H, W, 2]
        return tf.TensorShape(input_shape.as_list()[:-1] + [2])


class TWithLearnableExpDecay(tfkl.Layer):
    # TODO: increase the scale
    def __init__(self, scale=0.01, imposed_exp_rate=0.001, lower_bound=10/255, **kwargs):
        # `exp_rate` will be restricted within (0, scale)
        self.scale = scale
        # the input data has already been processed with `imposed_exp_rate`,
        # i.e., t_input = tf.exp(-imposed_exp_rate * (t_original_last - t_original))
        # and note that if t_input < 10/255, t_input = 10/255
        self.imposed_exp_rate = imposed_exp_rate
        self.lower_bound = lower_bound
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.exp_rate = self.add_weight(
            name='exp_rate',
            shape=(1, ),
            initializer=tfk.initializers.constant(0.),
            trainable=True
        )

    def call(self, x):
        # stretch the sigmoid function so that it is less sensitive
        # to the change of `self.exp_rate`
        exp_rate = self.scale * tf.sigmoid(0.1 * self.exp_rate)
        # tf.exp(-imposed_exp_rate * t_span) ** (exp_rate / imposed_exp_rate)
        # --> tf.exp(-exp_rate * t_span)
        # Note: Assume the t map is in the last channel
        t = tf.clip_by_value(
            x[..., -1] ** (exp_rate / self.imposed_exp_rate),
            clip_value_max=1.,
            clip_value_min=self.lower_bound,
            )
        return tf.concat([x[..., :-1], t[..., None]], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape


class PVFeatureExtrator(tfk.Model):
    def __init__(
            self,
            is_bayesian,
            unit_list,
            activation,
            kl_weight=None,
            dropout_rate=None,
            output_dist=None,
            weight_decay=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        if is_bayesian:
            assert dropout_rate is not None or kl_weight is not None, \
                'Please specify `kl_weight` or `dropout_rate` for Bayesian layers.'
            assert output_dist is not None, \
                'Please specify an output distribution to `output_dist` as `gaussian` or `student`.'
            if output_dist == 'gaussian':
                num_params = 2
            elif output_dist == 'student':
                num_params = 4
            else:
                raise ValueError('output distribution {} is not defined'.format(output_dist))
        else:
            num_params = 1
        self.dense_list = [DenseBlock(is_bayesian, unit, activation, kl_weight, dropout_rate, weight_decay)
                           for unit in unit_list]
        if is_bayesian and dropout_rate is not None:
            self.dense_list += [DenseBlock(False, num_params, 'relu', weight_decay)]
        else:
            self.dense_list += [DenseBlock(True, num_params, 'relu', kl_weight, dropout_rate=None)]

    def call(self, inputs, training=None, mask=None):
        for dense in self.dense_list:
            inputs = dense(inputs)
        return inputs


class TestModel1(tfk.Model):
    def __init__(
            self,
            blocks,
            filters,
            weight_map_blocks,
            weight_map_filters,
            units,
            activation,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.conv_batch_norm_blocks = [[
            ConvBlock(False, i * filters, activation) for _ in range(2)
        ]
            for i in [2 ** j for j in range(blocks)]]
        self.max_pooling = [tfkl.MaxPooling2D(pool_size=(2, 2)) for _ in range(blocks)]
        self.flatten = tfkl.Flatten()
        self.n_dense = [
            DenseBlock(False, units, activation),
            DenseBlock(False, 1, 'relu')
        ]
        self.pv_dense = DenseBlock(False, 1, 'relu')
        self.n_to_energy_layer = tfkl.Lambda(self.pv_to_energy1)
        # self.t_to_weight_layer = tfk.Sequential([
        #     ConvBlock(False, i * weight_map_filters, activation) for _ in range(2)
        #     for i in [2 ** j for j in range(weight_map_blocks)]
        # ] + [ConvBlock(False, 1, 'sigmoid')]
        # )

    def call(self, x):
        pv, n = x
        # weight_map = self.t_to_weight_layer(n[..., 2][..., tf.newaxis])
        weight_map = n[..., 2]
        n = self.n_to_energy_layer([n[..., 0], n[..., 1], weight_map])
        for blocks, max_pooling in zip(self.conv_batch_norm_blocks, self.max_pooling):
            for block in blocks:
                n = block(n)
            n = max_pooling(n)
        n = self.flatten(n)
        for dense_layer in self.n_dense:
            n = dense_layer(n)
        pv = self.pv_dense(pv)
        return tf.add_n([n, pv])

    @staticmethod
    def pv_to_energy(inputs):
        p_map, v_map, weight_map = inputs
        e_map = (p_map + 1) / (v_map + 1) * weight_map
        return e_map[..., tf.newaxis]

    @staticmethod
    def pv_to_energy1(inputs):
        p_map, v_map, weight_map = inputs
        e_map = (p_map + 1) / (v_map + 1) * weight_map
        return e_map[..., tf.newaxis]


def posterior_mean_field(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])


# for DenseVariational layers
def prior_trainable(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1))
    ])


# for reparameterization layers
def trainable_prior_fn(dtype, shape, name, trainable, add_variable_fn):
    loc = add_variable_fn(
        name=name + '_loc',
        shape=shape,
        initializer=tf.initializers.random_normal(stddev=0.1),
        regularizer=None,
        constraint=None,
        dtype=dtype,
        trainable=trainable
    )
    dist = tfd.Normal(loc=loc, scale=1.)
    batch_ndims = tf.size(dist.batch_shape_tensor())
    return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)


def transform_to_StudentT(t):
    m, beta, a, b = t[..., 0], t[..., 1], t[..., 2], t[..., 3]
    return tfd.StudentT(df=2 * a, loc=m, scale=tf.math.sqrt(b * (beta + 1) / (a * beta)))


def transform_to_Gaussian(t):
    return tfd.Normal(loc=t[..., 0], scale=t[..., 1])


def relative_error(y_true, y_pred):
    return tf.abs(y_true - y_pred) / y_true * 100


def kl_approx(q, p, q_tensor):
    return tf.reduce_mean(q.log_prob(q_tensor) - p.log_prob(q_tensor))
