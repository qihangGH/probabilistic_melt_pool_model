import argparse
import datetime
import json
import os.path
import time
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PiecewiseConstantDecay
from tensorflow.keras.regularizers import L2
from lr_schedule import CosineDecayRestarts
from MPModels.mp_models import MPSigRegressor, relative_error
from training_loops import training_mp_models
from utils.dataload_utils import load_data_randomly, get_small_patches_from_larger_ones
from utils.data_preprocess_utlis import augment_data

parser = argparse.ArgumentParser('Training melt pool models')
parser.add_argument('--data_dir', default=None)
parser.add_argument('--training_mode', default='train', choices=['train', 'cross_validation'],
                    help='`train` or `cross_validation`')
parser.add_argument('--save_dir', default='test',
                    help='The directory to save results')

# ----------------------------------- data parameters -----------------------------------------------------
parser.add_argument('--target_signature', default='size', help='size or ap', choices=['size', 'ap'])
parser.add_argument('--seed', type=int, default=0,
                    help='A random seed used to split train, validation, and test data')
parser.add_argument('--patch_sizes', default='32',
                    help='patch size, input multiple scales for multiscale-based prediction, e.g., \'15,25\'')
parser.add_argument('--use_train_prob', action='store_false', default=False, help='whether to use weighted train data')
parser.add_argument('--prob_name', default='mp_size',
                    choices=['energy_density', 'mp_size', 'mp_size_bins_100', 'mp_size_bins_200'],
                    help='weight train data based on energy density or melt pool size')
parser.add_argument('--power_exp_for_prob', default=1, help='scale the probability')
parser.add_argument('--filter_len', default=None, help='use filtered data where the filter length is `filter_len`')

# ------------------------------------ model parameters ---------------------------------------------------
parser.add_argument('--model_name', default='non_bayesian', choices=['student', 'gaussian', 'non_bayesian'],
                    help='model to use: `student`, `gaussian`, or `non_bayesian`')

# for Bayesian model, specify the dropout rate and weight decay
parser.add_argument('--dropout_rate', default=None, help='dropout rate for Monte Carlo dropout')
parser.add_argument('--weight_decay', default=1e-5, help='weight decay for Monte Carlo dropout')

# preprocess and extract features for neighbor patches
parser.add_argument('--preprocess_pvt_mode', default=None,
                    choices=['only_p', 'only_v', 'only_t', 'only_pv', 'only_pt', 'only_vt'
                                                                                 't_dot_energy_without_weight',
                             'td_dot_energy_without_weight',
                             't_dot_pv_without_weight',
                             't_dot_pv_with_learnable_exp_decay', 'td_dot_pv_without_weight'],
                    help='the mode to preprocess neighbor pvt data')
parser.add_argument('--blocks_feature_extractor', type=int, default=3,
                    help='the number of feature extraction blocks used for neighbor patches')
parser.add_argument('--filters_feature_extractor', type=int, default=64,
                    help='the number of filters used to extract the features of neighbor patches')
parser.add_argument('--activation', default='elu', help='activation')

# preprocess and extract features for p, v data
parser.add_argument('--pv_units', default=None, help='activation')

# fusion mode
parser.add_argument('--fusion_opt', default='concat', choices=['concat', 'add'], help='fusion mode')

# specify parameters in concatenation-based fusion
parser.add_argument('--units_before_concat', type=int, default=10,
                    help='units for a dense layer before concatenation in concatenation-based fusion')
parser.add_argument('--units_after_concat', default=12,
                    help='units for a dense layer after concatenation in concatenation-based fusion')

# ---------------------------------------- training parameters --------------------------------------------
parser.add_argument('--epochs', type=int, default=200, help='The number of training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--lr_schedule', default='PiecewiseConstantDecay',
                    choices=['NoDecay', 'ExponentialDecay', 'CosineDecayRestarts', 'PiecewiseConstantDecay'])

# CosineDecayRestarts
parser.add_argument('--first_decay_steps', type=int, default=40000, help='first decay steps for CosineDecayRestarts')
parser.add_argument('--t_mul', type=float, default=1,
                    help='used to derive the number of iterations in the i-th period.')
parser.add_argument('--m_mul', type=float, default=1,
                    help='used to derive the initial learning rate of the i-th period')

# ExponentialDecay
parser.add_argument('--decay_steps', type=int, default=80000, help='decay steps for ExponentialDecay')
parser.add_argument('--decay_rate', type=float, default=0.8, help='decay rate for ExponentialDecay')

# PiecewiseConstantDecay
parser.add_argument('--boundaries', default='12500,50000')
parser.add_argument('--values', default='0.01,0.005,0.001')

parser.add_argument('--mse_weight_schedule', default='0,0.9995',
                    help='The weight of MSE loss to be added. '
                         'The weight schedule is params[0] * params[1] ** epoch')

# ---------------------------------------- validation parameters --------------------------------------------
parser.add_argument('--val_batch_size', type=int, default=10000,
                    help='batch size used in validation, which can be set as large as possible to speed up validation')
parser.add_argument('--infer_times_val', type=int, default=3,
                    help='The inference times of a Bayesian model used in validation')
# parser.add_argument('--valid_freq_batch', type=int, default=None,
#                     help='Do validation for every specified batches')

# ---------------------------------------- resume training parameters ---------------------------------------
parser.add_argument('--resume_weights', default=
None,
                    help='the filepath of model weights for resuming training')
parser.add_argument('--resume_opt_weights', default=
None,
                    help='the filepath of optimizer weights for resuming training')

args = parser.parse_args()
if args.resume_weights is not None:
    assert args.resume_opt_weights is not None
    if args.training_mode == 'cross_validation':
        warnings.warn('Cannot resume training in the cross validation mode. Ignoring ...')
    if args.model_name == 'non_bayesian':
        warnings.warn('Resuming training of the non-Bayesian model has not been implemented. Ignoring ...')

# if args.valid_freq_batch is not None and args.model_name == 'non_bayesian':
#     warnings.warn('Validation after certain batches has not been implemented for non Bayesian models. Ignoring ...')

patch_sizes = [int(size) for size in args.patch_sizes.split(',')]
loss_fn = lambda y, p_y: -p_y.log_prob(y)
w_schedule = [float(a) for a in args.mse_weight_schedule.split(',')]
if args.lr_schedule == 'NoDecay':
    lr = args.lr
elif args.lr_schedule == 'ExponentialDecay':
    lr = (
        ExponentialDecay(
            initial_learning_rate=args.lr,
            decay_steps=args.decay_steps,
            decay_rate=args.decay_rate,
            staircase=True
        )
    )
elif args.lr_schedule == 'CosineDecayRestarts':
    lr = (
        CosineDecayRestarts(
            initial_learning_rate=args.lr,
            first_decay_steps=args.first_decay_steps,
            t_mul=args.t_mul,
            m_mul=args.m_mul
        )
    )
else:
    boundaries = [int(b) for b in args.boundaries.split(',')]
    values = [float(v) for v in args.values.split(',')]
    assert len(boundaries) == len(values) - 1
    lr = PiecewiseConstantDecay(boundaries, values)

weight_decay = L2(args.weight_decay) if args.dropout_rate is not None else None

if isinstance(args.units_after_concat, int):
    units_after_concat = args.units_after_concat
else:
    units_after_concat = [int(i) for i in args.units_after_concat.split(',')]
model_params_dict = dict(
    blocks_feature_extractor=args.blocks_feature_extractor,
    filters_feature_extractor=args.filters_feature_extractor,
    units_before_concat=args.units_before_concat,
    units_after_concat=units_after_concat,
    activation=args.activation,
    output_dist=args.model_name,
    scales=len(patch_sizes),
    pv_units=args.pv_units,
    fusion_opt=args.fusion_opt,
    dropout_rate=args.dropout_rate,
    preprocess_pvt_mode=args.preprocess_pvt_mode,
    weight_decay=weight_decay
)


def train(train_layers=None, val_layers=None, suffix=None):
    prob_name = None if args.use_train_prob is False else args.prob_name
    train_data = load_data_randomly(data_dir=args.data_dir, mode='train', seed=args.seed, patch_size=max(patch_sizes),
                                    prob_name=prob_name, filter_len=args.filter_len)
    val_data = load_data_randomly(data_dir=args.data_dir, mode='valid', seed=args.seed, patch_size=max(patch_sizes),
                                  prob_name=prob_name, filter_len=args.filter_len)

    # data augmentation
    # train_data = augment_data(train_data)
    # val_data = augment_data(val_data)

    train_n_data = [get_small_patches_from_larger_ones(train_data['n'], size)
                    for size in patch_sizes if size != max(patch_sizes)] + [train_data['n']]
    val_n_data = [get_small_patches_from_larger_ones(val_data['n'], size)
                  for size in patch_sizes if size != max(patch_sizes)] + [val_data['n']]

    if args.use_train_prob is True:
        # filter the probabilities to reduce over-fitting due to measurement noises
        sample_weights_train = 1 / (train_data['prob'] ** args.power_exp_for_prob)
        sample_weights_train /= np.sum(sample_weights_train)
    else:
        sample_weights_train = None

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = args.model_name + '_' + date
    if args.save_dir is not None:
        save_dir = args.save_dir
    if suffix is not None:
        save_dir += '_' + suffix

    print("Save dir:", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, save_dir)
    print("######################################  args  ######################################")
    with open(os.path.join(save_dir, save_dir + r'.txt'), 'w') as f:
        for k, v in args.__dict__.items():
            f.write("{0: <30}\t{1: <30}\t{2: <20}\n".format(k, str(v), str(type(v))))
            print("{0: <30}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("####################################################################################")
    with open(os.path.join(save_dir, save_dir + r'.json'), 'w') as f:
        json.dump(args.__dict__, f)
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    if args.model_name == 'non_bayesian':
        train_model = MPSigRegressor(is_bayesian=False, **model_params_dict)
        train_model.compile(optimizer=optimizer, loss='mse', metrics=[relative_error])
        checkpoint_params = {'save_best_only': True, 'save_weights_only': True, 'mode': 'min'}
        checkpoint = ModelCheckpoint(filepath=filepath + r'_mse.tf', monitor='val_loss', **checkpoint_params)
        checkpoint1 = ModelCheckpoint(filepath=filepath + r'_relative_error.tf', monitor='val_relative_error',
                                      **checkpoint_params)
        log_dir = 'logs/' + filepath + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        train_model.fit(
            [train_data['X']] + train_n_data, train_data['y'],
            validation_data=([val_data['X']] + val_n_data, val_data['y']),
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=[checkpoint, checkpoint1, tensorboard_callback],
            sample_weight=sample_weights_train
        )
    else:
        kl_weight = 1 / len(train_data['n']) if args.dropout_rate is None else None
        train_model = MPSigRegressor(
            is_bayesian=True,
            kl_weight=kl_weight,
            **model_params_dict
        )
        if args.resume_weights is not None:
            # perform forward pass to build the graph, so then we can get trainable weights
            _pred = train_model((train_data['X'][:1], train_data['n'][:1]))
            grad_vars = train_model.trainable_weights
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            optimizer.apply_gradients(zip(zero_grads, grad_vars))
            opt_weights = np.load(args.resume_opt_weights, allow_pickle=True)
            optimizer.set_weights(opt_weights)
            train_model.load_weights(args.resume_weights)
        training_mp_models(
            model_name=args.model_name,
            X_train=train_data['X'],
            y_train=train_data['y'],
            n_train=train_data['n'],
            sample_weights_train=sample_weights_train,
            X_valid=val_data['X'],
            y_valid=val_data['y'],
            n_valid=val_data['n'],
            sample_weights_valid=None,
            model=train_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            filepath=filepath,
            optimizer=optimizer,
            loss_fn=loss_fn,
            mse_weight_schedule=lambda t: w_schedule[0] * w_schedule[1] ** t,
            num_bayesian_inference=args.infer_times_val,
            val_batch_size=args.val_batch_size,
            # X_train_prob=train_data['prob'],
            # X_valid_prob=val_data['prob']
        )


if __name__ == '__main__':
    train()
