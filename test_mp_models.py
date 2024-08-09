import argparse
import datetime
import json
import os.path

import numpy as np
import tensorflow as tf

from collections import namedtuple
from tensorflow.keras.regularizers import L2

from MPModels.mp_models import MPSigRegressor, relative_error
from utils.test_utils import test_with_bayesian_stu, test_with_bayesian_gaussian, test_with_non_bayesian
from utils.visual_utils import predict_a_layer, plot_ordered_relative_error
from utils.dataload_utils import load_data_randomly, load_mp_data, load_layer_data

parser = argparse.ArgumentParser('Testing melt pool models')

# ----------------------------------------- testing --------------------------------------------------
parser.add_argument('--save_dir', default=None)
parser.add_argument('--write_to_txt', action='store_true',
                    default=True, help='write results into .txt file or not')

# ----------------------------------------- testing parameters --------------------------------------------
parser.add_argument('--infer_times_test', type=int, default=50,
                    help='the inference times of a Bayesian model used in test')

opts = parser.parse_args()
model_dir_name = os.path.split(opts.save_dir)[-1]
weight_filepath = os.path.join(opts.save_dir, model_dir_name + '_relative_error.tf')
results_savepath = os.path.join(opts.save_dir, 'relative_error.txt')

with open(os.path.join(opts.save_dir, model_dir_name + '.json'), 'r') as f:
    params = json.load(f)

Args = namedtuple('Args', params.keys())
args = Args(**params)

patch_sizes = [int(size) for size in args.patch_sizes.split(',')]
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


def test():
    prob_name = None if args.use_train_prob is False else args.prob_name
    test_data = load_data_randomly(data_dir=args.data_dir, mode='test', seed=args.seed, patch_size=max(patch_sizes),
                                   prob_name=prob_name, filter_len=args.filter_len)

    if args.model_name == 'non_bayesian':
        test_model = MPSigRegressor(is_bayesian=False, **model_params_dict)
        test_model.compile(loss='mse', metrics=[relative_error])
        print('Testing the model with weight {} ...'.format(weight_filepath))
        test_model.load_weights(weight_filepath).expect_partial()
        test_results = test_with_non_bayesian(test_model, test_data, batch_size=args.val_batch_size)
    else:
        kl_weight = 1 / len(test_data['n']) if args.dropout_rate is None else None
        test_model = MPSigRegressor(
            is_bayesian=True,
            kl_weight=kl_weight,
            **model_params_dict
        )
        print('Testing the model with weight {} ...'.format(weight_filepath))
        test_model.load_weights(weight_filepath).expect_partial()

        if args.model_name == 'student':
            test_results = test_with_bayesian_stu(test_model, test_data,
                                                  args.val_batch_size, opts.infer_times_test)
        elif args.model_name == 'gaussian':
            test_results = test_with_bayesian_gaussian(test_model, test_data,
                                                       args.val_batch_size, opts.infer_times_test)
        else:
            raise ValueError('model name {} has not been implemented'.format(args.model_name))

    plot_ordered_relative_error(test_data['y'], test_results['per_sample_relative_error'],
                                os.path.join(os.path.split(results_savepath)[0], r'ordered_relative_error.png'))

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if results_savepath is not None:
        if args.model_name != 'non_bayesian':
            np.save(os.path.join(os.path.split(results_savepath)[0],
                                 f'log_evidence_per_iter_{opts.infer_times_test}.npy'),
                    test_results['log_evidence_per_iter'])
            np.save(os.path.join(os.path.split(results_savepath)[0], r'within_std_per_all.npy'),
                    test_results['within_std_per_all'])
            np.save(os.path.join(os.path.split(results_savepath)[0], r'within_std_gau.npy'),
                    test_results['within_std_gau'])
        with open(results_savepath, 'a') as f:
            if opts.write_to_txt is True:
                f.write('\n' + date + '\nTesting results with weight {}:\n'.format(weight_filepath))
            for k, v in test_results.items():
                if k in ['mean_test_mse', 'mean_test_abe', 'mean_test_nnl', 'mean_test_relative_error',
                         'within_std_per', 'top_100_relative_error', 'ece']:
                    if opts.write_to_txt is True:
                        f.write("{0: <30}\t{1: <30}\n".format(k, str(v)))
                    print("{0: <30}\t{1: <30}".format(k, str(v)))
        print('Finish')


if __name__ == '__main__':
    test()
