import os
import warnings

from collections import namedtuple

import tqdm
import numpy as np
import json

# p_min and v_min must be zeros here, because directly use n[..., 0] - p_min for normalization
# if p_min is not zero, e.g., p_min = 50, then the grid that has p = 50 will be normalized to 0,
# which cannot be distinguished from empty grids
p_min, p_max = 0., 235.135148
v_min, v_max = 0., 906.86444687


def print_args(args):
    if isinstance(args, dict):
        items = args.items()
    else:
        items = args.__dict__.items()
    print("######################################  Arguments  ######################################")
    for k, v in items:
        print("{0: <30}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("#########################################################################################")


def save_args(args, save_dir, is_print_args=True):
    print("Save dir:", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if is_print_args:
        print_args(args)
    with open(os.path.join(save_dir, r'opt.json'), 'w') as f:
        json.dump(args.__dict__, f)


def load_args(filename, is_print_args=True):
    with open(filename, 'r') as f:
        args = json.load(f)
    if is_print_args:
        print_args(args)
    Args = namedtuple('Args', args.keys())
    args = Args(**args)
    return args


def get_small_patches_from_larger_ones(data, crop_size):
    assert crop_size <= data.shape[1]
    length = int((crop_size - 1) / 2)
    center = int((data.shape[1] - 1) / 2)
    temp = 1 if crop_size % 2 == 1 else 2
    return data[:, center - length:center + length + temp,
                center - length:center + length + temp].copy()


def load_data_randomly(data_dir, mode, seed, patch_size, prob_name='mp_size',
                       filter_len=None, NBEM_mode=False):
    layers = np.array([18 + 12 * i for i in range(20)])
    layers = np.hstack([np.array([13 + 12 * i for i in range(20)]), layers])
    data = load_layer_data(data_dir, layers, patch_size, prob_name=prob_name,
                           filter_len=filter_len, NBEM_mode=NBEM_mode)
    idx = np.arange(len(data['X']))
    np.random.seed(seed)
    np.random.shuffle(idx)
    n_train = int(len(data['X']) * 8 / 10)
    n_val = int(len(data['X']) * 1 / 10)
    if mode == 'train':
        ind = idx[:n_train]
    elif mode == 'valid':
        ind = idx[n_train:n_train + n_val]
    elif mode == 'test':
        ind = idx[n_train + n_val:]
    else:
        raise ValueError
    for k in data:
        data[k] = data[k][ind]
    return data


def load_layer_data(layer_data_dir, layers, patch_size, prob_name=None, filter_len=None):
    data = {'X': [], 'y': [], 'n': []}
    if prob_name is not None:
        data['prob'] = []
    record_layer_len = np.load(os.path.join(layer_data_dir, 'record_layer_len.npy'))
    record_layer_index = [0] + [int(np.sum(record_layer_len[:i]))
                                for i in range(1, len(record_layer_len) + 1)]
    for layer_index in layers:
        p = np.load(os.path.join(layer_data_dir, r'power/power_{}.npy'.format(layer_index)))
        v = np.load(os.path.join(layer_data_dir, r'velocity/velocity{}.npy'.format(layer_index)))
        n = np.load(os.path.join(layer_data_dir, r'neighbor_41/neighbor{}.npy'.format(layer_index)))
        n = get_small_patches_from_larger_ones(n, patch_size)
        if filter_len is None:
            s = np.load(os.path.join(layer_data_dir, 'size/size_{}.npy'.format(layer_index)))
        else:
            s = np.load(os.path.join(layer_data_dir, r'denoised_mp_size_median_{}.npy'.format(filter_len)))
            s = s[record_layer_index[(layer_index - 9) // 3]:
                  record_layer_index[(layer_index - 9) // 3 + 1]]

        p[p > p_max] = p_max
        v[v > v_max] = v_max
        n[..., 0][n[..., 0] > p_max] = p_max
        n[..., 1][n[..., 1] > v_max] = v_max

        # normalization
        p = (p - p_min) / (p_max - p_min)
        v = (v - v_min) / (v_max - v_min)
        # Because p_min is zero, so do not need to deal with empty grids specifically
        n[..., 0] = (n[..., 0] - p_min) / (p_max - p_min)
        n[..., 1] = (n[..., 1] - v_min) / (v_max - v_min)

        data['X'].append(np.hstack([p.reshape(-1, 1), v.reshape(-1, 1)]))
        data['y'].append(s)
        data['n'].append(n)
        if prob_name is not None:
            prob = np.load(os.path.join(layer_data_dir, r'{}_prob.npy'.format(prob_name)))
            data['prob'].append(prob[record_layer_index[(layer_index - 9) // 3]:
                                     record_layer_index[(layer_index - 9) // 3 + 1]])
    data['X'] = np.vstack(data['X'])
    data['y'] = np.hstack(data['y'])
    if prob_name is not None:
        data['prob'] = np.hstack(data['prob'])
    data['n'] = np.concatenate(data['n'], axis=0)

    valid_key = data['y'] > 0
    for k in data:
        data[k] = data[k][valid_key]

    return data


def load_mp_data(layer_data_dir, layers, crop_size, normalize=True, filter_len=None):
    if filter_len is not None:
        s = np.load(os.path.join(layer_data_dir, r'denoised_mp_size_median_{}.npy'.format(filter_len)))
        record_layer_len = np.load(os.path.join(layer_data_dir, 'record_layer_len.npy'))
        record_layer_index = [0] + [int(np.sum(record_layer_len[:i]))
                                    for i in range(1, len(record_layer_len) + 1)]

    def _normalize(r):
        # there are outlier points that v can be 1049.99691904 and 6785.468085
        r['power'][r['power'] > p_max] = p_max
        r['velocity'][r['velocity'] > v_max] = v_max
        r['n'][..., 0][r['n'][..., 0] > p_max] = p_max
        r['n'][..., 1][r['n'][..., 1] > v_max] = v_max
        # normalization
        r['power'] = (r['power'] - p_min) / (p_max - p_min)
        r['velocity'] = (r['velocity'] - v_min) / (v_max - v_min)
        r['n'][..., 0] = (r['n'][..., 0] - p_min) / (p_max - p_min)
        r['n'][..., 1] = (r['n'][..., 1] - v_min) / (v_max - v_min)
        return r

    keys = ['melt_x', 'melt_y', 'power', 'velocity', 'mp_size', 'mp_ap', 'n']
    idx = [0, 1, 2, 3, 6, 7]
    results = {k: [] for k in keys}
    if not isinstance(layers, list):
        layers = [layers]
    for layer in tqdm.tqdm(layers):
        data = np.load(os.path.join(layer_data_dir, r'layer_{}.npy'.format(layer)))
        half_patch_size = int((data.shape[1] - 1) / 2)
        effective_ids = np.where(data[:, half_patch_size, half_patch_size, 6] > 0)
        data = data[effective_ids]
        n = np.load(os.path.join(layer_data_dir, r'neighbor_41/neighbor_{}.npy'.format(layer)))
        n = get_small_patches_from_larger_ones(n, crop_size)
        if len(layers) == 1:
            for k in range(len(idx)):
                results[keys[k]] = data[:, half_patch_size, half_patch_size, idx[k]]
            results['n'] = n
            if normalize is True:
                results = _normalize(results)
            if filter_len is not None:
                results['mp_size'] = s[record_layer_index[(layer - 9) // 3]:
                                       record_layer_index[(layer - 9) // 3 + 1]]
            return results
        else:
            for k in range(len(idx)):
                if k == 4 and filter_len is not None:
                    results[keys[k]].append(
                        s[record_layer_index[(layer - 9) // 3]:
                          record_layer_index[(layer - 9) // 3 + 1]]
                    )
                else:
                    results[keys[k]].append(data[:, half_patch_size, half_patch_size, idx[k]])
            results['n'].append(n)
    for k in range(len(idx)):
        results[keys[k]] = np.hstack(results[keys[k]])
    results['n'] = np.concatenate(results['n'], axis=0)
    if normalize is True:
        results = _normalize(results)

    valid_key = results['mp_size'] > 0
    for k in results:
        results[k] = results[k][valid_key]

    return results
