import os
import time
import sys
import tqdm
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from functools import partial
from sklearn.neighbors import KernelDensity

tfd = tfp.distributions
tfpl = tfp.layers


def calc_dis_square_of_two_matrix(ma, mb):
    return np.sum(ma ** 2, axis=1).reshape(-1, 1) + np.sum(mb ** 2, axis=1) - 2 * ma @ mb.T


def calc_distance(key, data):
    dis = np.linalg.norm(data - key, axis=-1)
    return dis, np.argsort(dis)[1:]


def data_denoising(
        mp_size,
        power,
        velocity,
        hidden_code,
        p_thresh=1,
        v_thresh=1,
        dis_thresh=10,
        mode='median',
        **kwargs
):
    if mode == 'median':
        assert 'filter_len' in kwargs.keys(), \
            'Please specify the filtering length (an odd number) for the median filter.'
        filter_len = kwargs['filter_len']
        assert filter_len % 2 == 1, \
            'The filtering length for the median filter must be an odd number.'
    denoised_mp_size = []
    mp_size_std = []
    for i in tqdm.tqdm(range(len(hidden_code))):
        # for i in range(len(hidden_code)):
        neighbor_idx = find_neighbors_of_neighbor_maps(hidden_code[i], power[i], velocity[i],
                                                       hidden_code, power, velocity,
                                                       p_thresh=p_thresh, v_thresh=v_thresh,
                                                       dis_thresh=dis_thresh)
        if mode == 'median':
            if len(neighbor_idx) >= filter_len - 1:
                temp_mp_size = np.append(mp_size[neighbor_idx[:filter_len - 1]], mp_size[i])
                denoised_mp_size.append(np.sort(temp_mp_size)[int((filter_len - 1) / 2)])
            else:
                denoised_mp_size.append(mp_size[i])

            if len(neighbor_idx) >= 5:
                temp_mp_size1 = np.append(mp_size[neighbor_idx], mp_size[i])
                mp_size_std.append(np.sqrt(temp_mp_size1.var()))
            else:
                mp_size_std.append(0)
    return np.array(denoised_mp_size), np.array(mp_size_std)


def data_denoising_for_all_maps(
        mp_size,
        hidden_data,
        powers,
        velocities,
        p_thresh=1,
        v_thresh=1,
        dis_thresh=10,
        mode='median',
        **kwargs
):
    if mode == 'median':
        assert 'filter_len' in kwargs.keys(), \
            'Please specify the filtering length (an odd number) for the median filter.'
        filter_len = kwargs['filter_len']
        assert filter_len % 2 == 1, \
            'The filtering length for the median filter must be an odd number.'
    denoised_mp_size, mp_size_std = [], []
    dis_sqaure_thresh = dis_thresh ** 2
    num_seg = 400
    index = [len(hidden_data) // num_seg * i for i in range(num_seg)] + [len(hidden_data)]
    for i in tqdm.tqdm(range(len(index) - 1)):
        seg = hidden_data[index[i]:index[i + 1]]
        dis_square = calc_dis_square_of_two_matrix(seg, hidden_data)
        p_diff = np.abs(powers[index[i]:index[i + 1]].reshape(-1, 1) - powers)
        v_diff = np.abs(velocities[index[i]:index[i + 1]].reshape(-1, 1) - velocities)
        dis_idx = [set(np.where(dis_square[j] < dis_sqaure_thresh)[0]) for j in range(len(seg))]
        p_idx = [set(np.where(p_diff[j] < p_thresh)[0]) for j in range(len(seg))]
        del p_diff
        v_idx = [set(np.where(v_diff[j] < v_thresh)[0]) for j in range(len(seg))]
        del v_diff
        for j in range(len(seg)):
            v_idx[j].remove(index[i] + j)
        neighbor_idx = [list(dis_idx[j] & p_idx[j] & v_idx[j]) for j in range(len(seg))]
        neighbor_idx = [np.array(neighbor_idx[j])[np.argsort(dis_square[j][neighbor_idx[j]])]
                        if len(neighbor_idx[j]) > 0 else []
                        for j in range(len(seg))]
        del dis_square
        for j in range(len(seg)):
            if mode == 'median':
                if len(neighbor_idx[j]) >= filter_len - 1:
                    temp_mp_size = np.append(mp_size[neighbor_idx[j][:filter_len - 1]], mp_size[index[i] + j])
                    denoised_mp_size.append(np.sort(temp_mp_size)[int((filter_len - 1) / 2)])
                else:
                    denoised_mp_size.append(mp_size[index[i] + j])
                if len(neighbor_idx[j]) >= 20:
                    temp_mp_size1 = np.append(mp_size[neighbor_idx[j]], mp_size[index[i] + j])
                    mp_size_std.append(np.sqrt(temp_mp_size1.var()))
                else:
                    mp_size_std.append(0)
    return np.array(denoised_mp_size), np.array(mp_size_std)


def _get_filtered_mp_size(
        i,
        mp_size,
        power,
        velocity,
        hidden_code,
        mode='median',
        **kwargs
):
    neighbor_idx = find_neighbors_of_neighbor_maps(hidden_code[i], power[i], velocity[i],
                                                   hidden_code, power, velocity,
                                                   p_thresh=1, v_thresh=1, dis_thresh=10)
    if mode == 'median':
        filter_len = kwargs['filter_len']
        if len(neighbor_idx) >= filter_len - 1:
            temp_mp_size = np.append(mp_size[neighbor_idx[:filter_len - 1]], mp_size[i])
            size = np.sort(temp_mp_size)[int((filter_len - 1) / 2)]
        else:
            size = mp_size[i]
        if len(neighbor_idx) >= 5:
            temp_mp_size1 = np.append(mp_size[neighbor_idx], mp_size[i])
            size_std = np.sqrt(temp_mp_size1.var())
        else:
            size_std = 0
        return size, size_std


def find_neighbors_of_neighbor_maps(
        key_hidden,
        key_power,
        key_velocity,
        hidden_data,
        powers,
        velocities,
        p_thresh,
        v_thresh,
        dis_thresh
):
    dis, idx = calc_distance(key_hidden, hidden_data)
    temp_idx = 0
    while dis[idx[temp_idx]] < dis_thresh:
        temp_idx += 1
    neighbor_idx = [i for i in idx[:temp_idx]
                    if abs(key_power - powers[i]) < p_thresh and
                    abs(key_velocity - velocities[i]) < v_thresh]
    return neighbor_idx


def generate_data_idx_randomly(mp_size, random_seed=0):
    mp_size_sort = np.argsort(mp_size)
    train_idx, val_idx, test_idx = [], [], []
    np.random.seed(random_seed)
    data_split_idx = np.array([0, 1, 2, 3])
    for i in range(len(mp_size) // 4):
        temp_idx = mp_size_sort[4 * i:4 * (i + 1)][data_split_idx]
        train_idx.extend(temp_idx.tolist()[:2])
        val_idx.append(temp_idx[2])
        test_idx.append(temp_idx[3])
    j = len(mp_size) // 4
    if len(mp_size) > 4 * j:
        train_idx.extend(mp_size_sort.tolist()[4 * j:])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def get_small_patches_from_larger_ones(data, crop_size):
    assert crop_size <= data.shape[1]
    length = int((crop_size - 1) / 2)
    center = int((data.shape[1] - 1) / 2)
    temp = 1 if crop_size % 2 == 1 else 2
    return data[:, center - length:center + length + temp,
                center - length:center + length + temp]


def kernel_dens_estimate(mp_size, bandwidth=0.75, bins=200, **kwargs):
    _mp_size = mp_size.copy()
    print('Estimating melt pool size distribution ...', end='')
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(mp_size.reshape(-1, 1))
    print('finish')
    print('Calculating sample weights ...', end=' ')
    cores = multiprocessing.cpu_count()
    mp_size = mp_size.reshape(-1, 1)[..., np.newaxis]
    time1 = time.time()
    with multiprocessing.Pool(processes=cores) as pool:
        log_dens = pool.map(kde.score_samples, mp_size)
    # log_dens = kde.score_samples(mp_size.reshape(-1, 1))
    print(time.time() - time1)
    log_dens = np.squeeze(log_dens)
    prob = np.exp(log_dens)
    prob = prob / np.sum(prob)
    fig = plt.figure()
    dens, coords, _ = plt.hist(_mp_size, bins=bins, density=True)
    dens = np.append(dens, dens[-1])
    prob1 = dens[((_mp_size - coords.min()) // ((coords.max() - coords.min()) / bins)).astype(np.int)]
    prob1 /= np.sum(prob1)
    plt.plot(np.sort(_mp_size), prob1[np.argsort(_mp_size)] * bins / 2)
    plt.show()
    fig1 = plt.figure()
    plt.hist(_mp_size, bins=bins, density=True)
    plt.plot(np.sort(_mp_size), prob[np.argsort(_mp_size)] * bins / 2)
    plt.show()
    if 'save_path' in kwargs.keys():
        fig.savefig(kwargs['save_path'])
    print('finish')
    return prob, prob1


def hist_density_estimate(mp_size, bins=200, **kwargs):
    fig = plt.figure()
    dens, coords, _ = plt.hist(mp_size, bins=bins, density=True)
    dens = np.append(dens, dens[-1])
    prob = dens[((mp_size - coords.min()) // ((coords.max() - coords.min()) / bins)).astype(np.int)]
    plt.plot(np.sort(mp_size), prob[np.argsort(mp_size)])
    prob /= np.sum(prob)
    assert prob.min > 0., 'please choose proper number of bins.'
    plt.show()
    if 'save_path' in kwargs.keys():
        fig.savefig(kwargs['save_path'])
    print('finish')
    return prob


def load_data_layerwisely(mode, seed, patch_size, filter_len=None):
    # `p_min` is actually 58.943662, but in neighbor maps there are zero values
    # Note that p_max and v_max are not the maximum of record points but the melting points
    p_min, p_max = 0., 235.135148
    v_min, v_max = 0., 906.86444687
    s_min, s_max = 0., 1168
    idx_list = [np.array([12 + 3 * j + 12 * i for i in range(20)]) for j in range(1, 4)]
    np.random.seed(seed)
    for i in range(len(idx_list)):
        np.random.shuffle(idx_list[i])
    record_layer_len = np.load('record_layer_len.npy')
    record_layer_index = [0] + [np.sum(record_layer_len[:i])
                                for i in range(1, len(record_layer_len) + 1)]
    if mode == 'train':
        layers = np.hstack([np.array([9])] + [id_list[:14] for id_list in idx_list])
    elif mode == 'valid':
        layers = np.hstack([id_list[14:17] for id_list in idx_list])
    elif mode == 'test':
        layers = np.hstack([id_list[17:] for id_list in idx_list])
    else:
        raise ValueError('`mode` should be train, valid, or test')
    data = {'X': [], 'y': [], 'n': [], 'prob': []}
    for layer_index in layers:
        p = np.load(r'../power/power_{}.npy'.format(layer_index))
        v = np.load(r'../velocity/velocity_{}.npy'.format(layer_index))
        n = np.load(r'../neighbor_41/neighbor_{}.npy'.format(layer_index))
        n = get_small_patches_from_larger_ones(n, patch_size)
        if filter_len is None:
            s = np.load(r'../size/size_{}.npy'.format(layer_index))
        else:
            s = np.load(r'denoised_mp_size_median_{}.npy'.format(filter_len))
            s = s[record_layer_index[(layer_index - 9) // 3]:
                  record_layer_index[(layer_index - 9) // 3 + 1]]

        # there are outlier points that v can be 1049.99691904 and 6785.468085
        p[p > p_max] = p_max
        v[v > v_max] = v_max
        n[..., 0][n[..., 0] > p_max] = p_max
        n[..., 1][n[..., 1] > v_max] = v_max

        # normalization
        p = (p - p_min) / (p_max - p_min)
        v = (v - v_min) / (v_max - v_min)
        n[..., 0] = (n[..., 0] - p_min) / (p_max - p_min)
        n[..., 1] = (n[..., 1] - v_min) / (v_max - v_min)

        prob = np.load(r'mp_size_prob.npy')
        data['X'].append(np.hstack([p.reshape(-1, 1), v.reshape(-1, 1)]))
        data['y'].append(s)
        data['n'].append(n)
        data['prob'].append(prob[record_layer_index[(layer_index - 9) // 3]:
                                 record_layer_index[(layer_index - 9) // 3 + 1]])
    data['X'] = np.vstack(data['X'])
    data['y'] = np.hstack(data['y'])
    data['prob'] = np.hstack(data['prob'])
    data['n'] = np.concatenate(data['n'], axis=0)
    return data


def load_data(half_patch_size, layer_indexes, prepare_patch_data, **kwargs):
    """
    :param half_patch_size: (the length of patch in the dataset + 1) / 2
    :param layer_indexes: a list of layer to be loaded, e.g., [1, 2 ,3]
    :param prepare_patch_data: preparee `patch_data` or not
    :param kwargs:
                    patch_size: the patch size we want for down sampling
                    normalization: normalize the data or not
                   max_power: maximum power
                   min_power: minimum power
                   max_velocity: maximum scanning velocity
                   min_velocity: minimum scanning velocity
    :return:
        melt_x: the x coordinate of the current point (the center of the patch)
        melt_y: the y coordinate of the current point
        power: the laser power of the current point
        velocity: the scanning velocity of the current point
        melt_size: the melt pool size of the current point
        data: if prepare_patch_data is True, then return an ndarray including the data of all specified layers
    """
    print('Loading data ...')
    base_dir = r'../'
    melt_x, melt_y, power, velocity, melt_size = [[] for _ in range(5)]
    data = []
    # for layer in tqdm.tqdm([9 + 3*i for i in range(27)] + [9 + 3*i + 1 for i in range(5)]):
    for layer in tqdm.tqdm(layer_indexes):
        path = base_dir + r'/2D_layer_patch_data/layer_' + repr(layer) + r'.npy'
        assert os.path.exists(path), 'Layer {} does not exist.'.format(layer)
        temp_data = np.load(path)
        # melt pool size > 0
        effective_ids = np.where(temp_data[:, half_patch_size, half_patch_size, 6] > 0)
        temp_data = temp_data[effective_ids]
        data.append(temp_data.copy())
        # temp_data = temp_data[:, half_patch_size, half_patch_size]
        power.append(temp_data[:, half_patch_size, half_patch_size, 2])
        velocity.append(temp_data[:, half_patch_size, half_patch_size, 3])
        melt_size.append(temp_data[:, half_patch_size, half_patch_size, 6])
        melt_x.append(temp_data[:, half_patch_size, half_patch_size, 0])
        melt_y.append(temp_data[:, half_patch_size, half_patch_size, 1])
    power = np.hstack(power)
    velocity = np.hstack(velocity)
    melt_size = np.hstack(melt_size)
    melt_x = np.hstack(melt_x)
    melt_y = np.hstack(melt_y)
    if prepare_patch_data is True:
        assert 'patch_size' in kwargs.keys(), \
            'Please specify a new patch size by passing the `patch_size` argument.'
        assert 'normalization' in kwargs.keys(), \
            'Please specify if data normalization is needed by passing the `normalization` argument.'
        patch_size, normalization = kwargs['patch_size'], kwargs['normalization']
        print('Preparing patch data ...', end=' ')
        data = np.concatenate(data, axis=0)
        # data = np.array([pre_process_patches(patch, patch_size, order_mode='exp', exp_rate=0.0004)
        #                 for patch in data])
        data = parallel_pre_process_patches(data, patch_size, order_mode='exp', exp_rate=0.0004)
        # data normalization
        if normalization is True:
            if 'max_power' in kwargs.keys():
                max_power, min_power = kwargs['max_power'], kwargs['min_power']
                max_velocity, min_velocity = kwargs['max_velocity'], kwargs['min_velocity']
            else:
                max_power, min_power = np.max(data[..., 0]), np.min(data[..., 0])
                max_velocity, min_velocity = np.max(data[..., 1]), np.min(data[..., 1])
            data[..., 0] = (data[..., 0] - min_power) / (max_power - min_power)
            data[..., 1] = (data[..., 1] - min_velocity) / (max_velocity - min_velocity)
        print('finish')
        return melt_x, melt_y, power, velocity, melt_size, data
    return melt_x, melt_y, power, velocity, melt_size


def load_denoised_data(layer_indexes, data_path=r'denoised_mp_size_median_21.npy'):
    denoised_mp_size = np.load(data_path)
    record_layer_len = np.load(r'record_layer_len.npy')
    record_layer_index = [0] + [np.sum(record_layer_len[:i]) for i in range(1, len(record_layer_len) + 1)]
    record_mp_size = []
    for layer_index in layer_indexes:
        record_mp_size.append(denoised_mp_size[record_layer_index[(layer_index - 9) // 3]:
                                               record_layer_index[(layer_index - 9) // 3 + 1]])
    return np.hstack(record_mp_size)


def load_mp_data(layers, crop_size):
    p_min, p_max = 0., 235.135148
    v_min, v_max = 0., 906.86444687

    def _normalize(r):
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
        data = np.load(r'../new_2D_patch_data/new_2D_patch_data_layer_{}.npy'.format(layer))
        half_patch_size = int((data.shape[1] - 1) / 2)
        effective_ids = np.where(data[:, half_patch_size, half_patch_size, 6] > 0)
        data = data[effective_ids]
        n = np.load(r'../neighbor_41/neighbor_{}.npy'.format(layer))
        n = get_small_patches_from_larger_ones(n, crop_size)
        if len(layers) == 1:
            for k in range(len(idx)):
                results[keys[k]] = data[:, half_patch_size, half_patch_size, idx[k]]
            results['n'] = n
            results = _normalize(results)
            return results
        else:
            for k in range(len(idx)):
                results[keys[k]].append(data[:, half_patch_size, half_patch_size, idx[k]])
            results['n'].append(n)
    for k in range(len(idx)):
        results[keys[k]] = np.hstack(results[keys[k]])
    results['n'] = np.concatenate(results['n'], axis=0)
    results = _normalize(results)
    return results


def parallel_pre_process_patches(samples, map_size, order_mode='exp', **kwargs):
    cores = multiprocessing.cpu_count()
    _pre_process_a_patch = partial(pre_process_patches, map_size=map_size, order_mode=order_mode, **kwargs)
    # time1 = time.time()
    with multiprocessing.Pool(processes=cores) as pool:
        data = pool.map(_pre_process_a_patch, samples)
    # time2 = time.time()
    # print(time2-time1)
    return np.array(data)


def pre_process_patches(sample, map_size, order_mode='exp', **kwargs):
    """
    :param sample: one patch
    :param map_size: the size we need in down sampling
    :param order_mode: 'linear' or 'exp' attenuation
    :param kwargs: specify the `exp_rate` if `order_mode` is 'exp'
    :return: ndarray of shape (map_size, map_size, 3),
             the last 3 dimensions are laser power, scan velocity, and attenuation
    """
    assert order_mode in ['linear', 'exp'], order_mode + ' has not been defined.'
    if order_mode == 'exp':
        assert 'exp_rate' in kwargs.keys(), 'Please specify an exponential rate.'
        exp_rate = kwargs['exp_rate']
    patch_size_0 = sample.shape[0]
    assert patch_size_0 % 2 == 1, 'input size should be an odd number by default'
    half_patch_size_0 = int((patch_size_0 - 1) / 2)
    half_patch_size_1 = int((map_size - 1) / 2)
    if map_size > 0:
        if map_size % 2 == 1:
            sample = sample[half_patch_size_0 - half_patch_size_1:half_patch_size_0 + half_patch_size_1 + 1,
                     half_patch_size_0 - half_patch_size_1:half_patch_size_0 + half_patch_size_1 + 1]
        else:
            sample = sample[half_patch_size_0 - half_patch_size_1:half_patch_size_0 + half_patch_size_1 + 2,
                     half_patch_size_0 - half_patch_size_1:half_patch_size_0 + half_patch_size_1 + 2]
    sample1 = sample.copy()

    for i in range(map_size):
        for j in range(map_size):
            # melted after the current point
            if sample[i][j][4] > sample[half_patch_size_1][half_patch_size_1][4]:
                sample1[i][j] = np.array([0 for _ in range(7)])
                if order_mode == 'linear':
                    sample1[i][j][4] = -1
            else:
                # the current point or the powder
                if sample[i][j][4] == 0:
                    if order_mode == 'linear':
                        sample1[i][j][4] = -1
                else:
                    sample1[i][j][4] = sample[half_patch_size_1][half_patch_size_1][4] - \
                                       sample[i][j][4]
                    if order_mode == 'exp':
                        sample1[i][j][4] = np.exp(-exp_rate * sample1[i][j][4])
                        # set a lower bound
                        if sample1[i][j][4] < 10 / 255:
                            sample1[i][j][4] = 10 / 255
    if order_mode == 'linear':
        max_index_diff = np.max(sample1[:, :, 4])
        for i in range(map_size):
            for j in range(map_size):
                if sample1[i][j][4] == -1:
                    sample1[i][j][4] = 1.1 * max_index_diff
    return sample1[..., 2:5]


def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                          for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) + metrics, end=end)


def random_batch(X, y, batch_size=32):
    idx = np.random.randint(low=0, high=len(y), size=batch_size)
    return X[idx], y[idx]


def sample_data_with_given_prob(data, prob, n_sample):
    ids = np.random.choice(a=np.array(range(prob.shape[0])),
                           size=n_sample,
                           replace=False,
                           p=prob / np.sum(prob))
    return data[ids]
