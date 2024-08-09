import multiprocessing
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.neighbors import KernelDensity
from typing import Dict


def augment_data(data: Dict) -> Dict:
    data['X'] = np.vstack([data['X'] for _ in range(8)])
    data['y'] = np.hstack([data['y'] for _ in range(8)])
    if 'prob' in data:
        data['prob'] = np.hstack([data['prob'] for _ in range(8)])
    # rotation and mirror
    data['n'] = np.concatenate([tf.image.rot90(data['n'], k).numpy() for k in range(4)])
    data['n'] = np.concatenate([data['n'], data['n'][:, ::-1]], axis=0)
    return data


def kernel_dens_estimate(mp_size, bandwidth=0.75, prob_save_path=None, fig_save_path=None):
    print('Estimating probability by KDE ...', end='')
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(mp_size.reshape(-1, 1))
    # 可视化概率分布，计算得到的是密度的对数
    melt_size_plot = np.linspace(np.min(mp_size), np.max(mp_size), 500)
    log_dens_plot = kde.score_samples(melt_size_plot.reshape(-1, 1))
    density_fig = plt.figure()
    plt.fill(melt_size_plot, np.exp(log_dens_plot), fc='#AAAAFF')
    plt.title('melt pool size distribution')
    plt.show()
    if fig_save_path is not None:
        density_fig.savefig(fig_save_path)
    plt.close(density_fig)
    print('Calculating sample weights ...', end=' ')
    cores = multiprocessing.cpu_count()
    mp_size = mp_size.reshape(-1, 1)[..., np.newaxis]
    time1 = time.time()
    with multiprocessing.Pool(processes=cores) as pool:
        log_dens = pool.map(kde.score_samples, mp_size)
    # log_dens = kde.score_samples(mp_size.reshape(-1, 1))
    print(f'Time cost: {time.time() - time1}')
    log_dens = np.squeeze(log_dens)
    prob = np.exp(log_dens)
    prob = prob / np.sum(prob)
    if prob_save_path is not None:
        np.save(prob_save_path, prob)
    print('finish')
    return prob


def hist_dens_estimate(mp_size, bins=200, prob_save_path=None, fig_save_path=None):
    print('Estimating probability by histograms ...', end='')
    fig = plt.figure()
    dens, coords, _ = plt.hist(mp_size, bins=bins, density=True)
    dens = np.append(dens, dens[-1])
    prob = dens[((mp_size - coords.min()) // ((coords.max() - coords.min()) / bins)).astype(np.int)]
    plt.plot(np.sort(mp_size), prob[np.argsort(mp_size)])
    prob /= np.sum(prob)
    assert prob.min() > 0., 'please choose proper number of bins.'
    plt.show()
    if fig_save_path is not None:
        fig.savefig(fig_save_path)
    plt.close(fig)
    if prob_save_path is not None:
        np.save(prob_save_path, prob)
    print('finish')
    return prob


if __name__ == '__main__':
    p_min, p_max = 0., 235.135148
    v_min, v_max = 0., 906.86444687
    p = np.load(r'../power.npy')
    v = np.load(r'../velocity.npy')
    p[p > p_max] = p_max
    v[v > v_max] = v_max
    # normalization
    p = (p - p_min) / (p_max - p_min)
    v = (v - v_min) / (v_max - v_min)
    e = (p + 1) / (v + 1)
    prob = kernel_dens_estimate(e)
    np.save('../energy_density_prob.npy', prob)
