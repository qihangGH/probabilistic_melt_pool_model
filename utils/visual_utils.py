import os
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from scipy.stats import norm, t, gamma
from scipy.signal import medfilt
from matplotlib import ticker

from utils.dataload_utils import load_layer_data
from utils.test_utils import test_with_non_bayesian, test_with_bayesian_gaussian, test_with_bayesian_stu
from denoising.data_denoising import find_neighbors_of_neighbor_maps

mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 18


def predict_a_layer(model, layer, test_data, model_name, save_dir=None, infer_times=100,
                    figsize=None, plot_type='plot', apply_medfilt=None, apply_trunc=None):
    # data = load_mp_data(layer, crop_size, filter_len)
    # test_data = {'X': np.hstack([data['power'].reshape(-1, 1), data['velocity'].reshape(-1, 1)]),
    #              'n': data['n'], 'y': data['mp_size']}
    data = {'melt_x': test_data['melt_x'], 'melt_y': test_data['melt_y'], 'mp_size': test_data['y']}
    if model_name == 'student':
        results = test_with_bayesian_stu(model, test_data, batch_size=
        # len(test_data['y']),
        5000,
                                         infer_times=infer_times)
    elif model_name == 'gaussian':
        results = test_with_bayesian_gaussian(model, test_data, batch_size=
        # len(test_data['y']),
        5000,
                                              infer_times=infer_times)
    elif model_name == 'non_bayesian':
        results = test_with_non_bayesian(model, test_data, batch_size=len(test_data['y']))
    else:
        raise ValueError('{} has not been implemented.'.format(model_name))
    for k, v in results.items():
        if k in ['mean_test_mse', 'mean_test_abe', 'mean_test_nnl', 'mean_test_relative_error',
                 'within_std_per', 'top_100_relative_error']:
            print(r'{}: {}'.format(k, v))
    max_value = max(results['m_star'].max(), data['mp_size'].max())
    plot_func = plot_a_layer if plot_type == 'plot' else scatter_a_layer
    plot_func(data['mp_size'], data['melt_x'], data['melt_y'],
              os.path.join(save_dir, 'Melt pool size of layer ' + repr(layer) + '.png'),
              min_value=0., max_value=max_value, figsize=figsize, title='Ground truth')
    plot_func(results['m_star'], data['melt_x'], data['melt_y'],
              os.path.join(save_dir, 'Predict layer ' + repr(layer) + ' with ' + model_name + '.png'),
              min_value=0., max_value=max_value, figsize=figsize, title='Prediction')
    if 's_star' in results.keys():
        # normalize = colors.TwoSlopeNorm(vmin=min_value, vcenter=300., vmax=max_value)
        if apply_medfilt is not None:
            v_s_star = medfilt(results['s_star'], apply_medfilt)
            v_a = medfilt(results['aleatoric'], apply_medfilt)
            v_e = medfilt(results['epistemic'], apply_medfilt)
        else:
            v_s_star = results['s_star']
            v_a = results['aleatoric']
            v_e = results['epistemic']
        ind = np.arange(len(results['s_star']))
        if apply_trunc is not None:
            print(f'Max before trunc:', results['aleatoric'].max())
            ind = ind[results['aleatoric'] < apply_trunc]
        # min_value = min(results['aleatoric'].min(), results['epistemic'].min(), 4.22)
        min_value = min(v_e.min(), v_a.min())
        # max_value = 144.17
        max_value = v_s_star[ind].max()
        plot_func(v_s_star[ind], data['melt_x'][ind], data['melt_y'][ind],
                  os.path.join(save_dir, 'Predict std ' + repr(layer) + ' with ' + model_name + '.png'),
                  max_value=max_value, min_value=min_value, figsize=figsize, title='Uncertainty')
        plot_func(v_a[ind], data['melt_x'][ind], data['melt_y'][ind],
                  os.path.join(save_dir, 'Predict aleatoric ' + repr(layer) + ' with ' + model_name + '.png'),
                  max_value=max_value, min_value=min_value, figsize=figsize, title='Aleatoric')
        plot_func(v_e[ind], data['melt_x'][ind], data['melt_y'][ind],
                  os.path.join(save_dir, 'Predict epistemic ' + repr(layer) + ' with ' + model_name + '.png'),
                  max_value=results['epistemic'].max(), min_value=min_value, figsize=figsize, title='Epistemic')
    plot_ordered_relative_error(test_data['y'], results['per_sample_relative_error'],
                                os.path.join(save_dir, r'ordered_relative_error.png'))


def scatter_a_layer(mp_size, x, y, fig_name=None, min_value=None, max_value=None, cmap='viridis', figsize=None, title=None):
    plt.figure(figsize=figsize)
    plt.scatter(x, y, c=mp_size, vmin=min_value, vmax=max_value, cmap=cmap)
    if title is not None:
        plt.title(title)
    plt.colorbar(), plt.axis('equal'), plt.axis('off'), plt.show()


def plot_two_patch(data1, data2, fig_name=None):
    vmax = [234.871234, 904.5837454320892, 1.]
    vmin = [0, 0., 0.]
    if np.max(data1) <= 1:
        vmax = [1., 1., 1.]
    fig = plt.figure()
    for i in range(1, 7):
        plt.subplot(3, 2, i)
        if i % 2 == 1:
            plt.imshow(data1[..., (i - 1) // 2], cmap='gray', vmax=vmax[(i - 1) // 2], vmin=vmin[(i - 1) // 2])
        else:
            plt.imshow(data2[..., (i - 1) // 2], cmap='gray', vmax=vmax[(i - 1) // 2], vmin=vmin[(i - 1) // 2])
    plt.show()
    if fig_name is not None:
        fig.savefig(fig_name)
    plt.close(fig)


def plot_a_layer(mp_size, x, y, fig_name=None, log_mode=False, normalize=colors.Normalize(),
                 cmap='viridis', figsize=None, title=None, **kwargs):
    fig = plt.figure(figsize=figsize)
    if 'max_value' in kwargs.keys():
        assert 'min_value' in kwargs.keys(), 'Please also specify the minimum value.'
        vmin, vmax = kwargs['min_value'], kwargs['max_value']
        # if log_mode is True:
        #     vmin, vmax = 0, np.log(vmax - vmin + 1)
        #     if 'scale' in kwargs.keys():
        #         vmax *= kwargs['scale']
        cs = plt.tricontourf(x, y, mp_size, vmin=vmin, vmax=vmax, levels=500, cmap=cmap, norm=normalize)
        # cs = plt.tricontourf(x, y, mp_size, vmin=vmin, vmax=vmax, norm=normalize)
        # cs = plt.tricontourf(x, y, mp_size, levels=500, norm=normalize)
        # plt.tricontour(x, y, mp_size)
    else:
        cs = plt.tricontourf(x, y, mp_size, levels=100, cmap=cmap, norm=normalize)
        # plt.tricontour(x, y, mp_size)
    plt.colorbar(cs)
    if 'marked_points' in kwargs:
        # plt.scatter(kwargs['marked_points'][:, 0], kwargs['marked_points'][:, 1], c='r', s=1, alpha=0.5)
        plt.plot(kwargs['marked_points'][:, 0], kwargs['marked_points'][:, 1], c='r', linewidth=1, alpha=0.6)
    plt.axis('equal')
    plt.axis('off')
    if fig_name is not None:
        fig.savefig(fig_name)
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close(fig)


def plot_a_patch(patch, fig_name=None, title=None, show=True):
    fig = plt.figure()
    vmax = [235.135148, 906.86444687, 1.]
    vmin = [0., 0., 0.]
    if np.max(patch) <= 1:
        vmax = [1., 1., 1.]
    for i in range(1, patch.shape[-1] + 1):
        plt.subplot(1, patch.shape[-1], i)
        plt.axis('off')
        plt.imshow(patch[..., i - 1], cmap='gray', vmax=vmax[i - 1], vmin=vmin[i - 1])
    if title is not None:
        plt.title(title)
    if show is True:
        plt.show()
    if fig_name is not None:
        fig.savefig(fig_name)
    plt.close(fig)


def plot_hist(data, fig_name=None):
    fig = plt.figure()
    plt.hist(data)
    plt.show()
    if fig_name is not None:
        fig.savefig(fig_name)
    plt.close(fig)


def plot_hist_of_power_and_mp_size(power, mp_size, bins=100, **kwargs):
    fig = plt.figure()
    plt.hist2d(x=power, y=mp_size, bins=bins)
    plt.title('histogram of power and melt pool size')
    plt.show()
    if 'save_path' in kwargs.keys():
        fig.savefig(kwargs['save_path'])
    plt.close(fig)


def plot_hist_of_velocity_and_mp_size(velocity, mp_size, bins=100, **kwargs):
    fig = plt.figure()
    plt.hist2d(x=velocity, y=mp_size, bins=bins)
    plt.title('histogram of velocity and melt pool size')
    plt.show()
    if 'save_path' in kwargs.keys():
        fig.savefig(kwargs['save_path'])
    plt.close(fig)


def plot_ordered_relative_error(mp_size, error, fig_name=None):
    error = error[np.argsort(mp_size)]
    mp_size.sort()
    fig = plt.figure()
    plt.plot(mp_size, error)
    plt.show()
    if fig_name is not None:
        fig.savefig(fig_name)
    plt.close(fig)


def plot_gaussian(loc, scale, x_range=None):
    if x_range is None:
        x_range = np.linspace(loc - 4 * scale, loc + 4 * scale, 500)
    pdf = norm.pdf(x_range, loc, scale)
    plt.plot(x_range, pdf)
    plt.show()


def plot_student(df, loc, scale, x_range=None):
    if x_range is None:
        x_range = np.linspace(loc - 4 * scale, loc + 4 * scale, 500)
    pdf = t.pdf(x_range, df, loc, scale)
    plt.plot(x_range, pdf)
    plt.show()


def plot_normal_gamma(m, beta, a, b, mu_range=None, lamb_range=None, cmap='Blues', levels=20):
    assert beta > 0 and a > 1 and b > 0
    if mu_range is None:
        mu_range = np.linspace(m - 4 * np.sqrt(b / (a - 1) / beta), m + 4 * np.sqrt(b / (a - 1) / beta), 500)
    if lamb_range is None:
        lamb_range = np.linspace(1e-3, 5 * a / b ** 2, 500)
    mu, lamb = np.meshgrid(mu_range, lamb_range)
    mu, lamb = mu.reshape(-1), lamb.reshape(-1)
    marginal_lamb = gamma.pdf(lamb_range, a, 0, 1 / b)
    marginal_mu = t.pdf(mu_range, 2 * a, m, np.sqrt(b / a / beta))
    pdf = norm.pdf(mu, m, np.sqrt(1 / beta / lamb)) * gamma.pdf(lamb, a, 0, 1 / b)

    # definitions for the axes
    left, width = 0.1, 0.5
    bottom, height = 0.2, 0.5
    bar_width = 0.16666
    spacing = 0.005

    rect_scatter = [left, bottom, width + bar_width, height]
    rect_x = [left + bar_width, bottom + height + spacing, width, 0.12]
    rect_y = [left + width + bar_width + spacing, bottom, 0.12, height]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_axes(rect_scatter)
    ax_x = fig.add_axes(rect_x, sharex=ax)
    ax_y = fig.add_axes(rect_y, sharey=ax)

    ax_x.tick_params(axis="x", labelbottom=False)
    ax_y.tick_params(axis="y", labelleft=False)
    # contour = ax.contourf(mu_range, lamb_range, pdf.reshape([len(mu_range), len(lamb_range)]),
    #                       levels=levels, cmap=cmap)
    contour = ax.tricontourf(mu, lamb, pdf, levels=levels, cmap=cmap)
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$\lambda$')
    plt.colorbar(contour, ax=ax, location='left')
    ax_x.plot(mu_range, marginal_mu)
    ax_y.plot(marginal_lamb, lamb_range)
    plt.show()


def gaussian(x, mu, sigma):
    assert sigma > 0
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu) ** 2 / 2 * sigma ** 2)


def normal_gamma(mu, lamb, m, beta, a, b):
    return norm.pdf(mu, m, np.sqrt(1 / beta / lamb)) * gamma.pdf(mu, a, 0, 1 / b)


def plot_mean_with_std(x, mean, std, n=1, save_path=None):
    fig = plt.figure()
    plt.plot(x, mean, 'r', linewidth=2)
    plt.plot(x, mean + n * std, 'g', linewidth=1)
    plt.plot(x, mean - n * std, 'g', linewidth=1)
    plt.show()
    if save_path is not None:
        fig.savefig(save_path)
    plt.close(fig)


def predict_a_patch_with_varying_powers(model, n_test, v_test, powers, inference_times=100):
    X_test = np.stack([powers, np.ones_like(powers) * v_test], axis=-1)
    n_test = np.stack([n_test for _ in range(len(X_test))], axis=0)
    results = test_with_bayesian_stu(model, {'X': X_test, 'n': n_test},
                                     batch_size=len(X_test), infer_times=inference_times)
    plot_mean_with_std(powers, results['m_star'], results['s_star'])
    return {'m_star': results['m_star'], 's_star': results['s_star']}


def predict_a_patch_with_varying_velocities(model, n_test, p_test, velocities, inference_times=100):
    X_test = np.stack([np.ones_like(velocities) * p_test, velocities], axis=-1)
    n_test = np.stack([n_test for _ in range(len(X_test))], axis=0)
    results = test_with_bayesian_stu(model, {'X': X_test, 'n': n_test},
                                     batch_size=len(X_test), infer_times=inference_times)
    plot_mean_with_std(velocities, results['m_star'], results['s_star'])
    return {'m_star': results['m_star'], 's_star': results['s_star']}


def predict_a_patch_with_varying_powers_and_velocities(
        model, n_test, powers, velocities, inference_times=100, levels=20, cmap='viridis'
):
    p_min, p_max = 0., 235.135148
    v_min, v_max = 0., 906.86444687
    pp, vv = np.meshgrid(powers, velocities)
    p, v = pp.reshape([-1]), vv.reshape([-1])
    X_test = np.stack([p, v], axis=-1)
    n_test = np.stack([n_test for _ in range(len(X_test))], axis=0)
    batch_size = 10000 if len(p) > 10000 else len(p)
    results = test_with_bayesian_stu(model, {'X': X_test, 'n': n_test},
                                     batch_size=batch_size, infer_times=inference_times)
    m_star = results['m_star'].reshape([len(powers), len(velocities)])
    s_star = results['s_star'].reshape([len(powers), len(velocities)])
    # print(powers.shape, velocities.shape, m_star.shape)
    plt.tricontourf(p * p_max, v * v_max, results['m_star'], levels=levels, cmap=cmap)
    # , vmin=100, vmax=450)
    plt.colorbar()
    plt.axis('equal')
    plt.xticks([50 + i * 100 for i in range(4)])
    plt.yticks([50 + i * 100 for i in range(9)])
    plt.xlim([50, 400])
    plt.ylim([50, 900])
    plt.gca().set_adjustable('box')
    plt.show()
    plt.tricontourf(p * p_max, v * v_max, results['s_star'], levels=levels, cmap=cmap)
    # , vmin=15, vmax=110)
    print(r'mean range: [{}, {}]'.format(results['m_star'].min(), results['m_star'].max()))
    print(r'std range: [{}, {}]'.format(results['s_star'].min(), results['s_star'].max()))
    plt.colorbar()
    plt.axis('equal')
    plt.xticks([50 + i * 100 for i in range(4)])
    plt.yticks([50 + i * 100 for i in range(9)])
    plt.gca().set_adjustable('box')
    plt.show()
    return {'m_star': results['m_star'], 's_star': results['s_star']}


def plot_a_layer_with_all_melted_points(x, y, z=None, marked_points=None, save_path=None):
    fig = plt.figure(figsize=[10, 10])
    if z is None:
        z = np.ones_like(x)
    plt.scatter(x, y, c=z, s=0.5)
    if marked_points is not None:
        plt.scatter(marked_points[:, 0], marked_points[:, 1], c='r', s=4)
    plt.axis('equal')
    plt.show()
    if save_path is not None:
        fig.savefig(save_path)
    plt.close(fig)


def plot_two_lines_with_same_scale(x, y1, y2, name1=None, name2=None):
    fig = plt.figure()
    plt.plot(x, y1, 'r--')
    plt.plot(x, y2, 'b')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if name1 is not None:
        plt.legend([name1, name2])
    plt.show()


def plot_two_lines_with_diff_scales(x, y1, y2, name1=None, name2=None):
    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()

    # Plot the first data series on ax1
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    if name1 is not None:
        ax1.set_ylabel(name1, color=color)
    ax1.plot(x, y1, 'r--')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for the second data series
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    if name2 is not None:
        ax2.set_ylabel(name2, color=color)
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Show the plot
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


if __name__ == '__main__':
    # same testing code, just ignore it
    img = np.random.random([20, 20])
    plt.imshow(img)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=20)
    # text = cb.ax.yaxis.label
    # font = mpl.font_manager.FontProperties(size=32)
    # text.set_font_properties(font)
    plt.show()
