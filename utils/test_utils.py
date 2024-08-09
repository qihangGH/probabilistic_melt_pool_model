import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from scipy.stats import t, norm

from MPModels.mp_models import transform_to_StudentT, transform_to_Gaussian

tfd = tfp.distributions
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 18


def test_with_bayesian_stu(test_model, test_data, batch_size, infer_times, verbose=True, visualize=True):
    loss_fn = lambda y, p_y: -p_y.log_prob(y)
    record_test_nnl = []
    record_test_mean = []
    record_test_beta = []
    record_test_a = []
    record_test_b = []
    record_prior_log_prob = []
    record_posterior_log_prob = []
    random_seeds = np.arange(100 * infer_times)
    np.random.shuffle(random_seeds)
    if verbose:
        print('Perform {} times inference ...'.format(infer_times))
        progress_bar = tqdm.tqdm(range(infer_times))
    else:
        progress_bar = range(infer_times)
    for infer_time in progress_bar:
        params_test = []
        for j in range(len(test_data['X']) // batch_size + 1):
            if len(test_data['X'][j * batch_size:(j + 1) * batch_size]) > 0:
                # Make sure that the batches in the same iteration use the same weights
                tf.random.set_seed(random_seeds[infer_time])
                temp_params_test = test_model.layers[-2](
                    [test_data['X'][j * batch_size:(j + 1) * batch_size],
                     test_data['n'][j * batch_size:(j + 1) * batch_size]],
                    training=False
                )
                params_test.append(temp_params_test)
                # the sampled weights are the same regardless of `j`, so calculating it once is enough
                if j == 0:
                    temp_prior_log_prob = []
                    temp_posterior_log_prob = []
                    collect_prob_recursively(test_model, temp_prior_log_prob, temp_posterior_log_prob)
                    record_prior_log_prob.append(sum(temp_prior_log_prob))
                    record_posterior_log_prob.append(sum(temp_posterior_log_prob))

        params_test = tf.concat(params_test, axis=0)
        record_test_mean.append(params_test[..., 0])
        record_test_beta.append(params_test[..., 1])
        record_test_a.append(params_test[..., 2])
        record_test_b.append(params_test[..., 3])

        y_test_pred = tfp.layers.DistributionLambda(transform_to_StudentT)(params_test)
        if 'y' in test_data.keys():
            test_nnl = loss_fn(test_data['y'], y_test_pred)
            record_test_nnl.append(test_nnl.numpy())

    record_test_mean = tf.stack(record_test_mean).numpy()
    record_test_beta = tf.stack(record_test_beta).numpy()
    record_test_a = tf.stack(record_test_a).numpy()
    record_test_b = tf.stack(record_test_b).numpy()

    weights = record_test_a > 1.001
    if verbose:
        print(np.sum(np.sum(weights, axis=0) == 0))
    weights[:, np.sum(weights, axis=0) == 0] = 1.
    # weights = Nones
    m_star = np.average(record_test_mean, axis=0, weights=weights)
    aleatoric = np.sqrt(
        np.average(record_test_b / (record_test_a - 1) * (1 / record_test_beta + 1), axis=0, weights=weights)
    )
    epistemic = np.sqrt(np.average(record_test_mean ** 2, axis=0, weights=weights) - m_star ** 2)
    s_star = np.sqrt(
        np.average(record_test_mean ** 2 + record_test_b / (record_test_a - 1) * (1 / record_test_beta + 1),
                   axis=0, weights=weights) - m_star ** 2)
    # s_star = np.sqrt(aleatoric ** 2 + epistemic ** 2)
    # the 99% confidence interval
    ci = calc_confidence_interval(record_test_mean.T, 0.995)

    results = {'m_star': m_star, 's_star': s_star, 'aleatoric': aleatoric, 'epistemic': epistemic,
               'm': record_test_mean, 'beta': record_test_beta,
               'a': record_test_a, 'b': record_test_b, 'ci': ci}
    if 'y' in test_data.keys():
        valid_idx = np.where(test_data['y'] > 0)[0]
        m_star, test_data['y'], s_star = \
        m_star[valid_idx], test_data['y'][valid_idx], s_star[valid_idx]
        mean_test_mse = ((m_star - test_data['y']) ** 2).mean()
        record_test_nnl = np.array(record_test_nnl)
        per_test_nnl = np.mean(record_test_nnl, axis=0)
        log_prob_per_iter = -np.sum(record_test_nnl, axis=-1)
        record_prior_log_prob = np.array(record_prior_log_prob)
        record_posterior_log_prob = np.array(record_posterior_log_prob)
        log_evidence_per_iter = record_prior_log_prob + log_prob_per_iter - record_posterior_log_prob
        mean_test_nnl = record_test_nnl.mean()
        per_sample_relative_error = np.abs(m_star - test_data['y']) / test_data['y']
        mean_test_relative_error = per_sample_relative_error.mean() * 100.
        within_std_per = np.array([
            np.sum([((m_star - n_std * s_star) < test_data['y']) &
                    (test_data['y'] < (m_star + n_std * s_star))]) / len(m_star) for n_std in range(1, 4)])
        n_points = 100
        start = 0
        end = 4
        std_range = np.linspace(start, end, n_points)
        within_std_per_all = np.array([
            np.sum([((m_star - n_std * s_star) < test_data['y']) &
                    (test_data['y'] < (m_star + n_std * s_star))]) / len(m_star) for n_std in std_range])
        within_std_gau = 2 * norm.cdf(std_range) - 1
        ece = np.sum((within_std_per_all - within_std_gau) * (end - start) / n_points)
        if visualize:
            plt.plot(std_range, within_std_gau, 'r')
            plt.plot(std_range, within_std_per_all, 'b--')
            plt.legend(['Gaussian', 'Predicted'])
            plt.show()
            plt.plot(within_std_gau, within_std_per_all, 'b--')
            plt.plot(within_std_gau, within_std_gau, 'r')
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            # plt.axis('equal')
            plt.show()

        results['mean_test_mse'] = mean_test_mse
        results['mean_test_abe'] = np.mean(np.abs(m_star - test_data['y']))
        results['nnl'] = per_test_nnl
        results['mean_test_nnl'] = mean_test_nnl
        results['mean_test_relative_error'] = mean_test_relative_error
        results['within_std_per'] = within_std_per
        results['top_100_relative_error'] = np.sort(per_sample_relative_error)[-100:].mean() * 100
        results['per_sample_relative_error'] = per_sample_relative_error
        results['log_prob_per_iter'] = log_prob_per_iter
        results['prior_log_prob_per_iter'] = record_prior_log_prob
        results['posterior_log_prob_per_iter'] = record_posterior_log_prob
        results['log_evidence_per_iter'] = log_evidence_per_iter
        results['within_std_per_all'] = within_std_per_all
        results['within_std_gau'] = within_std_gau
        results['ece'] = ece
    return results


def test_with_bayesian_gaussian(test_model, test_data, batch_size, infer_times, verbose=True, visualize=True):
    loss_fn = lambda y, p_y: -p_y.log_prob(y)
    record_test_nnl = []
    record_test_mean = []
    record_test_std = []
    record_prior_log_prob = []
    record_posterior_log_prob = []
    random_seeds = np.arange(100 * infer_times)
    np.random.shuffle(random_seeds)
    if verbose:
        print('Perform {} times inference ...'.format(infer_times))
        progress_bar = tqdm.tqdm(range(infer_times))
    else:
        progress_bar = range(infer_times)
    for infer_time in progress_bar:
        params_test = []
        for j in range(len(test_data['X']) // batch_size + 1):
            # Make sure that the batches in the same iteration use the same weights
            tf.random.set_seed(random_seeds[infer_time])
            if len(test_data['X'][j * batch_size:(j + 1) * batch_size]) > 0:
                temp_params_test = test_model.layers[-2](
                    [test_data['X'][j * batch_size:(j + 1) * batch_size],
                     test_data['n'][j * batch_size:(j + 1) * batch_size]],
                    training=False
                )
                params_test.append(temp_params_test)
                # the sampled weights are the same regardless of `j`, so calculating it once is enough
                if j == 0:
                    temp_prior_log_prob = []
                    temp_posterior_log_prob = []
                    collect_prob_recursively(test_model, temp_prior_log_prob, temp_posterior_log_prob)
                    record_prior_log_prob.append(sum(temp_prior_log_prob))
                    record_posterior_log_prob.append(sum(temp_posterior_log_prob))

        params_test = tf.concat(params_test, axis=0)

        record_test_mean.append(params_test[..., 0])
        record_test_std.append(params_test[..., 1])

        y_test_pred = tfp.layers.DistributionLambda(transform_to_Gaussian)(params_test)
        if 'y' in test_data.keys():
            test_nnl = loss_fn(test_data['y'], y_test_pred)
            record_test_nnl.append(test_nnl.numpy())

    record_test_mean = tf.stack(record_test_mean)
    record_test_std = tf.stack(record_test_std)

    m_star = tf.reduce_mean(record_test_mean, axis=0).numpy()
    aleatoric = tf.reduce_mean(record_test_std, axis=0).numpy()
    if infer_times == 1:
        epistemic = np.zeros_like(m_star)
        s_star = aleatoric.copy()
    else:
        epistemic = tf.sqrt(tf.reduce_mean(record_test_mean ** 2, axis=0) - m_star ** 2).numpy()
        # s_star = np.sqrt(aleatoric ** 2 + epistemic ** 2)
        s_star = np.sqrt(tf.reduce_mean(record_test_std ** 2 + record_test_mean ** 2, axis=0).numpy() - m_star ** 2)

    # the 99% confidence interval
    ci = calc_confidence_interval(record_test_mean.numpy().T, 0.995)

    results = {'m_star': m_star, 's_star': s_star, 'aleatoric': aleatoric, 'epistemic': epistemic,
               'mu': record_test_mean.numpy(), 'sigma': record_test_std.numpy(), 'ci': ci}
    if 'y' in test_data.keys():
        valid_idx = np.where(test_data['y'] > 0)[0]
        m_star, test_data['y'], s_star = \
        m_star[valid_idx], test_data['y'][valid_idx], s_star[valid_idx]
        mean_test_mse = ((m_star - test_data['y']) ** 2).mean()
        record_test_nnl = np.array(record_test_nnl)
        per_test_nnl = np.mean(record_test_nnl, axis=0)
        mean_test_nnl = record_test_nnl.mean()
        log_prob_per_iter = -np.sum(record_test_nnl, axis=-1)
        record_prior_log_prob = np.array(record_prior_log_prob)
        # TODO:
        # record_posterior_log_prob = np.array(record_prior_log_prob)
        record_posterior_log_prob = np.array(record_posterior_log_prob)
        log_evidence_per_iter = record_prior_log_prob + log_prob_per_iter - record_posterior_log_prob
        per_sample_relative_error = np.abs(m_star - test_data['y']) / test_data['y']
        mean_test_relative_error = per_sample_relative_error.mean() * 100.
        within_std_per = np.array([
            np.sum([((m_star - n_std * s_star) < test_data['y']) &
                    (test_data['y'] < (m_star + n_std * s_star))]) / len(m_star) for n_std in range(1, 4)])

        n_points = 100
        start = 0
        end = 4
        std_range = np.linspace(start, end, n_points)
        within_std_per_all = np.array([
            np.sum([((m_star - n_std * s_star) < test_data['y']) &
                    (test_data['y'] < (m_star + n_std * s_star))]) / len(m_star) for n_std in std_range])
        within_std_gau = 2 * norm.cdf(std_range) - 1
        ece = np.sum((within_std_per_all - within_std_gau) * (end - start) / n_points)
        if visualize:
            plt.plot(std_range, within_std_gau, 'r')
            plt.plot(std_range, within_std_per_all, 'b--')
            plt.legend(['Gaussian', 'Predicted'])
            plt.show()
            plt.plot(within_std_gau, within_std_per_all, 'b--')
            plt.plot(within_std_gau, within_std_gau, 'r')
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            # plt.axis('equal')
            plt.show()

        results['mean_test_mse'] = mean_test_mse
        results['mean_test_abe'] = np.mean(np.abs(m_star - test_data['y']))
        results['mean_test_nnl'] = mean_test_nnl
        results['nnl'] = per_test_nnl
        results['mean_test_relative_error'] = mean_test_relative_error
        results['within_std_per'] = within_std_per
        results['top_100_relative_error'] = np.sort(per_sample_relative_error)[-100:].mean() * 100
        results['per_sample_relative_error'] = per_sample_relative_error
        results['log_prob_per_iter'] = log_prob_per_iter
        results['prior_log_prob_per_iter'] = record_prior_log_prob
        results['posterior_log_prob_per_iter'] = record_posterior_log_prob
        results['log_evidence_per_iter'] = log_evidence_per_iter
        results['within_std_per_all'] = within_std_per_all
        results['within_std_gau'] = within_std_gau
        results['ece'] = ece
    return results


def test_with_non_bayesian(model, test_data, batch_size, verbose=1):
    results = model.predict([test_data['X'], test_data['n']], batch_size=batch_size, verbose=verbose)
    # results = np.squeeze(model([test_data['X'], test_data['n']]).numpy())
    results = np.squeeze(results)
    test_results = {'m_star': results}
    if 'y' in test_data:
        per_sample_relative_error = np.abs(results - test_data['y']) / test_data['y']
        test_results['mean_test_mse'] = np.mean((results - test_data['y']) ** 2)
        test_results['mean_test_abe'] = np.mean(np.abs(results - test_data['y']))
        test_results['mean_test_relative_error'] = per_sample_relative_error.mean() * 100
        test_results['top_100_relative_error'] = np.sort(per_sample_relative_error)[-100:].mean() * 100
        test_results['per_sample_relative_error'] = per_sample_relative_error
    return test_results


def collect_prob_recursively(model_or_layer, prior_log_prob, posterior_log_prob):
    if isinstance(model_or_layer, tf.keras.Model):
        for m_or_l in model_or_layer.layers:
            collect_prob_recursively(m_or_l, prior_log_prob, posterior_log_prob)
    else:
        if hasattr(model_or_layer, 'prior_log_prob'):
            assert hasattr(model_or_layer, 'posterior_log_prob')
            prior_log_prob.append(model_or_layer.prior_log_prob.numpy())
            posterior_log_prob.append(model_or_layer.posterior_log_prob.numpy())


def calc_confidence_interval(samples, q):
    # `samples` is a ndarray of shape (num_variables, num_samples)
    # `q` is stands for quantile, q = 0.995 accords to the 99% confidence interval
    # sample_mean = np.mean(samples, axis=-1)
    sample_std = np.std(samples, axis=-1)
    t_star = t.ppf(q, samples.shape[-1] - 1)
    return t_star * sample_std / np.sqrt(samples.shape[-1])
