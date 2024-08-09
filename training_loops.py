import time
import datetime
import multiprocessing
import warnings
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from functools import partial

import tqdm
from tensorflow import keras as tfk


tfm = tf.math
tfd = tfp.distributions
tfpl = tfp.layers


def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                          for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) + metrics, end=end)


def training_mp_models(
        model_name,
        X_train,
        y_train,
        n_train,
        sample_weights_train,
        X_valid,
        y_valid,
        n_valid,
        sample_weights_valid,
        model,
        epochs,
        batch_size,
        filepath,
        optimizer,
        loss_fn,
        mse_weight_schedule=lambda t: 0,
        num_bayesian_inference=1,
        val_batch_size=None,
        evi_reg_coef=0.,
        **kwargs
):
    """
    :param model_name: 'student' or 'gaussian'
    :param X_train: Input training data of local scan parameters
    :param y_train: Training labels
    :param n_train: Input training data of neighboring maps
    :param sample_weights_train: Weights for weighting different training samples
    :param X_valid: Input validation data of local scan parameters
    :param y_valid: Validation labels
    :param n_valid: Input validation data of neighboring maps
    :param sample_weights_valid: Weights for weighting different validation samples
    :param model: Model to be trained
    :param epochs: Number of training epochs
    :param batch_size: Number of batch size
    :param filepath: Filepath to save model weights
    :param optimizer: Optimizer
    :param loss_fn: Loss function
    :param mse_weight_schedule: The weight for additional MSE loss, which is a function of epoch
    :param num_bayesian_inference: Inference times for validation
    :param val_batch_size: Validation batch size
    :param evi_reg_coef: For regularization
    :return: None
    """
    if model_name == 'student':
        transform_fn = transform_to_StudentT
        # m, beta, a, b
        cal_std = calc_student_std
    elif model_name == 'gaussian':
        transform_fn = lambda t: tfd.Normal(loc=t[..., 0], scale=t[..., 1])
        cal_std = calc_gaussian_std
    else:
        raise ValueError('`model_name` should be \'student\' or \'gaussian\'')
    if sample_weights_train is not None:
        # assert np.sum(sample_weights_train) == 1, '`sample_weights_train` must sum to 1.'
        sample_weights_train = sample_weights_train.astype(np.float32)
    if sample_weights_valid is not None:
        # assert np.sum(sample_weights_valid) == 1, '`sample_weights_valid` must sum to 1.'
        sample_weights_valid = sample_weights_valid.astype(np.float32)
    print('Please make sure that `kl_weight` is set to 1 / len(X_train)')
    if val_batch_size is None:
        val_batch_size = batch_size
    X_train, y_train, n_train = X_train.astype(np.float32), y_train.astype(np.float32), n_train.astype(np.float32)
    if X_valid is not None:
        X_valid, y_valid, n_valid = X_valid.astype(np.float32), y_valid.astype(np.float32), n_valid.astype(np.float32)
    mean_loss = tfk.metrics.Mean(name='loss')
    mean_mse = tfk.metrics.Mean(name='mse')
    mean_rel_error = tfk.metrics.Mean(name='rel_error')
    if model_name == 'student' and evi_reg_coef > 0:
        mean_evi_loss = tfk.metrics.Mean(name='evi')
    steps = len(y_train) // batch_size
    # Record
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/' + filepath + '/' + current_time + '/train'
    test_log_dir = 'logs/' + filepath + '/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    best_val_mse = np.inf
    best_val_loss = np.inf
    best_val_rel_error = np.inf
    for i in range(1, epochs + 1):
        print("Epoch: {}/{}".format(i, epochs))
        # Shuffle the data
        ids = np.arange(len(y_train))
        np.random.shuffle(ids)
        for j in range(1, steps + 2):
            if j == steps + 1:
                if len(y_train) > batch_size * steps:
                    batch_ids = ids[steps * batch_size:]
                else:
                    break
            else:
                batch_ids = ids[(j - 1) * batch_size:j * batch_size]
            with tf.GradientTape() as tape:
                # Forward propagation, `training` must be set as True.
                # y_pred = model([X_train[batch_ids], n_train[batch_ids]], training=True)
                params_pred = model.layers[-2](
                    [X_train[batch_ids], n_train[batch_ids]],
                    training=True
                )
                y_pred = tfpl.DistributionLambda(transform_fn)(params_pred)  # [B, 1]

                # mse
                train_mse_per_sample = (params_pred[..., 0] - y_train[batch_ids]) ** 2
                main_loss = loss_fn(y_train[batch_ids], y_pred)
                # main_loss = tf.squeeze(loss_fn(y_train[batch_ids][..., None], y_pred))
                if sample_weights_train is not None:
                    main_loss = tf.reduce_sum(sample_weights_train[batch_ids] * main_loss) / \
                                tf.reduce_sum(sample_weights_train[batch_ids])
                    main_mse_loss = tf.reduce_sum(sample_weights_train[batch_ids] * train_mse_per_sample) / \
                                    tf.reduce_sum(sample_weights_train[batch_ids])
                else:
                    main_loss = tf.reduce_mean(main_loss)
                    main_mse_loss = tf.reduce_mean(train_mse_per_sample)
                if model_name == 'student' and evi_reg_coef > 0:
                    # |m - y_true| * (2*beta + a) [m, beta, a, b]
                    evi_loss = tf.reduce_mean(
                        tf.abs(params_pred[..., 0] - y_train[batch_ids]) * (2 * params_pred[..., 1] + params_pred[..., 2]))
                    main_loss += evi_reg_coef * evi_loss
                main_rel_error = tf.reduce_mean(tf.abs(params_pred[..., 0] - y_train[batch_ids]) /
                                                y_train[batch_ids]) * 100
                train_loss = tf.add_n([main_loss, np.array(mse_weight_schedule(i), dtype=np.float32) * main_mse_loss]
                                      + model.losses)  # `tf.add_n` can add a list of tensors
            # Calculate the gradients for the trainable variables
            gradients = tape.gradient(train_loss, model.trainable_variables)
            # Apply the gradients to the corresponding variables and update their values.
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # Update the mean loss of this epoch.
            mean_loss.update_state([train_loss] * len(batch_ids))
            mean_mse.update_state([main_mse_loss] * len(batch_ids))
            mean_rel_error.update_state([main_rel_error] * len(batch_ids))
            if model_name == 'student' and evi_reg_coef > 0:
                mean_evi_loss.update_state([evi_loss] * len(batch_ids))
            if j < steps + 1:
                print_list = [mean_mse, mean_rel_error]
                if model_name == 'student' and evi_reg_coef > 0:
                    print_list.append(mean_evi_loss)
                print_status_bar(j * batch_size, len(y_train), mean_loss, print_list)
        print_list = [mean_mse, mean_rel_error]
        if model_name == 'student' and evi_reg_coef > 0:
            print_list.append(mean_evi_loss)
        print_status_bar(len(y_train), len(y_train), mean_loss, print_list)
        with train_summary_writer.as_default():
            tf.summary.scalar(mean_loss.name, mean_loss.result(), step=i)
            tf.summary.scalar(mean_mse.name, mean_mse.result(), step=i)
            tf.summary.scalar(mean_rel_error.name, mean_rel_error.result(), step=i)

        reset_list = [mean_loss, mean_mse, mean_rel_error]
        if model_name == 'student' and evi_reg_coef > 0:
            reset_list.append(mean_evi_loss)
        for metric in reset_list:
            metric.reset_states()

        if X_valid is not None:
            record_val_loss = []
            record_val_mean = []
            record_val_params = None
            for k in range(num_bayesian_inference):
                # params_valid = model.layers[2]([X_valid, n_valid], training=False)
                params_valid = []
                for j in range(len(y_valid) // val_batch_size + 1):
                    if len(y_valid[j * val_batch_size:(j + 1) * val_batch_size]) > 0:
                        temp_params_valid = model.layers[-2](
                            [X_valid[j * val_batch_size:(j + 1) * val_batch_size],
                             n_valid[j * val_batch_size:(j + 1) * val_batch_size]],
                            training=False
                        )
                        params_valid.append(temp_params_valid)
                params_valid = tf.concat(params_valid, axis=0)  # [B, n_params]
                if record_val_params is None:
                    record_val_params = params_valid[None, ...]
                else:
                    # [n_infer, B, n_params]
                    record_val_params = tf.concat([record_val_params, params_valid[None, ...]], axis=0)
                record_val_mean.append(params_valid[..., 0])
                y_valid_pred = tfpl.DistributionLambda(transform_fn)(params_valid)
                valid_loss = loss_fn(y_valid, y_valid_pred)
                if sample_weights_valid is not None:
                    valid_loss = tf.reduce_sum(sample_weights_valid * valid_loss)
                else:
                    valid_loss = tf.reduce_mean(valid_loss)
                # The `kl_weight` should always be 1 / num_samples
                # Because the input `kl_weight` argument is 1 / len(X_train),
                # so scale it to 1 / len(X_valid) for validation here
                ratio = np.array(len(X_train) / len(X_valid), dtype=np.float32)
                compensated_losses = [ratio * losses for losses in model.losses]
                valid_loss = tf.add_n([valid_loss] + compensated_losses)
                record_val_loss.append(valid_loss.numpy())

            std = cal_std(record_val_params)
            if model_name == 'student' and evi_reg_coef > 0:
                mean_val_params = tf.reduce_mean(record_val_params, axis=0)
                record_val_evi = \
                    tf.reduce_mean(tf.abs(mean_val_params[..., 0] - y_valid) * (2 * mean_val_params[..., 1] + mean_val_params[..., 2]))
            del record_val_params
            record_val_loss = np.array(record_val_loss)
            record_val_mean = tf.reduce_mean(tf.stack(record_val_mean), axis=0)
            record_val_mse = (record_val_mean - y_valid) ** 2
            val_relative_error = tf.abs(record_val_mean - y_valid) / y_valid * 100
            within_std_per = [tf.reduce_sum(
                tf.where((record_val_mean - n_std * std < y_valid) & (y_valid < record_val_mean + n_std * std), 1, 0)
            ) / len(std) * 100 for n_std in range(1, 4)]
            if sample_weights_valid is not None:
                mean_val_mse = tf.reduce_sum(sample_weights_valid * record_val_mse).numpy()
                mean_val_relative_error = tf.reduce_sum(sample_weights_valid * val_relative_error).numpy()
            else:
                mean_val_mse = tf.reduce_mean(record_val_mse).numpy()
                mean_val_relative_error = tf.reduce_mean(val_relative_error).numpy()
            # original_val_mse = tf.reduce_mean(record_val_mse).numpy()
            mean_val_loss = record_val_loss.mean()

            with test_summary_writer.as_default():
                tf.summary.scalar('val_loss', mean_val_loss, step=i)
                tf.summary.scalar('val_mse', mean_val_mse, step=i)
                tf.summary.scalar('val_relative_error', mean_val_relative_error, step=i)
                for j in range(1, 4):
                    tf.summary.scalar(f'within_std_{j}', within_std_per[j - 1], step=i)
            val_info = 'val_loss: {:.4f} - val_mse: {:.4f} - val_rel_error: {:.4f}'.format(
                mean_val_loss, mean_val_mse, mean_val_relative_error
            )
            if model_name == 'student' and evi_reg_coef > 0:
                val_info += f' - val_evi_error: {record_val_evi:.4f}'
            print(val_info)
            print('within_std_1: {:.4f} - within_std_2: {:.4f} - within_std_3: {:.4f}'.format(
                within_std_per[0], within_std_per[1], within_std_per[2]
            ))

            # Save the model when it improves
            if best_val_mse > mean_val_mse:
                print("\033[31mValidation mse improves from", best_val_mse, "to", mean_val_mse, "\033[m")
                best_val_mse = mean_val_mse
                model.save_weights(filepath + r'_mse.tf')
                np.save(filepath + r'_mse_opt_weights.npy', optimizer.get_weights())
            if best_val_loss > mean_val_loss:
                print("\033[31mValidation loss improves from", best_val_loss, "to", mean_val_loss, "\033[m")
                best_val_loss = mean_val_loss
                model.save_weights(filepath + r'_loss.tf')
                np.save(filepath + r'_loss_opt_weights.npy', optimizer.get_weights())
            if best_val_rel_error > mean_val_relative_error:
                print("\033[31mValidation relative error improves from", best_val_rel_error, "to", mean_val_relative_error, "\033[m")
                best_val_rel_error = mean_val_relative_error
                model.save_weights(filepath + r'_relative_error.tf')
                np.save(filepath + r'_relative_error_opt_weights.npy', optimizer.get_weights())

            print(f'Best val loss: {best_val_loss:.4f}, mse: {best_val_mse:.4f}, mre: {best_val_rel_error:.4f}')
        model.save_weights(filepath + r'.tf')
        np.save(filepath + r'_opt_weights.npy', optimizer.get_weights())


def calc_student_std(params):
    # [n_infer, B, n_params]  m, beta, a, b
    m_star = tf.reduce_mean(params[..., 0], axis=0)
    s_star = tf.sqrt(
        tf.reduce_mean(params[..., 0] ** 2 + params[..., 3] / (params[..., 2] - 1) * (1 / params[..., 1] + 1),
                       axis=0) - m_star ** 2)
    return s_star  # [B, ]


def calc_gaussian_std(params):
    # [n_infer, B, n_params]  loc, scale
    m_star = tf.reduce_mean(params[..., 0], axis=0)
    s_star = tf.sqrt(tf.reduce_mean(params[..., 1] ** 2 + params[..., 0] ** 2, axis=0) - m_star ** 2)
    return s_star  # [B, ]


def transform_to_StudentT(t):
    m, beta, a, b = t[..., 0], t[..., 1], t[..., 2], t[..., 3]
    return tfd.StudentT(df=2 * a, loc=m, scale=tf.math.sqrt(b * (beta + 1) / (a * beta)))
