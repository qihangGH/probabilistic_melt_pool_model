import argparse
import os
import tqdm
import numpy as np
from utilities import parallel_pre_process_patches

parser = argparse.ArgumentParser(description='Generate train, validation, and test dataset')
parser.add_argument('--root_save_dir', default=None)
parser.add_argument('--patch_data_path', default=None,
                    help='The file path of patch data to be processed, which is generated by `grid_based_data.py`')

parser.add_argument('--crop_patch_size', type=int, default=41,
                    help='The patch size that is used to crop patches from given patch data')
parser.add_argument('--generate_neigh_only', action='store_true', default=False,
                    help='Only generate neighbor patch data')

args = parser.parse_args()
sub_folders = ['power', 'velocity', 'size', 'ap', 'neighbor_{}'.format(args.crop_patch_size), 'x', 'y']
folders = []
for folder in sub_folders:
    folders.append(os.path.join(args.root_save_dir, folder))
    if not os.path.isdir(os.path.join(args.root_save_dir, folder)):
        os.makedirs(os.path.join(args.root_save_dir, folder))


if __name__ == '__main__':
    print('Data cropping and pre-processing...')
    # [x, y, P, v, idx, T, s, ap]
    record_layer_len = []
    power = []
    velocity = []
    mp_size = []
    mp_ap = []
    x = []
    y = []
    for layer in tqdm.tqdm([18 + 12 * i for i in range(20)] + [13 + 12 * i for i in range(20)]):
        data = np.load(args.patch_data_path + r'/layer_' + repr(layer) + r'.npy')
        half_patch_size = int((data.shape[1] - 1) / 2)
        effective_ids = np.where(data[:, half_patch_size, half_patch_size, 6] > 0)  # melt pool size > 0
        data = data[effective_ids]
        if not args.generate_neigh_only:
            record_layer_len.append(len(data))
            power.append(data[:, half_patch_size, half_patch_size, 2])
            velocity.append(data[:, half_patch_size, half_patch_size, 3])
            mp_size.append(data[:, half_patch_size, half_patch_size, 6])
            mp_ap.append(data[:, half_patch_size, half_patch_size, 7])
            x.append(data[:, half_patch_size, half_patch_size, 0])
            y.append(data[:, half_patch_size, half_patch_size, 1])
            np.save(folders[0] + '/power_' + repr(layer) + r'.npy', power[-1])
            np.save(folders[1] + '/velocity_' + repr(layer) + r'.npy', velocity[-1])
            np.save(folders[2] + '/size_' + repr(layer) + r'.npy', mp_size[-1])
            np.save(folders[3] + '/ap_' + repr(layer) + r'.npy', mp_ap[-1])
            np.save(folders[5] + '/x_' + repr(layer) + r'.npy', x[-1])
            np.save(folders[6] + '/y_' + repr(layer) + r'.npy', y[-1])
        data = parallel_pre_process_patches(data, args.crop_patch_size, order_mode='exp', exp_rate=0.0004)
        np.save(folders[4] + '/neighbor_' + repr(layer) + r'.npy', data)
    if not args.generate_neigh_only:
        power = np.hstack(power)
        velocity = np.hstack(velocity)
        record_layer_len = np.array(record_layer_len)
        mp_size = np.hstack(mp_size)
        mp_ap = np.hstack(mp_ap)
        np.save(os.path.join(args.root_save_dir, r'power.npy'), power)
        np.save(os.path.join(args.root_save_dir, r'velocity.npy'), velocity)
        np.save(os.path.join(args.root_save_dir, r'mp_size.npy'), mp_size)
        np.save(os.path.join(args.root_save_dir, r'mp_ap.npy'), mp_ap)
        np.save(os.path.join(args.root_save_dir, r'record_layer_len.npy'), record_layer_len)
    print('Finish')

