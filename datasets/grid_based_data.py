import argparse
import os
import time
import tqdm
import json
import numpy as np


parser = argparse.ArgumentParser(description='Preparation of 2D patch data')
parser.add_argument('--original_data', default="../../Build_command_data/",
                    help='the folder of the dataset to be processed')
parser.add_argument('--melt_pool_feature_path', default="../../mp_feature_info.json",
                    help='the path of extracted melt pool features')
parser.add_argument('--save_path', default=None,
                    help='the folder to save processing results')


parser.add_argument('--grid_size', type=float, default=
0.01,
                    help='the grid size of a map')
parser.add_argument('--out_stretch_size', type=float, default=0.4,
                    help='the length of a map that outstretches cd the layer')
parser.add_argument('--part_length', type=float, default=10, help='the length of a layer')
parser.add_argument('--part_width', type=float, default=10, help='the width of a layer')
parser.add_argument('--half_patch_size', type=float, default=
30,
                    help='the half size of a patch')
args = parser.parse_args()

if args.save_path is not None:
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

if args.save_reordered_mp_fea_path is not None:
    if not os.path.isdir(args.save_reordered_mp_fea_path):
        os.mkdir(args.save_reordered_mp_fea_path)

# Create a grid for a layer
# side length of a small grid
grid_size = args.grid_size
out_stretch_size = args.out_stretch_size
part_length = args.part_length
part_width = args.part_width
# Number of grids according to the radius of a neighborhood
half_patch_size = args.half_patch_size
# Because 10.6 // 0.01 = 1059.0, times 100 here to get correct results
num_grid_rows = int((part_width + 2 * out_stretch_size) * 100 // (grid_size * 100))
num_grid_cols = int((part_length + 2 * out_stretch_size) * 100 // (grid_size * 100))
# Load melt pool features
with open(args.melt_pool_feature_path, 'r') as f:
    mp_features = json.load(f)


def extract_patches_of_a_layer(layer):
    mp_size = np.array(mp_features[layer - 1])[2:, 0]  # Compensation for misalignment, so starts for the index 2
    mp_ap = np.array(mp_features[layer - 1])[2:, 1]
    xypt = np.loadtxt(args.original_data + r'T500_3D_Scan_Strategies_fused_layer'
                      + repr(layer).rjust(4, '0') + '.csv', delimiter=',')
    temp_speed = np.hstack([np.zeros(1),
                            np.linalg.norm(xypt[2:, :2] - xypt[:-2, :2], axis=1) / 0.00002,
                            np.zeros(1)])
    # [x, y, P, T, v, idx]
    xypt = np.hstack([xypt, temp_speed.reshape((-1, 1)), np.arange(len(xypt)).reshape(-1, 1)])
    # All melting points of a layer
    melt_points = xypt[np.where(xypt[:, 2] > 0)]
    del xypt
    # Change the order to [x, y, P, v, idx, T]
    melt_points = melt_points.T[np.array([0, 1, 2, 4, 5, 3])].T
    # The melting points that are recorded by the camera
    # Because start from index 2, the last 2 features are not used
    record_points = melt_points[np.where(melt_points[:, 5] > 0)][:-2]
    assert len(record_points) == len(mp_size)
    min_record_x, max_record_x = np.min(record_points[:, 0]), np.max(record_points[:, 0])
    min_record_y, max_record_y = np.min(record_points[:, 1]), np.max(record_points[:, 1])
    del record_points
    # All melting points of the component that is monitoring by the camera
    melts = melt_points[np.where((min_record_x <= melt_points[:, 0]) & (melt_points[:, 0] <= max_record_x)
                                 & (min_record_y <= melt_points[:, 1]) & (
                                         melt_points[:, 1] <= max_record_y))]
    del melt_points
    # Not all melting points are captured by the camera. Those melt pool sizes are set as -1.
    temp_mp_size = -1. * np.ones_like(melts[:, 0])
    temp_mp_ap = temp_mp_size.copy()
    record_melt_idx = np.where(melts[:, 5] > 0)[0][:-2]
    temp_mp_size[record_melt_idx] = mp_size
    temp_mp_ap[record_melt_idx] = mp_ap
    # [x, y, P, v, idx, T, s, ap]
    melts = np.hstack([melts, temp_mp_size.reshape(-1, 1), temp_mp_ap.reshape(-1, 1)])
    # The coordinate of the top left corner
    coord_top_left = [np.min(melts[:, 0]) - out_stretch_size,
                      np.max(melts[:, 1]) + out_stretch_size]
    # Get the indices of points that located at the ith row and jth column of the grid
    # x∈[coord_top_left[0] + j*grid_size, coord_top_left[0] + (j+1)*grid_size)
    # y∈[coord_top_left[1] - i*grid_size, coord_top_left[1] - (i+1)*grid_size)
    j_index = ((melts[:, 0] - coord_top_left[0]) // grid_size).astype(np.int32)
    i_index = ((coord_top_left[1] - melts[:, 1]) // grid_size).astype(np.int32)
    layer_idx_in_ij = [[[] for _ in range(num_grid_cols)] for _ in range(num_grid_rows)]
    for k, (i, j) in enumerate(zip(i_index, j_index)):
        layer_idx_in_ij[i][j].append(k)
    record_melt_info = np.hstack([i_index[record_melt_idx].reshape(-1, 1),
                                  j_index[record_melt_idx].reshape(-1, 1),
                                  record_melt_idx.reshape(-1, 1)])
    record_melt_group = []
    while len(record_melt_info) > 0:
        # Find all points whose indices are the same as that of the 0th point in the `record_melt_info`
        temp_idx = np.where(np.sum(record_melt_info[:, :2] == record_melt_info[0, :2], axis=1) == 2)[0]
        record_melt_group.append({'coordinate': record_melt_info[0, :2], 'idx': record_melt_info[temp_idx, 2]})
        # Delete the points that have been iterated
        record_melt_info = np.delete(record_melt_info, temp_idx, axis=0)
    # Iterate each dictionary in `record_melt_group`
    count_dic = 0
    record_patches = []
    for dic in record_melt_group:
        # time1 = time.time()
        count_dic += 1
        temp_patch = [np.zeros((2 * half_patch_size + 1, 2 * half_patch_size + 1, 8))
                      for _ in range(len(dic['idx']))]
        start_i, end_i = dic['coordinate'][0] - half_patch_size, dic['coordinate'][0] + half_patch_size + 1
        start_j, end_j = dic['coordinate'][1] - half_patch_size, dic['coordinate'][1] + half_patch_size + 1
        # Iterate each melting point located in the current grid
        for k, idx in enumerate(dic['idx']):
            # The current center point is melts[idx]
            # Iterate each grid in its neighborhood
            for i in range(start_i, end_i):
                for j in range(start_j, end_j):
                    idx_in_ij = np.array(layer_idx_in_ij[i][j])
                    if len(idx_in_ij) > 0:
                        # Find the points that are melted before the current one in the grid (i, j)
                        idx_in_ij = idx_in_ij[np.where(melts[idx_in_ij, 4] < melts[idx, 4])]
                        # Only one points in grid (i, j), just add melting information
                        if len(idx_in_ij) == 1:
                            temp_patch[k][i - start_i, j - start_j] = melts[idx_in_ij]
                        # Multiple points in grid (i, j)
                        # 有两种情况: 一是和当前点时间上距离很远的熔融点, 只是空间上接近
                        #            二是时间和空间都和当前点距离很近, 一个像素内包括了多个密集的点
                        if len(idx_in_ij) > 1:
                            # Group the consecutive points
                            separate_idx = [0]
                            for m in range(len(idx_in_ij) - 1):
                                if idx_in_ij[m + 1] - idx_in_ij[m] > 50:
                                    separate_idx.append(m + 1)
                            separate_idx.append(len(idx_in_ij))
                            # Only one group
                            if len(separate_idx) == 2:
                                temp_patch[k][i - start_i, j - start_j] = np.mean(melts[idx_in_ij], axis=0)
                            # Multiple groups
                            else:
                                idx_groups = [melts[idx_in_ij[separate_idx[n]:separate_idx[n + 1]]]
                                              for n in range(len(separate_idx) - 1)]
                                # Calculate the mean index of each group
                                mean_time = np.array([idx_group[:, 4].mean() for idx_group in idx_groups])
                                # Use the group whose time index is nearest to the current point
                                nearest_group = idx_groups[int(np.argmax(mean_time))]
                                temp_patch[k][i - start_i, j - start_j] = np.mean(nearest_group, axis=0)
            # Add the current center point
            temp_patch[k][half_patch_size, half_patch_size] = melts[idx]
        record_patches.extend(temp_patch)
        # print('\r{} - {}/{} - {:.4f}'.format(layer, count_dic, len(record_melt_group), time.time() - time1), end='')
    record_patches = np.asarray(record_patches)
    np.save(args.save_path + r'/layer_{}.npy'.format(layer), record_patches)
    print('Finish layer {}.'.format(layer))


if __name__ == '__main__':
    # You can change the layers
    layers = np.array([18 + 12 * i for i in range(20)] + [13 + 12 * i for i in range(20)])
    for layer in tqdm.tqdm(layers):
        extract_patches_of_a_layer(layer)
