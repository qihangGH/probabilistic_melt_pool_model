import argparse
import json
import os.path
import cv2
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import tqdm
from skimage.measure import label
from scipy import ndimage


parser = argparse.ArgumentParser("Extract melt pool features")
parser.add_argument("--data_dir", type=str, default="../Melt_Pool_Camera", help="The directory of the `Melt_Pool_Camera` folder")
parser.add_argument("--save_filepath", type=str, default="../mp_feature_info.json", help="The filepath to save results")

args = parser.parse_args()


def feature_extraction(img_path, thresh=150, ellipse=False):
    img = plt.imread(img_path)
    # binarize the img
    img = np.uint8(img >= thresh)
    # find the largest connected component
    labels = label(img)
    if labels.max() == 0:
        return [0., 0., 0.]
    else:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        # fill possible holes
        holes_filled = ndimage.binary_fill_holes(largestCC).astype(np.uint8)
        area = float(np.sum(holes_filled))
        # fit the eclipse
        contour, _ = cv2.findContours(holes_filled, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # There should be at least 5 points to fit the ellipse
        if len(contour[0]) >= 5:
            ((centx, centy), (width, height), angle) = cv2.fitEllipse(np.squeeze(contour))
            if ellipse is True:
                area = width * height * np.pi / 4
            return [area, width / height, angle]
        else:
            if ellipse is True:
                area = 0.
            return [area, 0., 0.]


if __name__ == '__main__':
    record_features = []
    _feature_extraction = partial(feature_extraction, thresh=150, ellipse=False)
    for layer in tqdm.tqdm(range(1, 251)):
        img_folder = os.path.join(args.data_dir, 'MIA_L') + repr(layer).rjust(4, '0')
        img_dir = [os.path.join(img_folder, i) for i in os.listdir(img_folder)]
        assert os.path.exists(os.path.join(img_folder, 'frame' + repr(len(img_dir)).rjust(5, '0') + r'.bmp')), \
            f'{layer}'
        cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=cores) as pool:
            features = pool.map(_feature_extraction, img_dir)
        record_features.append(features)
    with open(args.save_filepath, 'w') as f:
        json.dump(record_features, f)
