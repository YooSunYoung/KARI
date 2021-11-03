from Utilities.kari_regex import *
import numpy as np
import os
import json
from PIL import Image
import h5py
import spectral.io.envi as envi
import pandas as pd

data_type_order = {
    'raw': 0,
    'gray_scale_raw': 1,
    'subband': 2
}


def get_dataset(dir='F:/1_cropped_numpy', image_type='raw', data_type='selected', image_num=0):
    images = np.load(os.path.join(dir,
                                  '{}_{}'.format(data_type_order[image_type], image_type),
                                  'cropped_{}_{}_images_{}.npy'.format(data_type, image_type, image_num)), allow_pickle=True)
    labels = np.load(os.path.join(dir,
                                  '{}_{}'.format(data_type_order[image_type], image_type),
                                  'cropped_{}_{}_labels_{}.npy'.format(data_type, image_type, image_num)), allow_pickle=True)
    return images, labels


def get_dirs(root_dir="D:/20210517", debug=False):
    region_dirs = os.listdir(root_dir)
    region_dirs = [os.path.join(root_dir, x) for x in region_dirs
                         if REGION_DIR_NAME_REGEX.fullmatch(x) is not None]
    if debug:
        region_dirs = [os.path.join(root_dir, '202105_AI0001_KARI-AO-20210504-China_1')]
    region_period_dirs = dict()
    for region_dir in region_dirs:
        region = region_dir.split('-')[-1]
        period_dirs = os.listdir(region_dir)
        period_prefixes = list(map(lambda x: (x, x[:-13]), period_dirs))
        prefix_set = set([x[1] for x in period_prefixes])
        region_map = dict()
        for prefix in prefix_set:
            region_map[prefix] = dict()
        for period_dir, period_prefix in period_prefixes:
            if period_dir.endswith('L1C'):
                region_map[period_prefix]['L1C'] = os.path.join(region_dir, period_dir)
            elif period_dir.endswith('L1A'):
                region_map[period_prefix]['L1A'] = os.path.join(region_dir, period_dir)
        region_period_dirs[region] = region_map
    return region_period_dirs


def parse_target_dir(l1a_dir):
    items = os.listdir(l1a_dir)
    res = dict()
    for it in items:
        item_path = os.path.join(l1a_dir, it)
        if TARGET_CHIP_DIR_REGEX.fullmatch(it) is not None:
            res['TARGET_DIR'] = item_path
        elif it.endswith('h5'):
            res['RAW'] = item_path
        elif it.endswith("PDR.img"):
            res['SLC'] = item_path
        elif it.endswith("PDR.hdr"):
            res['SLC_HEADER'] = item_path
        elif it.endswith("BPF.img"):
            res['SUB_BAND_SLC'] = item_path
        elif it.endswith("BPF.hdr"):
            res['SUB_BAND_SLC_HEADER'] = item_path
        elif it.endswith("BPF_00.bmp"):
            res["FULL_BAND_BMP"] = item_path
        elif it.endswith("BPF_04_RGB.bmp"):
            res["RGB_REMAP_BMP"] = item_path
    return res


def get_target_csv(target_dir):
    if target_dir is None:
        return None
    items = os.listdir(target_dir)
    for it in items:
        if TARGET_CHIP_REGEX.fullmatch(it) is not None:
            return os.path.join(target_dir, it)
    return None


def get_target_files(region_period_dirs: dict, save_removed=True, removed_dir_path='../data/removed_dirs.json'):
    removed = dict()
    for region in region_period_dirs:
        removed[region] = dict()
        for period_prefix in region_period_dirs[region]:
            if 'L1A' not in region_period_dirs[region][period_prefix]:
                continue
            target_dir_items = parse_target_dir(region_period_dirs[region][period_prefix]['L1A'])
            if 'TARGET_DIR' not in target_dir_items:
                continue
            region_period_dirs[region][period_prefix].update(target_dir_items)
            target_csv = get_target_csv(target_dir_items['TARGET_DIR'])
            region_period_dirs[region][period_prefix]['TARGET_CSV'] = target_csv
    if save_removed:
        with open(removed_dir_path, 'w') as f:
            json.dump(removed, f, indent='\t')
    return region_period_dirs


def get_directory_structure(json_path='../data/directory_structure.json'):
    with open(json_path) as _f_:
        directory_structure = json.load(_f_)
    return directory_structure


def get_raw(image_path):
    return h5py.File(image_path, 'r')


def get_raw_slc(image_path, header_path):
    return envi.open(header_path)


def get_image_rgb(image_path):
    im = Image.open(image_path)
    return np.array(im)


def get_label(csv_path):
    return pd.read_csv(csv_path)

