from kari_regex import *
import numpy as np
import cv2
import os
import json
from PIL import Image
import h5py
import spectral.io.envi as envi
import pandas as pd
from math import sqrt
from Utilities.ImageUtilities import split_2d_image


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


def get_target_files(region_period_dirs: dict):
    for region in region_period_dirs:
        for period_prefix in region_period_dirs[region]:
            target_dir_items = parse_target_dir(region_period_dirs[region][period_prefix]['L1A'])
            region_period_dirs[region][period_prefix].update(target_dir_items)
            target_csv = get_target_csv(target_dir_items['TARGET_DIR'])
            region_period_dirs[region][period_prefix]['TARGET_CSV'] = target_csv
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


def crop_image(large_image: np.ndarray, labelX, labelY, labelClass, cropped_image_width=2048, cropped_image_height=2048):
    target_size = np.array([cropped_image_height, cropped_image_width])
    original_width = large_image.shape[0]
    original_height = large_image.shape[1]
    split_images = split_2d_image(large_image, target_size)
    cropped_images = split_images['images']
    labels = []
    for i_row, offset_row in enumerate(split_images['offsets']):
        row_labels = []
        for offset in offset_row:
            image_labels = []
            for x, y, cls in zip(labelX, labelY, labelClass):
                if (offset[0] <= x <= offset[0]+cropped_image_width) \
                    and (offset[1] <= y <= offset[1]+cropped_image_height):
                    x -= offset[0]
                    y -= offset[1]
                    image_labels.append((x, y, 150, 150, cls))
            row_labels.append(image_labels)
        labels.extend(row_labels)
    return cropped_images, labels, split_images['offsets']


def save_raw_image_to_numpy(dir_str, debug=False):
    meta_ = dict()
    raw_images_ = []
    raw_labels_ = []
    for region in dir_str:
        for period in dir_str[region]:
            raw = dir_str[region][period]['RAW']
            csv_path = dir_str[region][period]['TARGET_CSV']
            h5_img = get_raw(raw)
            raw_img_array = h5_img['S01']['SBI'][:]
            if debug:
                print(h5_img.keys())
                print(h5_img['S01']['SBI'][:])
            label_df = get_label(csv_path)
            label_df.columns = ['TARGET_ID', 'CLASS_ID', 'OBJECT_SIZE', 'PEAK_VAL', 'MEAN_VAL',
                                'IDX_X_SLC', 'IDX_Y_SLC', 'IDX_X_IMG', 'IDX_Y_IMG', 'VALID']
            label_df = label_df[['TARGET_ID', 'IDX_X_SLC', 'IDX_Y_SLC',
                                 'IDX_X_IMG', 'IDX_Y_IMG', 'OBJECT_SIZE', 'CLASS_ID']]
            raw_label = []
            for label in label_df.values:
                raw_label.append(label)
            raw_labels_.append(raw_label)
            cropped_images, labels, offsets = crop_image(raw_img_array, label_df['IDX_X_SLC'].to_numpy(),
                       label_df['IDX_Y_SLC'].to_numpy(),
                       label_df['CLASS_ID'].to_numpy())

            # if debug:
            #     row_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            #     sample_large_image = np.zeros(raw_img_array.shape)
            #     for irow, row in enumerate(raw_img_array):
            #         for icol, col in enumerate(row):
            #             sample_large_image[irow][icol] = [sqrt(col[0]**2+col[1]**2)]
            #     for i_row, offset_row in enumerate(offsets):
            #         for offset in offset_row:
            #             offset = np.array(offset)
            #             sample_large_image = cv2.rectangle(sample_large_image, offset, offset + np.array([2048, 2048]),
            #                                                row_colors[i_row % 3], 2)
            #     sample_large_image = cv2.resize(sample_large_image, dsize=(0, 0), fx=0.25, fy=0.25,
            #                                     interpolation=cv2.INTER_LINEAR)
            #     cv2.imshow('sample', sample_large_image)
            #     cv2.waitKey(0)
            #     for row_img in cropped_images:
            #         for img in row_img:
            #             img = np.array(list(map(lambda xy: [sqrt(xy[0]**2+xy[1]**2)], img)))
            #             cv2.imshow("sample", img)
            #             cv2.waitKey(0)
            cropped_images = np.array(cropped_images)
            shape_ = cropped_images.shape
            cropped_images = np.reshape(cropped_images, (shape_[0]*shape_[1], shape_[2], shape_[3], shape_[4]))
            labels = np.array(labels)
            np.save('input_images.npy', cropped_images)
            np.save('output_labels.npy', labels)
    pass


def save_images_to_numpy(dir_str: dict, debug=False):
    meta_ = dict()
    save_raw_image_to_numpy(dir_str, debug)
    # for region in dir_str:
    #     for period in dir_str[region]:
    #         raw = dir_str[region][period]['RAW']
    #         raw_slc = dir_str[region][period]['SLC']
    #         raw_slc_hdr = dir_str[region][period]['SLC_HEADER']
    #         sub_slc = dir_str[region][period]['SUB_BAND_SLC']
    #         sub_slc_hdr = dir_str[region][period]['SUB_BAND_SLC_HEADER']
    #         rgb_remap_bmp = dir_str[region][period]['RGB_REMAP_BMP']
    #         csv_path = dir_str[region][period]['TARGET_CSV']
    #         h5_img = get_raw(raw)
    #         print(h5_img['S01'].keys())
    #         # raw_img_array = h5_img['S01']['SBI'][:]
    #         # print(h5_img['S01']['SBI'][:])
    #         img = get_raw_slc(raw_slc, raw_slc_hdr)
    #         slc_array = img.read_band(0)
    #         print(type(slc_array[0][0]))
    #         sub_img = get_raw_slc(sub_slc, sub_slc_hdr)
    #         print(sub_img.bands)
    #         rgb_remap_image = get_image_rgb(rgb_remap_bmp)
    #         label_df = get_label(csv_path)
    #         label_df.columns = ['TARGET_ID', 'CLASS_ID', 'OBJECT_SIZE', 'PEAK_VAL', 'MEAN_VAL',
    #                             'IDX_X_SLC', 'IDX_Y_SLC', 'IDX_X_IMG', 'IDX_Y_IMG', 'VALID']
    #         label_df = label_df[['TARGET_ID', 'CLASS_ID', 'OBJECT_SIZE',
    #                              'IDX_X_SLC', 'IDX_Y_SLC', 'IDX_X_IMG', 'IDX_Y_IMG']]


if __name__ == '__main__':
    region_period_dirs = get_dirs(debug=True)
    region_period_dirs = get_target_files(region_period_dirs)
    with open('../data/directory_structure.json', 'w') as f:
        json.dump(region_period_dirs, f, indent='\t')
    dir_str = get_directory_structure()
    save_images_to_numpy(dir_str, debug=True)
