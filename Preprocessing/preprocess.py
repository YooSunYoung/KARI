from Utilities.loader import *
from Utilities.image_utilities import *
from functools import partial
from tqdm import tqdm
import os
import numpy as np


class STAGE:
    PARSE_DIRECTORY_STRUCTURE = 0
    IMAGES_TO_NUMPY = 1
    CROP_IMAGES = 2
    RESIZE_CROPPED_IMAGES = 3


class DATATYPE:
    SELECTED = 'selected'
    UNSELECTED = 'unselected'


class IMAGETYPE:
    RAW = 'raw'
    GRAYSCALE_RAW = 'gray_scale_raw'
    SUBBAND = 'subband'


low_tq = partial(tqdm, position=0, leave=True)


def save_raw_image_to_numpy(directory_structure: dict, debug=False, target_dir='../data/raw/', target_prefix='raw'):
    num_image = 0
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    for ir, region in enumerate(directory_structure):
        for ip, period in enumerate(directory_structure[region]):
            directory_structure[region][period]['image_id'] = num_image
            target_images_path = os.path.join(target_dir, '{}_images_{}.npy'.format(target_prefix, num_image))
            target_labels_path = os.path.join(target_dir, '{}_labels_{}.npy'.format(target_prefix, num_image))
            if os.path.exists(target_labels_path) or 'TARGET_CSV' not in directory_structure[region][period]:
                num_image += 1
                continue
            raw = directory_structure[region][period]['RAW']
            csv_path = directory_structure[region][period]['TARGET_CSV']
            h5_img = get_raw(raw)
            raw_img_array = h5_img['S01']['SBI'][:]
            raw_img_array = np.array(raw_img_array)
            label_df = get_label(csv_path)
            label_df.columns = ['TARGET_ID', 'CLASS_ID', 'OBJECT_SIZE', 'PEAK_VAL', 'MEAN_VAL',
                                'IDX_X_SLC', 'IDX_Y_SLC', 'IDX_X_IMG', 'IDX_Y_IMG', 'VALID']
            raw_label_df = label_df[['IDX_X_SLC', 'IDX_Y_SLC', 'OBJECT_SIZE', 'OBJECT_SIZE', 'CLASS_ID']]
            try:
                np.save(target_images_path, raw_img_array)
                np.save(target_labels_path, raw_label_df.to_numpy())
            except:
                print(target_images_path)

            del raw_label_df
            del raw_img_array
            num_image += 1
            if debug:
                break
    return directory_structure


def save_gray_scale_raw_image_to_numpy(directory_structure: dict, debug=False, target_dir='../data/raw/', target_prefix='raw'):
    num_image = 0
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    def get_magnitude_pixel(x):
        return [np.abs(complex(x[0], x[1]))]

    for ir, region in enumerate(directory_structure):
        for ip, period in enumerate(directory_structure[region]):
            directory_structure[region][period]['image_id'] = num_image
            target_images_path = os.path.join(target_dir, '{}_images_{}.npy'.format(target_prefix, num_image))
            target_labels_path = os.path.join(target_dir, '{}_labels_{}.npy'.format(target_prefix, num_image))
            if os.path.exists(target_images_path) or 'TARGET_CSV' not in directory_structure[region][period]:
                num_image += 1
                continue
            raw = directory_structure[region][period]['RAW']
            csv_path = directory_structure[region][period]['TARGET_CSV']
            h5_img = get_raw(raw)
            raw_img_array = h5_img['S01']['SBI'][:]
            raw_img_shape = raw_img_array.shape
            raw_img_array = np.reshape(raw_img_array, (raw_img_shape[0] * raw_img_shape[1], raw_img_shape[2]))
            raw_img_array = np.array(list(map(get_magnitude_pixel, low_tq(raw_img_array))))
            raw_img_array = np.reshape(raw_img_array, (raw_img_shape[0], raw_img_shape[1], 1))
            label_df = get_label(csv_path)
            label_df.columns = ['TARGET_ID', 'CLASS_ID', 'OBJECT_SIZE', 'PEAK_VAL', 'MEAN_VAL',
                                'IDX_X_SLC', 'IDX_Y_SLC', 'IDX_X_IMG', 'IDX_Y_IMG', 'VALID']
            raw_label_df = label_df[['IDX_X_SLC', 'IDX_Y_SLC', 'OBJECT_SIZE', 'OBJECT_SIZE', 'CLASS_ID']]
            np.save(target_images_path, raw_img_array)
            np.save(target_labels_path, raw_label_df.to_numpy())
            num_image += 1
            if debug:
                break
    return directory_structure


def save_full_band_image_to_numpy(directory_structure, debug=False, target_dir='../data/raw/', target_prefix='subband'):
    num_image = 0
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    for ir, region in enumerate(directory_structure):
        for ip, period in low_tq(enumerate(directory_structure[region])):
            directory_structure[region][period]['image_id'] = num_image
            target_images_path = os.path.join(target_dir, '{}_images_{}.npy'.format(target_prefix, num_image))
            target_labels_path = os.path.join(target_dir, '{}_labels_{}.npy'.format(target_prefix, num_image))
            if os.path.exists(target_images_path) or \
                    'TARGET_CSV' not in directory_structure[region][period] or \
                    'SUB_BAND_SLC_HEADER' not in directory_structure[region][period]:
                num_image += 1
                continue
            sub_slc = directory_structure[region][period]['SUB_BAND_SLC_HEADER']
            csv_path = directory_structure[region][period]['TARGET_CSV']
            subband_slc_array = get_raw_slc(None, sub_slc).asarray()
            label_df = get_label(csv_path)
            label_df.columns = ['TARGET_ID', 'CLASS_ID', 'OBJECT_SIZE', 'PEAK_VAL', 'MEAN_VAL',
                                'IDX_X_SLC', 'IDX_Y_SLC', 'IDX_X_IMG', 'IDX_Y_IMG', 'VALID']
            raw_label_df = label_df[['IDX_X_IMG', 'IDX_Y_IMG', 'OBJECT_SIZE', 'OBJECT_SIZE', 'CLASS_ID']]
            np.save(target_images_path, subband_slc_array)
            np.save(target_labels_path, raw_label_df.to_numpy())
            num_image += 1
            if debug:
                break
    return directory_structure


def save_images_to_numpy(dir_structure, data_type, image_type, debug=False):
    target_prefix = '{}_{}'.format(data_type, image_type)
    target_dir = 'F:/0_raw_numpy/{}_{}/'.format(iit, image_type)
    if image_type == IMAGETYPE.RAW:
        dir_structure = save_raw_image_to_numpy(dir_structure, target_prefix=target_prefix, target_dir=target_dir, debug=debug)
    elif image_type == IMAGETYPE.GRAYSCALE_RAW:
        dir_structure = save_gray_scale_raw_image_to_numpy(dir_structure, target_prefix=target_prefix, target_dir=target_dir, debug=debug)
    elif image_type == IMAGETYPE.SUBBAND:
        dir_structure = save_full_band_image_to_numpy(dir_structure, target_prefix=target_prefix, target_dir=target_dir, debug=debug)
    return dir_structure

large_images_tq = partial(tqdm, position=0, leave=True)


def crop_raw_image(raw_image_path=None, raw_label_path=None, debug=False,
                   target_dir='../data/cropped_raw/', prefix=''):
    if debug:
        raw_image_path = '../data/raw/debug_raw_images_0.npy'
        raw_label_path = '../data/raw/debug_raw_labels_0.npy'
        prefix = 'debug_raw'
    image_number = raw_label_path.split('_')[-1].split('.')[0]
    image_path = os.path.join(target_dir, 'cropped_{}_images_{}.npy'.format(prefix, image_number))
    label_path = image_path.replace('images', 'labels')
    offsets_path = image_path.replace('images', 'offsets')
    if os.path.exists(image_path) and os.path.exists(label_path) and os.path.exists(offsets_path):
        return True
    raw_img_array = np.load(raw_image_path, allow_pickle=True)
    labels = np.load(raw_label_path, allow_pickle=True)
    cropped_images, labels, offsets = crop_image(raw_img_array, labels,
                                                 cropped_image_width=1024, cropped_image_height=1024)
    cropped_images = np.array(cropped_images, dtype=object)
    labels = np.array(labels, dtype=object)
    offsets = np.array(offsets, dtype=object)
    np.save(image_path, cropped_images, allow_pickle=True)
    np.save(label_path, labels, allow_pickle=True)
    np.save(offsets_path, offsets, allow_pickle=True)
    return True


if __name__ == '__main__':

    stage = STAGE.PARSE_DIRECTORY_STRUCTURE

    if stage == STAGE.PARSE_DIRECTORY_STRUCTURE:
        # save directory structure of selected images
        selected_region_period_dirs = get_dirs(root_dir='D:/20210517', debug=False)
        selected_region_period_dirs = get_target_files(selected_region_period_dirs,
                                                       removed_dir_path='../data/directory_structure/selected_removed.json')
        with open('../data/directory_structure/selected.json', 'w') as f:
            json.dump(selected_region_period_dirs, f, indent='\t')
        # save directory structure of unselected images
        unselected_region_period_dirs = get_dirs(root_dir='E:/210601', debug=False)
        unselected_region_period_dirs = get_target_files(unselected_region_period_dirs,
                                                         removed_dir_path='../data/directory_structure/unselected_removed.json')
        with open('../data/directory_structure/unselected.json', 'w') as f:
            json.dump(unselected_region_period_dirs, f, indent='\t')
    elif stage == STAGE.IMAGES_TO_NUMPY:
        # save raw images from unselected group into numpy
        for idt, dt in enumerate([DATATYPE.SELECTED, DATATYPE.UNSELECTED]):
            dir_structure = get_directory_structure(json_path='../data/directory_structure/{}.json'.format(dt))
            for iit, it in enumerate([IMAGETYPE.RAW, IMAGETYPE.GRAYSCALE_RAW, IMAGETYPE.SUBBAND]):
                dir_structure = save_images_to_numpy(dir_structure, data_type=dt, image_type=it, debug=False)
            with open('../data/directory_structure/{}.json'.format(dt), 'w') as f:
                json.dump(dir_structure, f, indent='\t')
    elif stage == STAGE.CROP_IMAGES:
        raw_files_directory, raw_file_prefix, raw_file_label_prefix, raw_file_image_prefix = None, None, None, None
        # save raw images from unselected group into numpy
        for idt, dt in enumerate([DATATYPE.SELECTED, DATATYPE.UNSELECTED]):
            dir_structure = get_directory_structure(json_path='../data/directory_structure/{}.json'.format(dt))
            for iit, it in enumerate([IMAGETYPE.RAW, IMAGETYPE.GRAYSCALE_RAW, IMAGETYPE.SUBBAND]):
                if it == IMAGETYPE.GRAYSCALE_RAW or it == IMAGETYPE.RAW:
                    continue
                raw_files_directory = 'F:/0_raw_numpy/{}_{}/'.format(iit, it)
                raw_file_image_prefix = '{}_{}_images'.format(dt, it)
                raw_file_label_prefix = '{}_{}_labels'.format(dt, it)
                raw_file_prefix = '{}_{}'.format(dt, it)
                raw_image_labels_paths = os.listdir(raw_files_directory)
                image_paths = [os.path.join(raw_files_directory, x) for x in
                               raw_image_labels_paths if raw_file_image_prefix in x]
                label_paths = [os.path.join(raw_files_directory, x) for x in
                               raw_image_labels_paths if raw_file_label_prefix in x]
                for image_path, label_path in low_tq(large_images_tq(zip(image_paths, label_paths))):
                    crop_raw_image(image_path, label_path, prefix=raw_file_prefix,
                                   target_dir='F:/1_cropped_numpy/{}_{}/'.format(iit, it))

    elif stage == STAGE.RESIZE_CROPPED_IMAGES:
        files_directory = '../data/cropped_raw/'
        image_labels_paths = os.listdir(files_directory)
        image_paths = [os.path.join(files_directory, x) for x in image_labels_paths if 'raw_images' in x]
        label_paths = [os.path.join(files_directory, x) for x in image_labels_paths if 'raw_labels' in x]
        for image_path, label_path in zip(image_paths, label_paths):
            resize_cropped_images(image_path, label_path, prefix='debug_raw')
        image_paths = [os.path.join(files_directory, x) for x in image_labels_paths if 'subband_images' in x]
        label_paths = [os.path.join(files_directory, x) for x in image_labels_paths if 'subband_labels' in x]
        for image_path, label_path in zip(image_paths, label_paths):
            resize_cropped_images(image_path, label_path, prefix='debug_subband')

