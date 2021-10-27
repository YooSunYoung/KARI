from Utilities.loader import *
from Utilities.image_utilities import *
from tqdm import tqdm
import os
import numpy as np


def save_raw_image_to_numpy(directory_structure: dict, debug=False, target_dir='../data/raw/', target_prefix='raw'):
    num_image = 0

    def get_new_pixel(x):
        return [x[0], x[1], np.sqrt(x.dot(x))]

    for ir, region in tqdm(enumerate(directory_structure)):
        for ip, period in enumerate(directory_structure[region]):
            raw = directory_structure[region][period]['RAW']
            csv_path = directory_structure[region][period]['TARGET_CSV']
            h5_img = get_raw(raw)
            raw_img_array = h5_img['S01']['SBI'][:]
            raw_img_shape = raw_img_array.shape
            raw_img_array = np.reshape(raw_img_array, (raw_img_shape[0] * raw_img_shape[1], raw_img_shape[2]))
            raw_img_array = np.array(list(map(get_new_pixel, raw_img_array)))
            raw_img_array = np.reshape(raw_img_array, (raw_img_shape[0], raw_img_shape[1], 3))
            label_df = get_label(csv_path)
            label_df.columns = ['TARGET_ID', 'CLASS_ID', 'OBJECT_SIZE', 'PEAK_VAL', 'MEAN_VAL',
                                'IDX_X_SLC', 'IDX_Y_SLC', 'IDX_X_IMG', 'IDX_Y_IMG', 'VALID']
            raw_label_df = label_df[['IDX_X_SLC', 'IDX_Y_SLC', 'OBJECT_SIZE', 'CLASS_ID']]
            np.savez_compressed(os.path.join(target_dir, '{}_images_{}.npz'.format(target_prefix, num_image)),
                                raw_img_array)
            np.savez_compressed(os.path.join(target_dir, '{}_labels_{}.npz'.format(target_prefix, num_image)),
                                raw_label_df.to_numpy())
            num_image += 1
            if debug:
                break


def save_full_band_image_to_numpy(directory_structure, debug=False, target_dir='../data/raw/', target_prefix='subband'):
    num_image = 0
    for ir, region in tqdm(enumerate(directory_structure)):
        for ip, period in enumerate(directory_structure[region]):
            sub_slc = directory_structure[region][period]['SUB_BAND_SLC_HEADER']
            csv_path = directory_structure[region][period]['TARGET_CSV']
            subband_slc_array = get_raw_slc(None, sub_slc).asarray()
            label_df = get_label(csv_path)
            label_df.columns = ['TARGET_ID', 'CLASS_ID', 'OBJECT_SIZE', 'PEAK_VAL', 'MEAN_VAL',
                                'IDX_X_SLC', 'IDX_Y_SLC', 'IDX_X_IMG', 'IDX_Y_IMG', 'VALID']
            raw_label_df = label_df[['IDX_X_IMG', 'IDX_Y_IMG', 'OBJECT_SIZE', 'CLASS_ID']]
            np.savez_compressed(os.path.join(target_dir, '{}_images_{}.npz'.format(target_prefix, num_image)), subband_slc_array)
            np.savez_compressed(os.path.join(target_dir, '{}_labels_{}.npz'.format(target_prefix, num_image)), raw_label_df.to_numpy())
            num_image += 1
            if debug:
                break


def crop_raw_image(raw_image_path=None, raw_label_path=None, debug=False,
                   target_dir='../data/cropped_raw/', prefix=''):
    if debug:
        raw_image_path = '../data/raw/debug_raw_images_0.npz'
        raw_label_path = '../data/raw/debug_raw_labels_0.npz'
        prefix = 'debug_raw'
    raw_img_array = np.load(raw_image_path)
    raw_img_array = raw_img_array.get('arr_0')
    labels = np.load(raw_label_path)
    labels = labels.get('arr_0')
    cropped_images, labels, offsets = crop_image(raw_img_array, labels)
    cropped_images = np.array(cropped_images)
    shape_ = cropped_images.shape
    cropped_images = np.reshape(cropped_images, (shape_[0] * shape_[1], shape_[2], shape_[3], shape_[4]))
    labels = np.array(labels, dtype=object)
    offsets = np.array(offsets, dtype=object)
    image_number = raw_label_path.split('_')[-1].split('.')[0]
    image_path = os.path.join(target_dir, 'cropped_{}_images_{}.npz'.format(prefix, image_number))
    label_path = os.path.join(target_dir, 'cropped_{}_labels_{}.npz'.format(prefix, image_number))
    offsets_path = os.path.join(target_dir, 'cropped_{}_offsets_{}.npz'.format(prefix, image_number))
    np.savez_compressed(image_path, cropped_images, allow_pickle=True)
    np.savez_compressed(label_path, labels, allow_pickle=True)
    np.savez_compressed(offsets_path, offsets, allow_pickle=True)


def resize_cropped_images(cropped_image_path='../data/cropped_raw/cropped_debug_raw_images_0.npz',
                          cropped_label_path='../data/cropped_raw/cropped_debug_raw_labels_0.npz',
                          target_width=800, target_height=800,
                          target_dir='../data/resized_cropped_raw/',
                          prefix='debug_raw', debug=False):
    cropped_images = np.load(cropped_image_path, allow_pickle=True)
    cropped_images = cropped_images.get('arr_0')
    resized_images = []
    cropped_label_list = np.load(cropped_label_path, allow_pickle=True)
    cropped_label_list = cropped_label_list.get('arr_0')
    file_number = cropped_image_path.split('_')[-1].split('.')[0]
    for i_data, (image, labels) in tqdm(enumerate(zip(cropped_images, cropped_label_list))):
        width = image.shape[0]
        height = image.shape[1]
        width_ratio = target_width / width
        height_ratio = target_height / height
        # max_component = np.max(image)
        image = cv2.resize(image, dsize=(target_width, target_height), interpolation=cv2.INTER_LINEAR)
        resized_images.append(image)
        for i_label, label in enumerate(labels):
            label = [label[0] * width_ratio, label[1] * height_ratio,
                     label[2] * width_ratio, label[3] * height_ratio, label[4]]
            cropped_label_list[i_data][i_label] = label
    resized_images = np.array(resized_images)
    image_path = os.path.join(target_dir, 'cropped_resized_{}_images_{}.npy'.format(prefix, file_number))
    label_path = os.path.join(target_dir, 'cropped_resized_{}_labels_{}.npy'.format(prefix, file_number))
    np.save(image_path, resized_images)
    np.save(label_path, cropped_label_list)


if __name__ == '__main__':
    PARSE_DIRECTORY_STRUCTURE = 0
    IMAGES_TO_NUMPY = 1
    CROP_IMAGES = 2
    RESIZE_CROPPED_IMAGES = 3

    phase = RESIZE_CROPPED_IMAGES

    if phase == PARSE_DIRECTORY_STRUCTURE:
        # save directory structure for debugging
        region_period_dirs = get_dirs(debug=True)
        region_period_dirs = get_target_files(region_period_dirs,
                                              removed_dir_path='../data/directory_structure/debugging_removed.json')
        with open('../data/directory_structure/debugging.json', 'w') as f:
            json.dump(region_period_dirs, f, indent='\t')
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
    elif phase == IMAGES_TO_NUMPY:
        dir_str = get_directory_structure(json_path='../data/directory_structure/debugging.json')
        save_raw_image_to_numpy(dir_str, target_dir='../data/raw', target_prefix='debug_raw', debug=True)
        save_full_band_image_to_numpy(dir_str, target_dir='../data/raw', target_prefix='debug_subband', debug=True)
        # save raw images from selected group into numpy
        selected_dir_str = get_directory_structure(json_path='../data/directory_structure/selected.json')
        save_raw_image_to_numpy(selected_dir_str, target_dir='D:/raw_numpy/', target_prefix='selected_raw', debug=False)
        save_full_band_image_to_numpy(selected_dir_str, target_dir='D:/raw_numpy/', target_prefix='selected_subband', debug=False)
        # save raw images from unselected group into numpy
        unselected_dir_str = get_directory_structure(json_path='../data/directory_structure/unselected.json')
        save_raw_image_to_numpy(unselected_dir_str, target_dir='E:/raw_numpy/', target_prefix='unselected_raw', debug=False)
        save_full_band_image_to_numpy(unselected_dir_str, target_dir='E:/raw_numpy/', target_prefix='unselected_subband', debug=False)
    elif phase == CROP_IMAGES:
        raw_files_directory = '../data/raw/'
        raw_image_labels_paths = os.listdir(raw_files_directory)
        image_paths = [os.path.join(raw_files_directory, x) for x in raw_image_labels_paths if 'debug_raw_images' in x]
        label_paths = [os.path.join(raw_files_directory, x) for x in raw_image_labels_paths if 'debug_raw_labels' in x]
        for image_path, label_path in tqdm(zip(image_paths, label_paths)):
            crop_raw_image(image_path, label_path, prefix='debug_raw')
        image_paths = [os.path.join(raw_files_directory, x) for x in raw_image_labels_paths if 'debug_subband_images' in x]
        label_paths = [os.path.join(raw_files_directory, x) for x in raw_image_labels_paths if 'debug_subband_labels' in x]
        for image_path, label_path in tqdm(zip(image_paths, label_paths)):
            crop_raw_image(image_path, label_path, prefix='debug_subband')
    elif phase == RESIZE_CROPPED_IMAGES:
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

