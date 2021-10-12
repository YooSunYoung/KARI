from Utilities.loader import *
from Utilities.ImageUtilities import *
from math import sqrt
import os


def crop_image(large_image: np.ndarray, original_labels: np.ndarray, cropped_image_width=2048, cropped_image_height=2048):
    target_size = np.array([cropped_image_height, cropped_image_width])
    split_images = split_2d_image(large_image, target_size)
    cropped_images = split_images['images']
    labels = []
    for i_row, offset_row in enumerate(split_images['offsets']):
        row_labels = []
        for offset in offset_row:
            image_labels = []
            for x, y, obj_size, cls in original_labels:
                if (offset[0] <= x <= offset[0]+cropped_image_width) \
                    and (offset[1] <= y <= offset[1]+cropped_image_height):
                    x -= offset[0]
                    y -= offset[1]
                    image_labels.append((x, y, obj_size, obj_size, cls))
            row_labels.append(image_labels)
        labels.extend(row_labels)
    return cropped_images, labels, split_images['offsets']


def save_raw_image_to_numpy(dir_str, debug=False, target_dir='../data/raw/'):
    num_image = 0
    for ir, region in enumerate(dir_str):
        for ip, period in enumerate(dir_str[region]):
            raw = dir_str[region][period]['RAW']
            csv_path = dir_str[region][period]['TARGET_CSV']
            h5_img = get_raw(raw)
            raw_img_array = h5_img['S01']['SBI'][:]
            raw_img_shape = raw_img_array.shape
            raw_img_array = np.reshape(raw_img_array, (raw_img_shape[0]*raw_img_shape[1], raw_img_shape[2]))
            raw_img_array = np.array(list(map(lambda x: [x[0], x[1], sqrt(x[0]**2+x[1]**2)], raw_img_array)))
            raw_img_array = np.reshape(raw_img_array, (raw_img_shape[0], raw_img_shape[1], 3))
            label_df = get_label(csv_path)
            label_df.columns = ['TARGET_ID', 'CLASS_ID', 'OBJECT_SIZE', 'PEAK_VAL', 'MEAN_VAL',
                                'IDX_X_SLC', 'IDX_Y_SLC', 'IDX_X_IMG', 'IDX_Y_IMG', 'VALID']
            raw_label_df = label_df[['IDX_X_SLC', 'IDX_Y_SLC', 'OBJECT_SIZE', 'CLASS_ID']]
            raw_label = []
            for label in label_df.values:
                raw_label.append(label)
            np.save(os.path.join(target_dir, 'raw_images_{}.npy'.format(num_image)), raw_img_array)
            np.save(os.path.join(target_dir, 'raw_labels_{}.npy'.format(num_image)), raw_label_df.to_numpy())
            num_image += 1
            if debug:
                break


def save_full_band_image_to_numpy(dir_str, debug=False, target_dir='../data/raw/'):
    num_image = 0
    for ir, region in enumerate(dir_str):
        for ip, period in enumerate(dir_str[region]):
            sub_slc = dir_str[region][period]['SUB_BAND_SLC_HEADER']
            csv_path = dir_str[region][period]['TARGET_CSV']
            subband_slc_array = get_raw_slc(None, sub_slc)
            label_df = get_label(csv_path)
            label_df.columns = ['TARGET_ID', 'CLASS_ID', 'OBJECT_SIZE', 'PEAK_VAL', 'MEAN_VAL',
                                'IDX_X_SLC', 'IDX_Y_SLC', 'IDX_X_IMG', 'IDX_Y_IMG', 'VALID']
            raw_label_df = label_df[['IDX_X_IMG', 'IDX_Y_IMG', 'OBJECT_SIZE', 'CLASS_ID']]
            raw_label = []
            for label in label_df.values:
                raw_label.append(label)
            np.save(os.path.join(target_dir, 'subband_images_{}.npy'.format(num_image)), subband_slc_array)
            np.save(os.path.join(target_dir, 'subband_labels_{}.npy'.format(num_image)), raw_label_df.to_numpy())
            num_image += 1
            if debug:
                break


def crop_raw_image(raw_image_path='../data/raw/raw_images_0.npy', raw_label_path='../data/raw/raw_labels_0.npy',
                   target_dir='../data/cropped_raw/', debug=False):
    raw_img_array = np.load(raw_image_path)
    labels = np.load(raw_label_path)
    cropped_images, labels, offsets = crop_image(raw_img_array, labels)
    cropped_images = np.array(cropped_images)
    shape_ = cropped_images.shape
    cropped_images = np.reshape(cropped_images, (shape_[0] * shape_[1], shape_[2], shape_[3], shape_[4]))
    labels = np.array(labels)
    image_number = raw_label_path.split('_')[-1].split('.')[0]
    image_path = os.path.join(target_dir, 'cropped_images_{}.npy'.format(image_number))
    label_path = os.path.join(target_dir, 'cropped_labels_{}.npy'.format(image_number))
    np.save(image_path, cropped_images, allow_pickle=True)
    np.save(label_path, labels, allow_pickle=True)


def save_images_to_numpy(dir_str: dict, debug=False):
    meta_ = dict()
    # save_raw_image_to_numpy(dir_str, debug)
    save_full_band_image_to_numpy(dir_str, debug)
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


def resize_cropped_images(cropped_image_path='../data/cropped_raw/cropped_images_0.npy',
                          cropped_label_path='../data/cropped_raw/cropped_labels_0.npy',
                          target_width=800, target_height=800,
                          target_dir='../data/cropped_resized_raw/', debug=False):
    cropped_images = np.load(cropped_image_path, allow_pickle=True)
    resized_images = []
    cropped_label_list = np.load(cropped_label_path, allow_pickle=True)
    file_number = cropped_image_path.split('_')[-1].split('.')[0]
    for i_data, (image, labels) in enumerate(zip(cropped_images, cropped_label_list)):
        width = image.shape[0]
        height = image.shape[1]
        width_ratio = target_width/width
        height_ratio = target_height/height
        max_component = np.max(image)
        image = cv2.resize(image, dsize=(target_width, target_height), interpolation=cv2.INTER_LINEAR)
        resized_images.append(image/max_component)
        for i_label, label in enumerate(labels):
            label = [label[0]*width_ratio, label[1]*height_ratio,
                     label[2]*width_ratio, label[3]*height_ratio, label[4]]
            cropped_label_list[i_data][i_label] = label
    resized_images = np.array(resized_images)
    image_path = os.path.join(target_dir, 'cropped_resized_images_{}'.format(file_number))
    label_path = os.path.join(target_dir, 'cropped_resized_labels_{}'.format(file_number))
    np.save(image_path, resized_images)
    np.save(label_path, cropped_label_list)


if __name__ == '__main__':
    phase = 'resize_cropped_raw_images'
    if phase == 'preprocess_raw_images':
        region_period_dirs = get_dirs(debug=True)
        region_period_dirs = get_target_files(region_period_dirs)
        with open('../data/directory_structure.json', 'w') as f:
            json.dump(region_period_dirs, f, indent='\t')
        dir_str = get_directory_structure()
        save_images_to_numpy(dir_str, debug=True)
    elif phase == 'crop_raw_images':
        raw_files_directory = '../data/raw/'
        raw_image_labels_paths = os.listdir(raw_files_directory)
        image_paths = [os.path.join(raw_files_directory, x) for x in raw_image_labels_paths if 'raw_images' in x]
        label_paths = [os.path.join(raw_files_directory, x) for x in raw_image_labels_paths if 'raw_labels' in x]
        for image_path, label_path in zip(image_paths, label_paths):
            crop_raw_image(image_path, label_path)
    elif phase == 'resize_cropped_raw_images':
        files_directory = '../data/cropped_raw/'
        image_labels_paths = os.listdir(files_directory)
        image_paths = [os.path.join(files_directory, x) for x in image_labels_paths if 'cropped_images' in x]
        label_paths = [os.path.join(files_directory, x) for x in image_labels_paths if 'cropped_labels' in x]
        for image_path, label_path in zip(image_paths, label_paths):
            resize_cropped_images(image_path, label_path)