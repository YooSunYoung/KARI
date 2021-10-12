import numpy as np
from Utilities.loader import *


def crop_image(large_image: np.ndarray, labelX, labelY, labelClass, cropped_image_width=2048, cropped_image_height=2048):
    target_size = np.array([cropped_image_height, cropped_image_width])
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

# def crop_raw_images(raw_img_array, labels):
#     cropped_images, labels, offsets = crop_image(raw_img_array, label_df['IDX_X_SLC'].to_numpy(),
#                                                  label_df['IDX_Y_SLC'].to_numpy(),
#                                                  label_df['CLASS_ID'].to_numpy())
#
#     if debug:
#         row_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#         sample_large_image = np.zeros(raw_img_array.shape)
#         for irow, row in enumerate(raw_img_array):
#             for icol, col in enumerate(row):
#                 sample_large_image[irow][icol] = [sqrt(col[0] ** 2 + col[1] ** 2)]
#         for i_row, offset_row in enumerate(offsets):
#             for offset in offset_row:
#                 offset = np.array(offset)
#                 sample_large_image = cv2.rectangle(sample_large_image, offset, offset + np.array([2048, 2048]),
#                                                    row_colors[i_row % 3], 2)
#         sample_large_image = cv2.resize(sample_large_image, dsize=(0, 0), fx=0.025, fy=0.025,
#                                         interpolation=cv2.INTER_LINEAR)
#         cv2.imshow('sample', sample_large_image)
#         cv2.waitKey(0)
#         for row_img in cropped_images:
#             for img in row_img:
#                 img = np.array(list(map(lambda xy: [sqrt(xy[0] ** 2 + xy[1] ** 2)], img)))
#                 cv2.imshow("sample", img)
#                 cv2.waitKey(0)
#     cropped_images = np.array(cropped_images)
#     shape_ = cropped_images.shape
#     cropped_images = np.reshape(cropped_images, (shape_[0] * shape_[1], shape_[2], shape_[3], shape_[4]))
#     labels = np.array(labels)
#     image_path = os.path.join(target_dir, region, period, 'raw_image.npy')


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