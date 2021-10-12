import numpy as np
import logging
from math import ceil, floor
import cv2


def split_2d_image(raw_image: np.ndarray, target_size: tuple, overlapped_size=None,
                   TARGET_SIZE_RANDOM=False, CROPPING_MODE=0, EVENLY_SPLIT=True):
    if CROPPING_MODE != 0:
        logging.error("Not available cropping mode.")
        raise(AssertionError())
    # check if the target size is possible.
    for raw_val, tar_val in zip(raw_image.shape, target_size):
        if raw_val <= tar_val:
            logging.warning("Target size should be smaller than the raw image size.")
            logging.warning("Returning the raw image.")
            return {"images": [[raw_image]], "offsets": [[(0, 0)]]}
    raw_image_x_size = raw_image.shape[1]
    raw_image_y_size = raw_image.shape[0]
    raw_image_channel_size = raw_image.shape[2]
    target_x_size = target_size[0]
    target_y_size = target_size[1]
    if EVENLY_SPLIT:
        num_cols = floor(raw_image_x_size/target_x_size)
        num_rows = floor(raw_image_y_size/target_y_size)
        margin_x = raw_image_x_size - (target_x_size*num_cols)
        margin_y = raw_image_y_size - (target_y_size*num_rows)
        if margin_x > 0:
            num_cols += 1
        if margin_y > 0:
            num_rows += 1
        x_interval = int((raw_image_x_size-margin_x)/(num_cols))
        y_interval = int((raw_image_y_size-margin_y)/(num_rows))
    else:
        overlap_x_size = overlapped_size[0]
        overlap_y_size = overlapped_size[1]
        x_interval = target_x_size-overlap_x_size
        num_cols = ceil((raw_image_x_size-target_x_size)/x_interval)+1
        y_interval = target_y_size-overlap_y_size
        num_rows = ceil((raw_image_y_size-target_y_size)/y_interval)+1
    left_upper_coordinates = []
    cropped_images = []
    for i_row in range(num_rows+1):
        row_coord = []
        row_image = []
        y_offset = i_row*y_interval
        if i_row == num_rows:
            y_offset = raw_image_y_size-target_y_size
        for i_col in range(num_cols+1):
            x_offset = i_col * x_interval
            if i_col == num_cols:
                x_offset = raw_image_x_size-target_x_size
            row_coord.append((x_offset, y_offset))
            row_image.append(raw_image[y_offset:y_offset+target_y_size, x_offset:x_offset+target_x_size, :])
        left_upper_coordinates.append(row_coord)
        cropped_images.append(row_image)
    return {"images": cropped_images, "offsets": left_upper_coordinates}


if __name__ == "__main__":
    # sample_large_image = cv2.imread('./sample.jpeg')
    sample_large_image = np.random.rand(3200, 4500, 3)*100/255
    row_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    target_size = np.array([512, 512])
    overlap_size = np.array([30, 50])
    split_images = split_2d_image(sample_large_image, target_size, overlap_size)
    for i_row, offset_row in enumerate(split_images['offsets']):
        for offset in offset_row:
            offset = np.array(offset)
            sample_large_image = cv2.rectangle(sample_large_image, offset, offset + target_size,
                                               row_colors[i_row % 3], 2)
    sample_large_image = cv2.resize(sample_large_image, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("sample", sample_large_image)
    cv2.waitKey(0)
