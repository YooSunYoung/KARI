import os
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np


def preprocessing(image_data, final_height, final_width, apply_augmentation=False, evaluate=False):
    """Image resizing operation handled before batch operations.
    inputs:
        image_data = tensorflow dataset image_data
        final_height = final image height after resizing
        final_width = final image width after resizing
    outputs:
        img = (final_height, final_width, channels)
        gt_boxes = (gt_box_size, [y1, x1, y2, x2])
        gt_labels = (gt_box_size)
    """
    img = image_data["image"]
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    gt_boxes = image_data["objects"]["bbox"]
    gt_labels = tf.cast(image_data["objects"]["label"] + 1, tf.int32)
    if evaluate:
        not_diff = tf.logical_not(image_data["objects"]["is_difficult"])
        gt_boxes = gt_boxes[not_diff]
        gt_labels = gt_labels[not_diff]
    img = np.asarray(img).astype('float32')
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (final_height, final_width))
    if apply_augmentation:
        img, gt_boxes = randomly_apply_operation(flip_horizontally, img, gt_boxes)
    return img, gt_boxes, gt_labels


def get_random_bool():
    """Generating random boolean.
    outputs:
        random boolean 0d tensor
    """
    return tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5)

def randomly_apply_operation(operation, img, gt_boxes):
    """Randomly applying given method to image and ground truth boxes.
    inputs:
        operation = callable method
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    outputs:
        modified_or_not_img = (final_height, final_width, depth)
        modified_or_not_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    return tf.cond(
        get_random_bool(),
        lambda: operation(img, gt_boxes),
        lambda: (img, gt_boxes)
    )

def flip_horizontally(img, gt_boxes):
    """Flip image horizontally and adjust the ground truth boxes.
    inputs:
        img = (height, width, depth)
        gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    outputs:
        modified_img = (height, width, depth)
        modified_gt_boxes = (ground_truth_object_count, [y1, x1, y2, x2])
    """
    flipped_img = tf.image.flip_left_right(img)
    flipped_gt_boxes = tf.stack([gt_boxes[..., 0],
                                1.0 - gt_boxes[..., 3],
                                gt_boxes[..., 2],
                                1.0 - gt_boxes[..., 1]], -1)
    return flipped_img, flipped_gt_boxes


def get_dataset(name, split, data_dir="~/tensorflow_datasets"):
    """Get tensorflow dataset split and info.
    inputs:
        name = name of the dataset, voc/2007, voc/2012, etc.
        split = data split string, should be one of ["train", "validation", "test"]
        data_dir = read/write path for tensorflow datasets
    outputs:
        dataset = tensorflow dataset split
        info = tensorflow dataset info
    """
    assert split in ["train", "train+validation", "validation", "test"]
    dataset, info = tfds.load(name, split=split, data_dir=data_dir, with_info=True)
    return dataset, info


def get_raw_slc_dataset(root_dir='F:/1_cropped_numpy', image_type_dir='2_subband',
                    image_type='subband', data_type='selected', batch_size=4,
                    final_width=500, final_height=500):
    image_dir = os.path.join(root_dir, image_type_dir)
    file_list = os.listdir(image_dir)
    image_path_list = [os.path.join(image_dir, x) for x in file_list if 'images' in x]
    image_path_list = image_path_list[:2]
    label_path_list = []
    for image_path in image_path_list:
        label_path_list.append(image_path.replace('images', 'labels'))
    image_dataset = []
    for image_path, label_path in zip(image_path_list, label_path_list):
        cropped_images = np.load(image_path, allow_pickle=True)
        cropped_labels = np.load(label_path, allow_pickle=True)
        for image, label in zip(cropped_images, cropped_labels):
            if len(label) == 0:
                continue
            img = np.array(image)[:, :, :3].astype('float32')
            bboxes = np.array(label)[:, :4]
            clses = np.array(label)[:, 4]
            translated_bboxes = []
            translated_clses = []
            for ib, (bbox, cls) in enumerate(zip(bboxes, clses)):
                if cls >= 3 and bbox[2] > 50:
                    x1 = (bbox[0]-bbox[2]/2)/img.shape[0]
                    y1 = (bbox[1]-bbox[3]/2)/img.shape[1]
                    x2 = (bbox[0]+bbox[2]/2)/img.shape[0]
                    y2 = (bbox[1]+bbox[3]/2)/img.shape[1]
                    translated_bboxes.append(np.asarray([y1, x1, y2, x2]).astype('float32'))
                    translated_clses.append(1)
            if len(translated_bboxes) == 0:
                continue
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, (final_width, final_height))
            image_dataset.append((img, tf.cast(translated_bboxes, dtype=tf.float32),
                                  tf.cast(translated_clses, dtype=tf.int32)))

    return image_dataset


def get_slc_dataset(root_dir='F:/1_cropped_numpy', image_type_dir='2_subband',
                    image_type='subband', data_type='selected', batch_size=4,
                    final_width=500, final_height=500):
    image_dir = os.path.join(root_dir, image_type_dir)
    file_list = os.listdir(image_dir)
    image_path_list = [os.path.join(image_dir, x) for x in file_list if 'images' in x]
    image_path_list = image_path_list[:1]
    label_path_list = []
    for image_path in image_path_list:
        label_path_list.append(image_path.replace('images', 'labels'))
    image_dataset = []
    for image_path, label_path in zip(image_path_list, label_path_list):
        cropped_images = np.load(image_path, allow_pickle=True)
        cropped_labels = np.load(label_path, allow_pickle=True)
        for image, label in zip(cropped_images, cropped_labels):
            if len(label) == 0:
                continue
            img = np.array(image)[:, :, :3].astype('float32')
            bboxes = np.array(label)[:, :4]
            clses = np.array(label)[:, 4]
            translated_bboxes = []
            translated_clses = []
            for ib, (bbox, cls) in enumerate(zip(bboxes, clses)):
                if cls >= 3 and bbox[2] > 30:
                    x1 = (bbox[0]-bbox[2]/2)/img.shape[0]
                    y1 = (bbox[1]-bbox[3]/2)/img.shape[1]
                    x2 = (bbox[0]+bbox[2]/2)/img.shape[0]
                    y2 = (bbox[1]+bbox[3]/2)/img.shape[1]
                    translated_bboxes.append(np.asarray([y1, x1, y2, x2]).astype('float32'))
                    translated_clses.append(1)
            if len(translated_bboxes) == 0:
                continue
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.per_image_standardization(
                tf.image.adjust_contrast(
                        tf.image.resize(img, (final_width, final_height)), 0.7))
            image_dataset.append((img, tf.cast(translated_bboxes, dtype=tf.float32),
                                  tf.cast(translated_clses, dtype=tf.int32)))

    return image_dataset


def get_total_item_size(info, split):
    """Get total item size for given split.
    inputs:
        info = tensorflow dataset info
        split = data split string, should be one of ["train", "validation", "test"]
    outputs:
        total_item_size = number of total items
    """
    assert split in ["train", "train+validation", "validation", "test"]
    if split == "train+validation":
        return info.splits["train"].num_examples + info.splits["validation"].num_examples
    return info.splits[split].num_examples

def get_labels(info):
    """Get label names list.
    inputs:
        info = tensorflow dataset info
    outputs:
        labels = [labels list]
    """
    return info.features["labels"].names

def get_custom_imgs(custom_image_path):
    """Generating a list of images for given path.
    inputs:
        custom_image_path = folder of the custom images
    outputs:
        custom image list = [path1, path2]
    """
    img_paths = []
    for path, dir, filenames in os.walk(custom_image_path):
        for filename in filenames:
            img_paths.append(os.path.join(path, filename))
        break
    return img_paths


def custom_data_generator(images, final_height, final_width):
    """Yielding custom entities as dataset.
    inputs:
        img_paths = custom image paths
        final_height = final image height after resizing
        final_width = final image width after resizing
    outputs:
        img = (final_height, final_width, depth)
        dummy_gt_boxes = (None, None)
        dummy_gt_labels = (None, )
    """
    # for img_path in images:
    #     image = Image.open(images)
    for image in images:
        resized_image = image.resize((final_width, final_height), Image.LANCZOS)
        img = np.array(resized_image)
        img = tf.image.convert_image_dtype(img, tf.float32)
        yield img, tf.constant([[]], dtype=tf.float32), tf.constant([], dtype=tf.int32)

def get_data_types():
    """Generating data types for tensorflow datasets.
    outputs:
        data types = output data types for (images, ground truth boxes, ground truth labels)
    """
    return (tf.float32, tf.float32, tf.int32)

def get_data_shapes():
    """Generating data shapes for tensorflow datasets.
    outputs:
        data shapes = output data shapes for (images, ground truth boxes, ground truth labels)
    """
    return ([None, None, None], [None, None], [None, ])

def get_padding_values():
    """Generating padding values for missing values in batch for tensorflow datasets.
    outputs:
        padding values = padding values with dtypes for (images, ground truth boxes, ground truth labels)
    """
    return (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))