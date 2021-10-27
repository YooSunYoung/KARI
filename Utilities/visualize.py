import numpy as np
import cv2
import os


def draw_raw_image(image_path, label_path, predicted_labels=None):
    image = np.load(image_path)
    labels = np.load(label_path)
    image_shape = image.shape
    for label in labels:
        x1y1 = (int(label[0]-label[2]/2), int(label[1]-label[2]/2))
        x2y2 = (int(label[0]+label[2]/2), int(label[1]+label[2]/2))
        image = cv2.rectangle(image, x1y1, x2y2, (255, 255, 255), 50)
    if predicted_labels is not None:
        for pred_label in predicted_labels:
            x1y1 = (pred_label[0]-pred_label[2]/2, pred_label[1]-pred_label[2]/2)
            x2y2 = (pred_label[0]+pred_label[2]/2, pred_label[1]+pred_label[2]/2)
            image = cv2.rectangle(image, x1y1, x2y2, (255, 0, 0), 2)

    image = cv2.resize(image, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('sample.png', image)


def draw_cropped_images(target_dir='../data/cropped_samples',
                        cropped_image_path='../data/cropped_raw/cropped_debug_raw_images_0.npz',
                        cropped_label_path='../data/cropped_raw/cropped_debug_raw_labels_0.npz',
                        predicted_labels=None):
    images = np.load(cropped_image_path, allow_pickle=True)
    images = images[:, :, :, :3]
    label_list = np.load(cropped_label_path, allow_pickle=True)
    if cropped_image_path.endswith('npz'):
        images = images.get('arr_0')
        label_list = label_list.get('arr_0')
    image_number = cropped_image_path.split('_')[-1].split('.')[0]
    for i_data, (image, labels) in enumerate(zip(images, label_list)):
        for label in labels:
            x1y1 = (int(label[0]-label[2]/2), int(label[1]-label[2]/2))
            x2y2 = (int(label[0]+label[2]/2), int(label[1]+label[2]/2))
            # images[i_data] = cv2.rectangle(image, x1y1, x2y2, (255, 255, 255), 50)
    if predicted_labels is not None:
        for i_data, (image, pred_labels) in enumerate(zip(images, predicted_labels)):
            for pred_label in pred_labels:
                x1y1 = (pred_label[0]-pred_label[2]/2, pred_label[1]-pred_label[2]/2)
                x2y2 = (pred_label[0]+pred_label[2]/2, pred_label[1]+pred_label[2]/2)
                # images[i_data] = cv2.rectangle(image, x1y1, x2y2, (255, 0, 0), 2)

    for i_image, image in enumerate(images):
        image = cv2.resize(image, dsize=(0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
        image_path = os.path.join(target_dir, 'sample_{}_{}.png'.format(image_number, i_image))
        cv2.imwrite(image_path, image)


if __name__ == "__main__":
    # draw_raw_image('../data/raw/raw_images_0.npy', '../data/raw/raw_labels_0.npy')
    # draw_cropped_images()
    draw_cropped_images(target_dir='../data/resized_cropped_samples',
                        cropped_image_path='../data/resized_cropped_raw/cropped_resized_debug_subband_images_0.npy',
                        cropped_label_path='../data/resized_cropped_raw/cropped_resized_debug_subband_labels_0.npy')
