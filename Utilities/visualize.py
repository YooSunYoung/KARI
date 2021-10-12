import numpy as np
import pandas as pd
import cv2


def draw_rectangles(image_path, label_path, predicted_labels=None):
    image = np.load(image_path)
    labels = np.load(label_path)
    image_shape = image.shape
    image = np.resize(image, (image_shape[0]*image_shape[1], 3))
    image = np.array(list(map(lambda x: [int(x[2])], image)))
    image = np.resize(image, (image_shape[0], image_shape[1], 1))
    for label in labels:
        x1y1 = (label[0]-label[2]/2, label[1]-label[2]/2)
        x2y2 = (label[0]+label[2]/2, label[1]+label[2]/2)
        image = cv2.rectangle(image, x1y1, x2y2,
                              (255, 0, 0), 2)
    if predicted_labels is not None:
        for pred_label in predicted_labels:
            x1y1 = (pred_label[0]-pred_label[2]/2, pred_label[1]-pred_label[2]/2)
            x2y2 = (pred_label[0]+pred_label[2]/2, pred_label[1]+pred_label[2]/2)
            image = cv2.rectangle(image, x1y1, x2y2, (255, 0, 0), 2)

    image = cv2.resize(image, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("sample", image)
    cv2.waitKey(0)


if __name__=="__main__":
    draw_rectangles('../data/raw/raw_images_0.npy', '../data/raw/raw_labels_0.npy')