import tensorflow as tf
from utils import io_utils, data_utils, train_utils, bbox_utils, drawing_utils, eval_utils
from models import faster_rcnn
import random
from random import shuffle

train_utils.set_gpu('3')

batch_size = 1
evaluate = True
use_custom_images = False
custom_image_path = "data/images/"

from models.rpn_vgg16 import get_model as get_rpn_model
backbone = 'vgg16'
hyper_params = train_utils.get_hyper_params('vgg16')
image_dataset = data_utils.get_slc_dataset(root_dir='F:/1_cropped_numpy/',
                                           data_type='selected', image_type='subband', image_type_dir='2_subband')

# image_dataset = shuffle(image_dataset)
random.seed(0)
shuffle(image_dataset)
test_dataset = image_dataset[500:600]
total_items = len(test_dataset)
labels = ['ship']
labels = ["bg"] + labels
hyper_params["total_labels"] = len(labels)
img_size = hyper_params["img_size"]

data_types = data_utils.get_data_types()
data_shapes = data_utils.get_data_shapes()
padding_values = data_utils.get_padding_values()

test_data = tf.data.Dataset.from_generator(
    lambda: iter(test_dataset), (tf.float32, tf.float32, tf.int32))
test_data = test_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
#
anchors = bbox_utils.generate_anchors(hyper_params)
rpn_model, feature_extractor = get_rpn_model(hyper_params)
frcnn_model = faster_rcnn.get_model(feature_extractor, rpn_model, anchors, hyper_params, mode="inference")
#
frcnn_model_path = io_utils.get_model_path("faster_rcnn", backbone)
frcnn_model.load_weights(frcnn_model_path)
frcnn_model.summary()
step_size = train_utils.get_step_size(total_items, batch_size)
pred_bboxes, pred_labels, pred_scores = frcnn_model.predict(test_data, steps=step_size, verbose=1)

if evaluate:
    eval_utils.evaluate_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, batch_size)
drawing_utils.draw_predictions(test_data, pred_bboxes, pred_labels, pred_scores, labels, batch_size)
