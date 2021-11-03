import tensorflow as tf
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from MachineLearning.utils import train_utils, bbox_utils, io_utils
from Utilities.loader import get_dataset
from models import faster_rcnn


def set_gpu(gpu_num = '0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
set_gpu('0')


batch_size = 4
epochs = 50
load_weights = False
with_voc_2012 = True
backbone = 'vgg16'


from models.rpn_vgg16 import get_model as get_rpn_model


hyper_params = train_utils.get_hyper_params(backbone)

train_data, labels = get_dataset(image_type='raw', data_type='selected')
gt_data = []
gt_boxes = []
gt_labels = []
for img, label in zip(train_data, labels):
    if len(label) == 0:
        continue
    gt_box_set = []
    gt_label_set = []
    for lab in label:
        if len(lab) == 0:
            continue
        gt_box_coord = [lab[1]-lab[2]/2, lab[0]-lab[2]/2, lab[1]+lab[2]/2, lab[0]+lab[2]/2]
        gt_box_set.append(gt_box_coord)
        gt_label_set.append(lab[-1])
    gt_data.append((img, gt_box_set, gt_label_set))

img = tf.convert_to_tensor([[train_data[0]]], dtype=tf.float32)
train_data = [[img, [[[0.0, 10.0, 0.0, 10.0]]], [[1.0]]]]
# train_data = gt_data

# We add 1 class for background
hyper_params["total_labels"] = 5
#
img_size = hyper_params["img_size"]

anchors = bbox_utils.generate_anchors(hyper_params)
frcnn_train_feed = train_utils.faster_rcnn_generator(train_data, anchors, hyper_params)
# frcnn_val_feed = train_utils.faster_rcnn_generator(val_data, anchors, hyper_params)
#
rpn_model, feature_extractor = get_rpn_model(hyper_params)
frcnn_model = faster_rcnn.get_model(feature_extractor, rpn_model, anchors, hyper_params)
frcnn_model.summary()
frcnn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-5),
                    loss=[None] * len(frcnn_model.output))
faster_rcnn.init_model(frcnn_model, hyper_params)
# If you have pretrained rpn model
# You can load rpn weights for faster training
rpn_load_weights = False
if rpn_load_weights:
    rpn_model_path = io_utils.get_model_path("rpn", backbone)
    rpn_model.load_weights(rpn_model_path)
# Load weights
frcnn_model_path = io_utils.get_model_path("faster_rcnn", backbone)

if load_weights:
    frcnn_model.load_weights(frcnn_model_path)
log_path = ('../log/')

checkpoint_callback = ModelCheckpoint(frcnn_model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
tensorboard_callback = TensorBoard(log_dir=log_path)

frcnn_model.save_weights('./checkpoints/base_model')
# step_size_train = train_utils.get_step_size(train_total_items, batch_size)
# step_size_val = train_utils.get_step_size(val_total_items, batch_size)
frcnn_model.fit(frcnn_train_feed,
                steps_per_epoch=1,
                # validation_data=frcnn_val_feed,
                # validation_steps=step_size_val,
                epochs=epochs,
                callbacks=[checkpoint_callback, tensorboard_callback])
