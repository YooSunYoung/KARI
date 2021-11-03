import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from utils import io_utils, data_utils, train_utils, bbox_utils
from random import shuffle
from models import faster_rcnn
args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()

batch_size = 1
epochs = 50
load_weights = False
with_voc_2012 = False
# backbone = args.backbone
backbone = 'vgg16'
io_utils.is_valid_backbone(backbone)

if backbone == "mobilenet_v2":
    from models.rpn_mobilenet_v2 import get_model as get_rpn_model
else:
    from models.rpn_vgg16 import get_model as get_rpn_model

train_utils.set_gpu('0')
hyper_params = train_utils.get_hyper_params(backbone)
image_dataset = data_utils.get_slc_dataset(root_dir='F:/1_cropped_numpy/',
                                           data_type='selected', image_type='subband', image_type_dir='2_subband')

# image_dataset = shuffle(image_dataset)
train_total_items = 10
val_total_items = 10
train_data = image_dataset
val_data = image_dataset

# labels = data_utils.get_labels(dataset_info)
labels = ['1', '2', '3', '4']
# We add 1 class for background
hyper_params["total_labels"] = len(labels) + 1
#

img_size = hyper_params["img_size"]
# img_size = 256
# train_data = image_dataset.map(lambda x : data_utils.preprocessing(x, img_size, img_size, apply_augmentation=False))
# val_data = image_dataset.map(lambda x : data_utils.preprocessing(x, img_size, img_size))

all_data = tf.data.Dataset.from_generator(
    lambda: iter(image_dataset), (tf.float32, tf.float32, tf.int32))
# train_data = all_data.
# val_data =
data_shapes = data_utils.get_data_shapes()
padding_values = data_utils.get_padding_values()
train_data = all_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
val_data = all_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)

anchors = bbox_utils.generate_anchors(hyper_params)
# frcnn_train_feed = train_utils.faster_rcnn_generator(train_data, anchors, hyper_params)
# frcnn_val_feed = train_utils.faster_rcnn_generator(val_data, anchors, hyper_params)
frcnn_train_feed = train_utils.faster_rcnn_generator(train_data, anchors, hyper_params)
frcnn_val_feed = train_utils.faster_rcnn_generator(train_data, anchors, hyper_params)
#
rpn_model, feature_extractor = get_rpn_model(hyper_params)
frcnn_model = faster_rcnn.get_model(feature_extractor, rpn_model, anchors, hyper_params)
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
log_path = io_utils.get_log_path("faster_rcnn", backbone)

checkpoint_callback = ModelCheckpoint(frcnn_model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
tensorboard_callback = TensorBoard(log_dir=log_path)

step_size_train = train_utils.get_step_size(train_total_items, batch_size)
step_size_val = train_utils.get_step_size(val_total_items, batch_size)
frcnn_model.summary()
frcnn_model.fit(frcnn_train_feed,
                steps_per_epoch=step_size_train,
                epochs=epochs,
                callbacks=[checkpoint_callback, tensorboard_callback],
                validation_data=frcnn_val_feed,
                validation_steps=step_size_val)