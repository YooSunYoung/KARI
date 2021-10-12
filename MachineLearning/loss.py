import tensorflow as tf


def GenerateGrid(grid_size):
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # the grid shape becomes (gridSize,gridSize,1,2)
    grid = tf.tile(grid, tf.constant([1, 1, 3, 1], tf.int32) )  # the grid shape becomes (gridSize,gridSize,3,2)
    grid = tf.cast(grid, tf.float32)
    return grid


def compute_iou(boxes1, boxes2):
    boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                         boxes1[..., 1] - boxes1[..., 3] / 2.0,
                         boxes1[..., 0] + boxes1[..., 2] / 2.0,
                         boxes1[..., 1] + boxes1[..., 3] / 2.0],
                        axis=-1)

    boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                         boxes2[..., 1] - boxes2[..., 3] / 2.0,
                         boxes2[..., 0] + boxes2[..., 2] / 2.0,
                         boxes2[..., 1] + boxes2[..., 3] / 2.0],
                        axis=-1)
    lu = tf.maximum(tf.cast(boxes1_t[..., :2], dtype=tf.float32), tf.cast(boxes2_t[..., :2], dtype=tf.float32))
    rd = tf.minimum(tf.cast(boxes1_t[..., 2:], dtype=tf.float32), tf.cast(boxes2_t[..., 2:], dtype=tf.float32))

    intersection = tf.maximum(0.0, rd - lu)
    inter_square = intersection[..., 0] * intersection[..., 1]

    square1 = boxes1[..., 2] * boxes1[..., 3]
    square2 = boxes2[..., 2] * boxes2[..., 3]

    union_square = tf.maximum(tf.cast(square1,dtype=tf.float32) + tf.cast(square2,dtype=tf.float32) - tf.cast(inter_square,dtype=tf.float32), 1e-10)
    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


def get_loss(y_true=None, y_pred=None, train_state=True, grid_size=4, n_boxes=3,
             point_loss_factor=10, size_loss_factor=13, loc_loss_factor=1.5,
             obj_loss_factor=2, non_obj_loss_factor=0.5):
    grid = GenerateGrid(grid_size=4)
    pred_obj_conf = y_pred[:, :, :, :, 0]
    pred_box_offset_coord = y_pred[:, :, :, :, 1:]

    pred_box_normalized_coord = tf.concat([(pred_box_offset_coord[:, :, :, :, 0:2] + grid) / grid_size,
                                           tf.square(pred_box_offset_coord[:, :, :, :, 2:])], axis=-1)

    target_obj_conf = y_true[:, :, :, 0]
    target_obj_conf = tf.reshape(target_obj_conf, shape=[-1, grid_size, grid_size, 1])
    target_obj_conf = tf.cast(target_obj_conf, dtype=tf.float32)

    target_box_coord = y_true[:, :, :, 1:]
    target_box_coord = tf.reshape(target_box_coord, shape=[-1, grid_size, grid_size, 1, 4])
    target_box_coord_aT = tf.tile(target_box_coord, multiples=[1, 1, 1, n_boxes, 1])
    target_box_normalized_coord = target_box_coord_aT

    target_box_offset_coord = tf.concat(
        [tf.cast(target_box_normalized_coord[:, :, :, :, 0:2] * grid_size, dtype=tf.float32) - grid,
         tf.cast(tf.sqrt(target_box_normalized_coord[:, :, :, :, 2:]), dtype=tf.float32), ], axis=-1)

    pred_ious = compute_iou(target_box_normalized_coord, pred_box_normalized_coord)
    predictor_mask_max = tf.reduce_max(pred_ious, axis=-1, keepdims=True)
    predictor_mask = tf.cast(pred_ious >= tf.cast(predictor_mask_max, dtype=tf.float32),
                             tf.float32) * target_obj_conf
    noobj_mask = tf.ones_like(predictor_mask) - predictor_mask

    # computing the confidence loss
    obj_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(predictor_mask * (pred_obj_conf - predictor_mask)), axis=[1, 2, 3]))
    noobj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobj_mask * pred_obj_conf), axis=[1, 2, 3]))

    # computing the localization loss
    predictor_mask_none = predictor_mask[:, :, :, :, None]
    loc_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(predictor_mask_none * (target_box_offset_coord - pred_box_offset_coord)),
                      axis=[1, 2, 3]))

    point_loss = tf.reduce_mean(tf.reduce_sum(tf.square(
        predictor_mask_none * (target_box_offset_coord[:, :, :, :, 1:3] - pred_box_offset_coord[:, :, :, :, 1:3]))))
    size_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(
            predictor_mask_none * (target_box_offset_coord[:, :, :, :, 3:] - pred_box_offset_coord[:, :, :, :, 3:]))))

    loc_loss = point_loss_factor * point_loss + size_loss_factor * size_loss
    loss = loc_loss_factor * loc_loss + obj_loss_factor * obj_loss + non_obj_loss_factor * noobj_loss

    if train_state is True:
        tf.summary.scalar("loc_loss", K.sum(10 * loc_loss))
        tf.summary.scalar("obj_loss", K.sum(2 * obj_loss))
        tf.summary.scalar("nonObj_loss", K.sum(0.5 * noobj_loss))
        tf.summary.scalar("total_loss", K.sum(loss))
    return loss