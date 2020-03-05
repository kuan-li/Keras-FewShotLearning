import tensorflow as tf


def yolo_loss(anchors, threshold):
    """

    Args:
        anchors (pandas.DataFrame): dataframe of the anchors with width and height columns.
        threshold:

    """

    @tf.function
    def _yolo_loss(y_true, y_pred):
        """
        y_true and y_pred are (batch_size, number of boxes, 4 (+ 1) + number of classes (+ anchor_id for y_pred)).
        The number of boxes is determined by the network architecture as in single-shot detection one can only predict
        grid_width x grid_height boxes per anchor.
        """
        loss_coordinates = tf.Variable(0.0)
        loss_box = tf.Variable(0.0)
        loss_objectness = tf.Variable(0.0)
        loss_classes = tf.Variable(0.0)

        for image, pred in zip(y_true, y_pred):
            loss_objectness.assign_add(
                tf.math.reduce_sum(tf.keras.backend.binary_crossentropy(tf.zeros_like(y_pred[..., 4]), y_pred[..., 4])))
            for box in image:
                if box[4] < 1:
                    continue
                height_width_min = tf.minimum(box[2:4], anchors[['height', 'width']].values)
                height_width_max = tf.maximum(box[2:4], anchors[['height', 'width']].values)
                intersection = tf.reduce_prod(height_width_min, axis=-1)
                union = tf.reduce_prod(height_width_max, axis=-1)
                iou = intersection / union
                best_iou = tf.reduce_max(iou)
                for i, iou_ in enumerate(iou):
                    if iou_ < threshold:
                        continue
                    selected_anchor_map = pred[pred[..., -1] == tf.cast(i, pred.dtype)]
                    selected_cell = tf.argmin(tf.norm(box[:2] - selected_anchor_map[..., :2], axis=1))
                    selected_pred = selected_anchor_map[selected_cell]
                    loss_objectness.assign_sub(tf.keras.backend.binary_crossentropy(0.0, selected_pred[4]))

                    if iou_ == best_iou:
                        loss_objectness.assign_add(tf.keras.backend.binary_crossentropy(box[4], selected_pred[4]))
                        loss_coordinates.assign_add(tf.norm(box[:2] - selected_pred[:2], ord=2))
                        loss_box.assign_add(tf.norm(box[2:4] - selected_pred[2:4], ord=2))
                        loss_classes.assign_add(tf.reduce_sum(tf.keras.backend.binary_crossentropy(box[5:], selected_pred[5:-1])))

        return loss_coordinates + loss_box + loss_objectness + loss_classes

    return _yolo_loss
