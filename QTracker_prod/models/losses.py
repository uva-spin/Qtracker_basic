import tensorflow as tf

OVERLAP_LAMBDA = 0.1
DISTANCE_LAMBDA = 5e-4


def custom_loss(y_true, y_pred):
    y_muPlus_true, y_muMinus_true = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_muPlus_pred, y_muMinus_pred = tf.split(y_pred, num_or_size_splits=2, axis=1)

    y_muPlus_true = tf.squeeze(y_muPlus_true, axis=1)
    y_muMinus_true = tf.squeeze(y_muMinus_true, axis=1)

    y_muPlus_pred = tf.squeeze(y_muPlus_pred, axis=1)
    y_muMinus_pred = tf.squeeze(y_muMinus_pred, axis=1)

    loss_mup = tf.keras.losses.sparse_categorical_crossentropy(y_muPlus_true, y_muPlus_pred)
    loss_mum = tf.keras.losses.sparse_categorical_crossentropy(y_muMinus_true, y_muMinus_pred)

    overlap_penalty = tf.reduce_sum(tf.square(y_muPlus_pred - y_muMinus_pred), axis=-1)

    return tf.reduce_mean(loss_mup + loss_mum + OVERLAP_LAMBDA * overlap_penalty)


def custom_loss_v2(y_true, y_pred):
    y_muPlus_true, y_muMinus_true = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_muPlus_pred, y_muMinus_pred = tf.split(y_pred, num_or_size_splits=2, axis=1)

    y_muPlus_true = tf.squeeze(y_muPlus_true, axis=1)
    y_muMinus_true = tf.squeeze(y_muMinus_true, axis=1)

    y_muPlus_pred = tf.squeeze(y_muPlus_pred, axis=1)
    y_muMinus_pred = tf.squeeze(y_muMinus_pred, axis=1)

    loss_mup = tf.keras.losses.sparse_categorical_crossentropy(y_muPlus_true, y_muPlus_pred)
    loss_mum = tf.keras.losses.sparse_categorical_crossentropy(y_muMinus_true, y_muMinus_pred)

    overlap_penalty = tf.reduce_sum(tf.square(y_muPlus_pred - y_muMinus_pred), axis=-1)

    elementIDs = tf.range(tf.shape(y_muPlus_pred)[-1], dtype=tf.float32)
    y_muPlus_pred = tf.reduce_sum(y_muPlus_pred * elementIDs, axis=-1)
    y_muMinus_pred = tf.reduce_sum(y_muMinus_pred * elementIDs, axis=-1)

    distance_penalty = (
        tf.square(y_muPlus_pred - y_muPlus_true) + 
        tf.square(y_muMinus_pred - y_muMinus_true)
    )

    return tf.reduce_mean(loss_mup + loss_mum + OVERLAP_LAMBDA * overlap_penalty + DISTANCE_LAMBDA * distance_penalty)

def cross_validation_loss(y_trues, y_preds):
    """
    Compute the mean sparse categorical crossentropy loss across folds.
    Assumes:
        - y_trues[i]: ground truth tensor for fold i,
                      shaped (batch, 2, det)  where axis=1 holds [mu+, mu-].
        - y_preds[i]: prediction tensor for fold i,
                      shaped (batch, det, elem, 2).
    Returns:
        Scalar tensor: average loss across all folds.
    """
    fold_losses = []

    for y_true, y_pred in zip(y_trues, y_preds):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)

        # Split true labels into mu+ and mu-
        y_muPlus_true, y_muMinus_true = tf.split(y_true, 2, axis=1)
        y_muPlus_true = tf.squeeze(y_muPlus_true, axis=1)   # (batch, det)
        y_muMinus_true = tf.squeeze(y_muMinus_true, axis=1) # (batch, det)

        # Split predictions into mu+ and mu- along the last axis
        y_muPlus_pred, y_muMinus_pred = tf.split(y_pred, 2, axis=-1)
        y_muPlus_pred = tf.squeeze(y_muPlus_pred, axis=-1)   # (batch, det, elem)
        y_muMinus_pred = tf.squeeze(y_muMinus_pred, axis=-1) # (batch, det, elem)

        # Compute sparse categorical crossentropy for each
        loss_mup = tf.keras.losses.sparse_categorical_crossentropy(y_muPlus_true, y_muPlus_pred)
        loss_mum = tf.keras.losses.sparse_categorical_crossentropy(y_muMinus_true, y_muMinus_pred)

        # Mean loss for this fold
        fold_loss = tf.reduce_mean(loss_mup + loss_mum)
        fold_losses.append(fold_loss)

    # Average across folds
    return tf.reduce_mean(tf.stack(fold_losses))