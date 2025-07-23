import tensorflow as tf


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

    return tf.reduce_mean(loss_mup + loss_mum + 10 * overlap_penalty)


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

    return tf.reduce_mean(loss_mup + loss_mum + 10 * overlap_penalty + 5e-4 * distance_penalty)


def regular_loss(y_true, y_pred):
    y_muPlus_true, y_muMinus_true = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_muPlus_pred, y_muMinus_pred = tf.split(y_pred, num_or_size_splits=2, axis=1)

    y_muPlus_true = tf.squeeze(y_muPlus_true, axis=1)
    y_muMinus_true = tf.squeeze(y_muMinus_true, axis=1)

    y_muPlus_pred = tf.squeeze(y_muPlus_pred, axis=1)
    y_muMinus_pred = tf.squeeze(y_muMinus_pred, axis=1)

    loss_mup = tf.keras.losses.sparse_categorical_crossentropy(y_muPlus_true, y_muPlus_pred)
    loss_mum = tf.keras.losses.sparse_categorical_crossentropy(y_muMinus_true, y_muMinus_pred)

    return tf.reduce_mean(loss_mup + loss_mum)


def overlap_loss(y_true, y_pred):
    y_muPlus_true, y_muMinus_true = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_muPlus_pred, y_muMinus_pred = tf.split(y_pred, num_or_size_splits=2, axis=1)

    y_muPlus_true = tf.squeeze(y_muPlus_true, axis=1)
    y_muMinus_true = tf.squeeze(y_muMinus_true, axis=1)

    y_muPlus_pred = tf.squeeze(y_muPlus_pred, axis=1)
    y_muMinus_pred = tf.squeeze(y_muMinus_pred, axis=1)

    overlap_penalty = tf.reduce_sum(tf.square(y_muPlus_pred - y_muMinus_pred), axis=-1)

    return tf.reduce_mean(10 * overlap_penalty)
    

def distance_loss(y_true, y_pred):
    y_muPlus_true, y_muMinus_true = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_muPlus_pred, y_muMinus_pred = tf.split(y_pred, num_or_size_splits=2, axis=1)

    y_muPlus_true = tf.squeeze(y_muPlus_true, axis=1)
    y_muMinus_true = tf.squeeze(y_muMinus_true, axis=1)

    y_muPlus_pred = tf.squeeze(y_muPlus_pred, axis=1)
    y_muMinus_pred = tf.squeeze(y_muMinus_pred, axis=1)

    elementIDs = tf.range(tf.shape(y_muPlus_pred)[-1], dtype=tf.float32)
    y_muPlus_pred = tf.reduce_sum(y_muPlus_pred * elementIDs, axis=-1)
    y_muMinus_pred = tf.reduce_sum(y_muMinus_pred * elementIDs, axis=-1)

    y_muPlus_true = tf.cast(y_muPlus_true, dtype=tf.float32)
    y_muMinus_true = tf.cast(y_muMinus_true, dtype=tf.float32)

    distance_penalty = (
        tf.square(y_muPlus_pred - y_muPlus_true) + 
        tf.square(y_muMinus_pred - y_muMinus_true)
    )

    return tf.reduce_mean(5e-4 * distance_penalty)
