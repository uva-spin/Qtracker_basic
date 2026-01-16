import tensorflow as tf


def residual(y_true, y_pred):
    # MAE
    y_muPlus_true, y_muMinus_true = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_muPlus_pred, y_muMinus_pred = tf.split(y_pred, num_or_size_splits=2, axis=1)

    y_muPlus_true = tf.cast(tf.squeeze(y_muPlus_true, axis=1), tf.float32)
    y_muMinus_true = tf.cast(tf.squeeze(y_muMinus_true, axis=1), tf.float32)

    y_muPlus_pred = tf.cast(tf.argmax(tf.squeeze(y_muPlus_pred, axis=1), axis=-1), tf.float32)
    y_muMinus_pred = tf.cast(tf.argmax(tf.squeeze(y_muMinus_pred, axis=1), axis=-1), tf.float32)

    res_plus = tf.abs(y_muPlus_true - y_muPlus_pred)
    res_minus = tf.abs(y_muMinus_true - y_muMinus_pred)

    return tf.reduce_mean((res_plus + res_minus) / 2)


def chi_squared(y_true, y_pred):
    # MSE
    y_muPlus_true, y_muMinus_true = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_muPlus_pred, y_muMinus_pred = tf.split(y_pred, num_or_size_splits=2, axis=1)

    y_muPlus_true = tf.cast(tf.squeeze(y_muPlus_true, axis=1), tf.float32)
    y_muMinus_true = tf.cast(tf.squeeze(y_muMinus_true, axis=1), tf.float32)

    y_muPlus_pred = tf.cast(tf.argmax(tf.squeeze(y_muPlus_pred, axis=1), axis=-1), tf.float32)
    y_muMinus_pred = tf.cast(tf.argmax(tf.squeeze(y_muMinus_pred, axis=1), axis=-1), tf.float32)

    chi_plus = tf.square(y_muPlus_true - y_muPlus_pred)
    chi_minus = tf.square(y_muMinus_true - y_muMinus_pred)

    return tf.reduce_mean((chi_plus + chi_minus) / 2)


def precision(y_true, y_pred):
    # Average distribution variance
    y_muPlus_true, y_muMinus_true = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_muPlus_pred, y_muMinus_pred = tf.split(y_pred, num_or_size_splits=2, axis=1)

    y_muPlus_pred = tf.squeeze(y_muPlus_pred, axis=1)
    y_muMinus_pred = tf.squeeze(y_muMinus_pred, axis=1)

    def average_variance(pred):
        # Calculate average variance for each event
        num_elements = tf.shape(pred)[2]
        i = tf.cast(tf.range(num_elements), tf.float32)

        # Calculate mean and variance for each detector (distribution of element softmax)
        mean = tf.reduce_sum(pred * i, axis=2)
        variance = tf.reduce_sum(pred * tf.square(i - tf.expand_dims(mean, axis=2)), axis=2)
        avg_var = tf.reduce_mean(variance, axis=1)

        return avg_var     # shape: (batch_size,)

    var_plus = average_variance(y_muPlus_pred)
    var_minus = average_variance(y_muMinus_pred)
    return tf.reduce_mean((var_plus + var_minus) / 2)   # average variance across all events