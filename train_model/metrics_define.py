from keras import backend as K


def r_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

rmse = RMSE = r_mean_squared_error