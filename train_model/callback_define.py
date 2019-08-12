import keras


# 定义callback类
class MyCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        return

    def on_batch_end(self, batch, logs={}):  # batch 为index, logs为当前batch的日志acc, loss...
        self.losses.append(logs.get('loss'))
        with open('log/batchsize150_log','a') as file:
            file.writelines('batchsize:'+str(logs.get('batchsize'))+'---loss:'+str(logs.get('loss'))
                            +'---decoder_user_loss:'+str(logs.get('decoder_user_loss'))
                            +'---decoder_item_loss:'+str(logs.get('decoder_item_loss'))
                            +'---predict_score_output_loss:'+str(logs.get('predict_score_output_loss'))
                            +'---predict_score_output_mean_absolute_error:'+str(logs.get('predict_score_output_mean_absolute_error'))
                            +'---predict_score_output_r_mean_squared_error:'+str(logs.get('predict_score_output_r_mean_squared_error'))+'\n')
        return


class MyCallback_epoch(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        with open('log_val/7','a') as file:
            file.writelines('batchsize:'+str(logs.get('batchsize'))+'---loss:'+str(logs.get('loss'))
                            +'---predict_score_output_loss:'+str(logs.get('predict_score_output_loss'))
                            +'---predict_score_output_mean_absolute_error:'+str(logs.get('predict_score_output_mean_absolute_error'))
                            +'---predict_score_output_r_mean_squared_error:'+str(logs.get('predict_score_output_r_mean_squared_error'))
                            + '---val_loss:' + str(logs.get('val_loss'))
                            + '---val_predict_score_output_loss:' + str(logs.get('val_predict_score_output_loss'))
                            + '---val_predict_score_output_mean_absolute_error:' + str(logs.get('val_predict_score_output_mean_absolute_error'))
                            + '---val_predict_score_output_r_mean_squared_error:' + str(logs.get('val_predict_score_output_r_mean_squared_error'))+'\n')
        return