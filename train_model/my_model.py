#coding:utf-8
from keras.layers import Dense, LSTM, Input, dot, Masking, concatenate, Dropout
from keras.models import Model
from myModel.train_model.AttentionLayer import AttentionLayer
from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import optimizers
from myModel.train_model.callback_define import MyCallback,MyCallback_epoch
import h5py
import myModel.train_model.process_data as process_data
# import process_test_data
import myModel.train_model.metrics_define as metrics_define
import myModel.train_model.config_ as config_
import numpy as np
from numpy import array
import os
'''
lstm 编码，全连接层解码

'''
path_save_weight =os.path.join(os.getcwd(), 'savemodel/7')

def build_model():
    '''
    lstm_encoder_decoder_model
    :return:
    '''
    encoder_user_input = Input(shape=(config['timestep_user'], config['review_vec_dim']))  # 每个用户的输入的维度(None,5)
    encoder_user_input_1 = Dropout(config['dropout'], name='encoder_user_input_1')(encoder_user_input)
    encoder_user_masking = Masking(mask_value=0., )(encoder_user_input_1)
    encoder_user = LSTM(config['latent_dim'], return_state=False, dropout=config['dropout'],
                        recurrent_dropout=config['dropout'], name="LSTM_user")  # latent_dim是LSTM层输出向量的维度
    encoder_user_outputs = encoder_user(encoder_user_masking)
    # '''
    # lstm的输出经过一个全连接层进行解码，希望解码之后的结果与计数得到的偏好向量尽可能的相似
    # decoder是解码后的输出，而y_train是计数得到的偏好向量
    # '''
    # decoder_user_1=Dropout(config['dropout'], name='decoder_user')(encoder_user_outputs)
    decoder_user = Dense(config['num_decoder_tokens'], activation='softmax', name="user_decoder")(encoder_user_outputs)
    encoder_item_input = Input(shape=(config['timestep_item'], config['review_vec_dim']))  # 每个用户的输入的维度
    encoder_item_input_1 = Dropout(config['dropout'], name='decoder_item')(encoder_item_input)
    encoder_item_masking = Masking(mask_value=0.)(encoder_item_input_1)
    encoder_item = LSTM(config['latent_dim'], return_state=False, dropout=config['dropout'],
                        recurrent_dropout=config['dropout'], name="LSTM_item")  # latent_dim是LSTM层输出向量的维度
    encoder_item_outputs = encoder_item(encoder_item_masking)
    # decoder_item_1 = Dropout(config['dropout'], name='decoder_item')(encoder_item_outputs)
    decoder_item = Dense(config['num_decoder_tokens'], activation='softmax', name="item_decoder")(encoder_item_outputs)
    '''
    attention_model
    '''

    attention_left_inputs = Input(shape=(config['item_num'],))
    # attention_left_inputs = Input(shape=(4, 4))  # 5表示有5个item
    attention_left_out = AttentionLayer(name='item_based_attention')(attention_left_inputs)
    attention_right_inputs = Input(shape=(config['user_num'],))
    # attention_right_inputs = Input(shape=(3, 3))  # 4表示有4个用户
    attention_right_out = AttentionLayer(name='user_based_attention')(attention_right_inputs)

    user_vec = concatenate([attention_left_out, encoder_user_outputs], axis=-1)
    item_vec = concatenate([attention_right_out, encoder_item_outputs], axis=-1)

    predict_score_output = dot([user_vec, item_vec], axes=-1, name='predict_score_output')
    model = Model([attention_left_inputs, encoder_user_input, encoder_item_input, attention_right_inputs],
                  [decoder_user, decoder_item, predict_score_output])
    adam = optimizers.adam(lr=config['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam,
                  loss='mean_squared_error',
                  loss_weights=[1., 0.2, 0.2],
                  # loss={'predict_score_output': 'mean_squared_error', 'decoder_user': 'mean_squared_error',
                  #       'decoder_item': 'mean_squared_error'},
                  # loss_weights={'predict_score_output': 1., 'decoder_user': 0.2, 'decoder_item': 0.2},
                  metrics=[metrics.mae,metrics_define.rmse])
                  # metrics=[{'predict_score_output': metrics.mae}, {'predict_score_output': metrics_define.rmse}])
    model.summary()
    # Train the model, iterating on the data in batches of batch_size samples
    return model


def train_model(train_user_item_score, train_user_rev, train_item_rev, train_user_out, train_item_out, train_item_based_vec, train_user_based_vec,
                test_user_item_score, test_user_rev, test_item_rev, test_user_out, test_item_out, test_item_based_vec, test_user_based_vec):
    model = build_model()
    # model.load_weights(os.path.join(path_save_weight, 'save-014- 0.21360664-0.84322284-1.07262605.hdf5'))
    print('Train the model, iterating on the data in batches of batch_size samples...')
    filepath = 'save-{epoch:03d}- {loss:.8f}-{val_predict_score_output_mean_absolute_error:.8f}-{val_predict_score_output_r_mean_squared_error:.8f}.hdf5'
    checkpoint = ModelCheckpoint(os.path.join(path_save_weight, filepath), monitor='val_predict_score_output_r_mean_squared_error', mode='min',
                                 save_best_only=True)
    # cb = MyCallback()
    ab = MyCallback_epoch()
    early_stopping = EarlyStopping(monitor='val_predict_score_output_r_mean_squared_error', patience=20,mode='min')
    callback_lists = [checkpoint, early_stopping, ab]

    hist = model.fit_generator(generator_data(config['batch_size'],train_user_item_score, train_user_rev, train_item_rev, train_user_out, train_item_out, train_item_based_vec, train_user_based_vec),
                               steps_per_epoch=len(train_user_item_score)/config['batch_size'],
                               # steps_per_epoch=1,
                               callbacks=callback_lists,
                               validation_data=generator_data(config['batch_size'],test_user_item_score, test_user_rev, test_item_rev, test_user_out, test_item_out, test_item_based_vec, test_user_based_vec),
                               validation_steps=len(train_user_item_score)/config['batch_size'],
                               epochs=config['epochs'])
    with open('log_2','a') as f:
        f.write(str(hist.history))
    print('model.metrics_names',model.metrics_names)
    # score= model.evaluate_generator(generator_data(config['batch_size'],test_user_item_score, test_user_rev, test_item_rev, test_user_out, test_item_out, test_item_based_vec, test_user_based_vec),
    #                                 steps=len(test_user_item_score)/config['batch_size'])
    print('score',hist.history)
    # with open('metric','a') as flie:
    #     flie.writelines('mae:'+mae+'rmse:'+rmse)

    # 取某一层的输出为输出新建为model，采用函数模型
    # lstm_layer_model = Model(inputs = model.input, outputs = model.get_layer('LSTM_user').output)

    # 以这个model的预测值作为输出
    # lstm_output = lstm_layer_model.predict([user_rev, item_rev])

    # 这里的输出为LSTM的输出：当输入的形式为(3,2)的时候，lstm_output的输出形式为[[latent_dim维][latent_dim维][latent_dim维]],[[...][...][...]]

    # print('lstm_output:', lstm_output)

    # 这里的输出是解码之后的输出

    # decoder_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense_decoder').output)
    # decoder_layer_output = decoder_layer_model.predict([user_rev, item_rev])
    # print('lstm_output:', decoder_layer_output)
def generator_data(batch_size,train_user_item_score, train_user_rev, train_item_rev, train_user_out, train_item_out, train_item_based_vec, train_user_based_vec):

    res_x_1 = []
    res_x_2 = []
    res_x_3 = []
    res_x_4 = []
    res_y_1 = []
    res_y_2 = []
    res_y_3 = []
    # item_based_vec, user_rev, item_rev, user_based_vec
    # user_out, item_out, user_item_score
    while True:
        order = np.arange(len(train_user_item_score))
        np.random.shuffle(order)
        order = order[0:batch_size]
        # res_y_1 = train_user_out[order]
        # res_y_2 = train_item_out[order]
        # res_y_3 = train_user_item_score[order]
        # res_x_2 = train_user_rev[order]
        # res_x_3 = train_item_rev[order]
        for i in order:
            res_x_1.append(train_item_based_vec[i])
            res_x_2.append(train_user_rev[i])
            res_x_3.append(train_item_rev[i])
            res_x_4.append(train_user_based_vec[i])
            res_y_1.append(train_user_out[i])
            res_y_2.append(train_item_out[i])
            res_y_3.append(train_user_item_score[i])
        # yield ([array(res_x_2), array(res_x_3)], [array(res_y_1), array(res_y_2), array(res_y_3)])

        yield ([array(res_x_1), array(res_x_2), array(res_x_3), array(res_x_4)], [array(res_y_1), array(res_y_2), array(res_y_3)])
        # print(sys.getsizeof(array(res_x_1)))
        # print(sys.getsizeof(array(res_x_4)))
        # print(res_x_1)
        # print(res_y_3)
        res_x_1 = []
        res_x_2 = []
        res_x_3 = []
        res_x_4 = []
        res_y_1 = []
        res_y_2 = []
        res_y_3 = []


'''
读数据，初始化中间向量
初始化item嵌入矩阵和user嵌入矩阵
在注意力机制部分
用户生成user的嵌入和item的嵌入
'''

if __name__ =='__main__':
    global config
    config = config_.configs
    train_user_item_score, train_user_rev, train_item_rev, train_user_out, train_item_out, train_item_based_vec, train_user_based_vec = process_data.pre_load_data()
    # # predata(path+'inputtext', path+'input_item',path+'output',path+'output_item', path+'score',)
    # # user_item_score, user_rev, item_rev, user_out, item_out = predata_2(path + 'score', path + 'user_item_review.npz', path + 'user_item_out.npz')
    # #
    # print('user_item_score', user_item_score.shape)
    # print('user_rev', user_rev.shape)
    # print('item_rev', item_rev.shape)
    # print('user_out', user_out.shape)
    # print('item_out', item_out.shape)
    # print('item_based_vec', item_based_vec.shape)
    # print('user_based_vec', user_based_vec.shape)
    # # train_lstm_model(user_item_score, user_rev, item_rev, user_out, item_out)
    # # attention_data(path+'user_mutil_hot', path+'item_mutil_hot', path+'score')
    # val_user_item_score, val_user_rev, val_item_rev, val_user_out, val_item_out, val_item_based_vec, val_user_based_vec = process_data.pre_val_load_data()
    test_user_item_score, test_user_rev, test_item_rev, test_user_out, test_item_out, test_item_based_vec, test_user_based_vec = process_data.pre_test_load_data()
    print('train_model')
    train_model(train_user_item_score, train_user_rev, train_item_rev, train_user_out, train_item_out, train_item_based_vec, train_user_based_vec,
                test_user_item_score, test_user_rev, test_item_rev, test_user_out, test_item_out, test_item_based_vec, test_user_based_vec)
