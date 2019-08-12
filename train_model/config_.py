

'''
Default configurations.
'''
'''
参考：
    the learning rate was searched in [0.005, 0.01, 0.02, 0.05]
    the dropout ratio in [0.1, 0.3, 0.5, 0.7, 0.9]
    the batch size was tested in [50as, 100, 150]
    the latent factor number was tested in [8, 16, 32, 64]
'''

configs = {
    'batch_size':32,  # Batch size for training.
    'epochs': 300,  # Number of epochs to train for.
    'max_features': 300,  # 输出文本向量的维度？doc2vec的结果是300维
    'latent_dim': 64,  # lstm的输出维度
    'num_decoder_tokens': 57,  # 解码之后的向量的维度,98 , 57
    'timestep_user': 17,  # 用户lstm的个数，根据用户涉及的最大月数得到,113,,16
    'timestep_item': 16,  # item的lstm的个数，根据item涉及的最大月数得到,test中的一段代码统计得到84,,15
    'review_vec_dim': 300,  # 将评论转换为向量的维度
    # 'user_rnn_max_dim': 15,
    # 'item_rnn_max_dim': 10,
    'user_num': 24303,  # user的个数，在item_based 的地方需要用,5541
    'item_num': 10672,  # item的个数，在user_based 的地方需要用 3568
    'dropout': 0.3,
    'lr': 0.0005,
    'path': r'/home/yneversky/lhr/lo/targetfolder/',
    'pathmiddle': r'/home/yneversky/lhr/lo/targetfolder/middlefile/'
}
