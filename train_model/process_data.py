#coding:utf-8

import numpy as np
from numpy import array
import pandas as pd
import csv
import myModel.train_model.config_ as config_
from scipy import sparse
import gc
global config
config = config_.configs

'''
lstm 编码，全连接层解码

'''

def predata(filename1,filename2, filename3, filename4, filename_score,fla):
    '''
    1、user_review :生成字典,从filename1文件中读取user-reviews
        每一行，user-reviewvec
        结果： dict{user1：reviewsvec，user2：reviewsvec}  reviewsvec 是矩阵
    2、item_review :生成字典
    3、user_vec_out：user_out_vec  解码部分的输出
    4、item_vec_out:item解码部分的输出
    5、根据user-item-score生成对齐的数据
    :param filename:
    :return:
    '''
    df = pd.DataFrame(pd.read_csv(filename1).sort_index())
    userdata = df['user']
    itemreviewsvec = df['reviewsvec']
    user_review_dict = {}
    for i in range(len(userdata)):
        user = userdata[i]
        if user in user_review_dict.keys():
            temp = user_review_dict[user]
        else:
            #初始化一个3*5的矩阵
            temp = []
        # temp.append(np.array(reviewsvec[i].split()).astype(float))
        # datamat = np.zeros((1, 5)).astype(float)
        datamat = itemreviewsvec[i].split()
        for index, i in enumerate(datamat):
            datamat[index] = float(i)
        temp.append(array(datamat))
        user_review_dict[user] = temp
        # del datamat
        # del temp
        # gc.collect()
    # print('user_review_dict:', type(user_review_dict[1][0]))
    cou = 1
    for user in user_review_dict.keys():
        print('cou', cou)
        cou = cou + 1
        while len(user_review_dict[user]) < config['timestep_user']:
            datamat = np.zeros((1, 300)).astype(float)
            temp = user_review_dict[user]
            temp.extend(datamat)
            user_review_dict[user] = temp
    #     print(array(user_review_dict[user]))
    # print('user_review_dict', array(user_review_dict))

    # 第二步
    df = pd.DataFrame(pd.read_csv(filename2).sort_index())
    itemdata = df['item']
    userreviewsvec = df['reviewsvec']
    itemdict = {}
    for i in range(len(itemdata)):
        item = itemdata[i]
        if item in itemdict.keys():
            temp = itemdict[item]
        else:
            temp = []
        datamat = userreviewsvec[i].split()
        for index, i in enumerate(datamat):
            datamat[index] = float(i)
        temp.append(array(datamat))
        itemdict[item] = temp
        # del datamat
        # del temp
        # gc.collect()
    # print('itemdict_shape', array(itemdict[1]).shape)
    # with open('tmmp.npy','a') as f:
    #     f.writelines(str(dict[1]))
    couitem = 1
    for item in itemdict.keys():
        print('couitem', couitem)
        couitem = couitem + 1
        while len(itemdict[item]) < config['timestep_item']:
            datamat = np.zeros((1, 300)).astype(float)
            temp = itemdict[item]
            temp.extend(datamat)
            itemdict[item] = temp

    # 第三步
    df = pd.DataFrame(pd.read_csv(filename3).sort_index())
    userdata_2 = df['user']
    vecdata_1 = df['vec']
    dict_out_user = {}
    for i in range(len(userdata_2)):
        user = userdata_2[i]
        datamat = vecdata_1[i].split()
        for index, i in enumerate(datamat):
            datamat[index] = float(i)
        dict_out_user[user] = datamat
    # print('itemdict_shape', array(dict_out_user[1]).shape)
    # 第四步
    df = pd.DataFrame(pd.read_csv(filename4).sort_index())
    itemdata_2 = df['item']
    vecdata_2 = df['vec']
    dict_out_item = {}
    for i in range(len(itemdata_2)):
        item = itemdata_2[i]
        datamat = vecdata_2[i].split()
        for index, i in enumerate(datamat):
            datamat[index] = float(i)
        dict_out_item[item] = datamat
    # print('dict_out_item', array(dict_out_item[1]).shape)

    df_1 = pd.DataFrame(pd.read_csv(filename_score).sort_index())
    userdata_1 = df_1['user']
    itemdata_1 = df_1['item']
    temptemp = []
    temptemp1 = []
    temptemplist = []
    temptemplist1 = []
    print("daozhelile")
    for i in range(len(userdata_1)):
        print(i)
        user = userdata_1[i]
        item = itemdata_1[i]
        # print("np.array(user_review_dict[user]", np.array(user_review_dict))
        temptemp.append(user_review_dict[user])
        temptemp1.append(itemdict[item])
        temptemplist.append(dict_out_user[user])
        temptemplist1.append(dict_out_item[item])
    # np.savez(config['pathmiddle'] + 'user_item_review_'+fla+'.npz', user=array(temptemp), item=array(temptemp1))
    # np.savez(config['pathmiddle'] + 'user_item_out_'+fla+'.npz', user=array(temptemplist), item=array(temptemplist1))
    return temptemp, temptemp1, temptemplist, temptemplist1
'''
   原始文件：user_mutil_hot 将用户对应的mutil_hot变成矩阵
   生成字典：
   user：对应m*m的矩阵，m表示item的个数
'''
# def attention_user_data(filename):
#     df = pd.DataFrame(pd.read_csv(filename).sort_index())
#     userdata = df['user']
#     itemdata = df['item']
#     item_based_att_dic = {}
#     for index1 in range(len(userdata)):
#
#         user = userdata[index1]
#         item = itemdata[index1].split()
#         print('user_data_index1', len(userdata), index1,len(item))
#         index_item_1 = [i for i,x in enumerate(item) if x == '1.0']
#         item_based_att = np.zeros((len(item), len(item)))
#         for k in index_item_1:
#             item_based_att[k][k] = 1
#         item_based_att_dic[user] = item_based_att
#     return item_based_att_dic

def attention_user_data(filename):
    df = pd.DataFrame(pd.read_csv(filename).sort_index())
    userdata = df['user']
    itemdata = df['item']
    item_based_att_dic = {}
    for index1 in range(len(userdata)):

        user = userdata[index1]
        item = itemdata[index1].split()
        print('user_data_index1', len(userdata), index1,len(item))
        index_item_1 = [i for i,x in enumerate(item) if x == '1.0']
        item_based_att = [0] * len(item)
        # item_based_att_row = []
        # item_based_att_col = []
        # item_based_att_data = []
        for k in index_item_1:
            item_based_att[k] = 1
            # item_based_att_row.append(0)
            # item_based_att_col.append(k)
            # item_based_att_data.append(1)
        item_based_att_dic[user] = np.array(item_based_att)
        # item_based_att_dic[user] = sparse.csc_matrix((array(item_based_att_data), (array(item_based_att_row), array(item_based_att_col))), shape=(1, len(item)))
    return item_based_att_dic



def attention_item_data(filename):
    df = pd.DataFrame(pd.read_csv(filename).sort_index())
    itemdata = df['item']
    userdata = df['user']
    user_based_att_dic = {}
    for index1 in range(len(userdata)):
        print('item_data_index1',index1)
        item = itemdata[index1]
        user = userdata[index1].split()
        index_user_1 = [i for i,x in enumerate(user) if x == '1.0']
        # print('type(item)', type(index_item_1), index_item_1)
        # user_based_att = np.zeros((len(user), len(user)))
        user_based_att = [0]*len(user)
        # user_based_att_row = []
        # user_based_att_col = []
        # user_based_att_data = []
        for k in index_user_1:
            user_based_att[k]=1
            # user_based_att_row.append(0)
            # user_based_att_col.append(k)
            # user_based_att_data.append(1)
        user_based_att_dic[item] = np.array(user_based_att)
        # user_based_att_dic[item] = sparse.csc_matrix((array(user_based_att_data), (array(user_based_att_row), array(user_based_att_col))), shape=(1, len(user)))
    return user_based_att_dic
# def attention_item_data(filename):
#     df = pd.DataFrame(pd.read_csv(filename).sort_index())
#     itemdata = df['item']
#     userdata = df['user']
#     user_based_att_dic = {}
#     for index1 in range(len(userdata)):
#         print('item_data_index1',index1)
#         item = itemdata[index1]
#         user = userdata[index1].split()
#         index_user_1 = [i for i,x in enumerate(user) if x == '1.0']
#         # print('type(item)', type(index_item_1), index_item_1)
#         user_based_att = np.zeros((len(user), len(user)))
#         for k in index_user_1:
#             user_based_att[k][k] = 1
#         # print('item_based_att', array(user_based_att))
#         # print('item_based_att', type(array(user_based_att)))
#         user_based_att_dic[item] = user_based_att
#         del user
#         del item
#         del index_user_1
#         del user_based_att
#         # gc.collect()
#     return user_based_att_dic

def attention_data(user_mutil_hot_file, item_mutil_hot_file, user_item_score_file,fla):
    item_based_att_dic = attention_user_data(user_mutil_hot_file)
    user_based_att_dic = attention_item_data(item_mutil_hot_file)


    df_1 = pd.DataFrame(pd.read_csv(user_item_score_file).sort_index())
    userdata_1 = df_1['user']
    itemdata_1 = df_1['item']
    item_based = []
    user_based = []
    for i in range(len(userdata_1)):
        print('attention_data', i)
        user = userdata_1[i]
        item = itemdata_1[i]
        item_based.append(item_based_att_dic[user])
        user_based.append(user_based_att_dic[item])
    # np.savez_compressed(config['pathmiddle'] + 'user_item_attention_'+fla+'.npz', user=item_based, item=user_based)
    return item_based,user_based




'''
   loaddata
'''
# def loaddata(filename_score,filename_user_item_review, filename_user_item_out, item_based_vec, user_based_vec):
def loaddata(filename_score):
    # 评分数据准备 user——item——score
    df = pd.DataFrame(pd.read_csv(filename_score).sort_index())
    scoredata = df['score']
    user_item_score = array(scoredata)

    # df = np.load(filename_user_item_review)
    # user_rev = df['user']
    # item_rev = df['item']
    #
    #
    # df = np.load(filename_user_item_out)
    # user_out = df['user']
    # item_out = df['item']

    # df = np.load(filename_attention)
    # item_based_vec = df['user']
    # user_based_vec = df['item']

    print("loaddata!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # return user_item_score, user_rev, item_rev, user_out, item_out, item_based_vec, user_based_vec

    return user_item_score


'''
   pre_load_data
'''
def pre_load_data():
    user_rev, item_rev, user_out, item_out = predata(config['path'] + 'Fillstep_time_user_reviewSortVec', config['path'] + 'Fillstep_time_item_reviewSortVec', config['path'] + 'output_user', config['path'] + 'output_item', config['path'] + 'score/score_train','train' )
    print("train_load_1_predata!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    item_based_vec, user_based_vec = attention_data(config['path'] + 'user_mutil_hot', config['path'] + 'item_mutil_hot', config['path'] + 'score/score_train','train')
    print("train_load_2_attention_data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    user_item_score = loaddata(config['path'] + 'score/score_train')
    # user_item_score = loaddata(config['path'] + 'score/score_train', config['pathmiddle'] + 'user_item_review_train.npz', config['pathmiddle'] + 'user_item_out_train.npz', item_based_vec, user_based_vec)
    return user_item_score, user_rev, item_rev, user_out, item_out, item_based_vec, user_based_vec

'''
读数据，初始化中间向量
初始化item嵌入矩阵和user嵌入矩阵
在注意力机制部分
用户生成user的嵌入和item的嵌入
'''
# def pre_val_load_data():
#     user_rev, item_rev, user_out, item_out = predata(config['path'] + 'Fillstep_time_user_reviewSortVec', config['path'] + 'Fillstep_time_item_reviewSortVec', config['path'] + 'output_user', config['path'] + 'output_item', config['path'] + 'score/score_val','val' )
#     print("test_load_1_predata!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     item_based_vec, user_based_vec = attention_data(config['path'] + 'user_mutil_hot', config['path'] + 'item_mutil_hot', config['path'] + 'score/score_val','val')
#     print("test_load_2_attention_data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     # return loaddata(config['path'] + 'score/score_test', config['pathmiddle'] + 'user_item_review_val.npz', config['pathmiddle'] + 'user_item_out_val.npz', item_based_vec, user_based_vec)
#     user_item_score = loaddata(config['path'] + 'score/score_val')
#     return user_item_score, user_rev, item_rev, user_out, item_out, item_based_vec, user_based_vec

def pre_test_load_data():
    user_rev, item_rev, user_out, item_out = predata(config['path'] + 'Fillstep_time_user_reviewSortVec', config['path'] + 'Fillstep_time_item_reviewSortVec', config['path'] + 'output_user', config['path'] + 'output_item', config['path'] + 'score/score_test','test' )
    print("test_load_1_predata!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    item_based_vec, user_based_vec = attention_data(config['path'] + 'user_mutil_hot', config['path'] + 'item_mutil_hot', config['path'] + 'score/score_test','test')
    print("test_load_2_attention_data!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # return loaddata(config['path'] + 'score/score_test', config['pathmiddle'] + 'user_item_review_val.npz', config['pathmiddle'] + 'user_item_out_val.npz', item_based_vec, user_based_vec)
    user_item_score = loaddata(config['path'] + 'score/score_test')
    return user_item_score, user_rev, item_rev, user_out, item_out, item_based_vec, user_based_vec