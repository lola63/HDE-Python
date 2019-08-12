#coding:utf-8

import numpy as np
from numpy import array
import pandas as pd
import csv
import config_
global config
config = config_.configs

'''
lstm 编码，全连接层解码

'''


def predata(filename1, filename2, filename_score):
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
    # print('user_review_dict:', type(user_review_dict[1][0]))
    for user in user_review_dict.keys():
        while len(user_review_dict[user]) < config['timestep_user']:
            datamat = np.zeros((1, 5)).astype(float)
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
    for i in range(len(userdata)):
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
    # print('itemdict_shape', array(itemdict[1]).shape)
    # with open('tmmp.npy','a') as f:
    #     f.writelines(str(dict[1]))
    for item in itemdict.keys():
        while len(itemdict[item]) < config['timestep_item']:
            datamat = np.zeros((1, 5)).astype(float)
            temp = itemdict[item]
            temp.extend(datamat)
            itemdict[item] = temp
    #     print(array(itemdict[item]))
    # print('item_review_dict', array(itemdict))


    df_1 = pd.DataFrame(pd.read_csv(filename_score).sort_index())
    userdata_1 = df_1['user']
    itemdata_1 = df_1['item']
    # scoredata_1 = df_1['score']
    with open(config['path']+'user_item_review.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["user", "item"])
    with open(config['path']+'user_item_out.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["user", "item"])
    temptemp = []
    temptemp1 = []
    for i in range(len(userdata_1)):
        user = userdata_1[i]
        item = itemdata_1[i]
        # print("np.array(user_review_dict[user]", np.array(user_review_dict))
        temptemp.append(np.array(user_review_dict[user]))
        temptemp1.append(np.array(itemdict[item]))
    np.savez(config['path'] + 'user_item_review.npz', user=array(temptemp), item=array(temptemp1))

'''
   原始文件：user_mutil_hot 将用户对应的mutil_hot变成矩阵
   生成字典：
   user：对应m*m的矩阵，m表示item的个数
'''
def attention_user_data(filename):
    df = pd.DataFrame(pd.read_csv(filename).sort_index())
    userdata = df['user']
    itemdata = df['item']
    item_based_att_dic = {}
    for index1 in range(len(userdata)):
        user = userdata[index1]
        item = itemdata[index1].split()
        index_item_1 = [i for i,x in enumerate(item) if x == '1.0']
        # print('type(item)', type(index_item_1), index_item_1)
        item_based_att = np.zeros((len(item), len(item)))
        for index2, k in enumerate(index_item_1):
            item_based_att[k][k] = 1
        # print('item_based_att', array(item_based_att))
        # print('item_based_att', type(array(item_based_att)))
        item_based_att_dic[user] = array(item_based_att)
    return item_based_att_dic


def attention_item_data(filename):
    df = pd.DataFrame(pd.read_csv(filename).sort_index())
    itemdata = df['item']
    userdata = df['user']
    user_based_att_dic = {}
    for index1 in range(len(userdata)):
        item = itemdata[index1]
        user = userdata[index1].split()
        index_user_1 = [i for i,x in enumerate(user) if x == '1.0']
        # print('type(item)', type(index_item_1), index_item_1)
        user_based_att = np.zeros((len(user), len(user)))
        for index2, k in enumerate(index_user_1):
            user_based_att[k][k] = 1
        # print('item_based_att', array(user_based_att))
        # print('item_based_att', type(array(user_based_att)))
        user_based_att_dic[item] = array(user_based_att)
    return user_based_att_dic


def attention_data(user_mutil_hot_file, item_mutil_hot_file, user_item_score_file):
    item_based_att_dic = attention_user_data(user_mutil_hot_file)
    user_based_att_dic = attention_item_data(item_mutil_hot_file)

    df_1 = pd.DataFrame(pd.read_csv(user_item_score_file).sort_index())
    userdata_1 = df_1['user']
    itemdata_1 = df_1['item']
    item_based = []
    user_based = []
    for i in range(len(userdata_1)):
        user = userdata_1[i]
        item = itemdata_1[i]
        item_based.append(array(item_based_att_dic[user]))
        user_based.append(array(user_based_att_dic[item]))
    np.savez(config['path'] + 'user_item_attention.npz', user=array(item_based), item=array(user_based))


'''
   loaddata
'''
def loaddata(filename_score,filename_user_item_review, filename_attention):
    # 评分数据准备 user——item——score
    df = pd.DataFrame(pd.read_csv(filename_score).sort_index())
    scoredata = df['score']
    user_item_score = array(scoredata)

    df = np.load(filename_user_item_review)
    user_rev = df['user']
    item_rev = df['item']

    df = np.load(filename_attention)
    item_based_vec = df['user']
    user_based_vec = df['item']

    return user_item_score, user_rev, item_rev, item_based_vec, user_based_vec


'''
   pre_load_data
'''
def pre_load_data():
    predata(config['path'] + 'inputtext', config['path'] + 'input_item', config['path'] + 'score', )
    attention_data(config['path'] + 'user_mutil_hot', config['path'] + 'item_mutil_hot', config['path'] + 'score')
    return loaddata(config['path'] + 'score', config['path'] + 'user_item_review.npz', config['path'] + 'user_item_attention.npz')
'''
读数据，初始化中间向量
初始化item嵌入矩阵和user嵌入矩阵
在注意力机制部分
用户生成user的嵌入和item的嵌入
'''