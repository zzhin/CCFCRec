import gc
import pickle
import random
import os
import sys
import time
import numpy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from tqdm import tqdm


# 返回一个dict， userId: [positive_sample(list), negative_sample(list(list))]
def read_user_positive_negative_movies(user_positive_movie_csv, refresh=False):
    pkl_name = 'pkl/user_pn_dict.pkl'
    if (os.path.exists(pkl_name) is True) and (refresh is False):
        pkl_file = open(pkl_name, 'rb')
        data = pickle.load(pkl_file)
        return data['user_pn_dict']
    user_position_dict = {}
    last_user = -1
    user_position_dict[last_user] = [-1, -1]
    for index, row in tqdm(user_positive_movie_csv.iterrows()):
        u = row['userId']
        if u != last_user:
            user_position_dict[u] = [index, index]
            user_position_dict[last_user] = [user_position_dict.get(last_user)[0], index-1]
            last_user = u
    # 更新最后一项
    user_position_dict[last_user] = [user_position_dict.get(last_user)[0], user_positive_movie_csv.__len__()-1]
    with open(pkl_name, 'wb') as file:
        pickle.dump({'user_pn_dict': user_position_dict}, file)
    return user_position_dict


def read_img_feature(img_feature_csv):
    df = pd.read_csv(img_feature_csv, dtype={'feature': object, 'movie_id': int})
    img_feature_dict = {}
    for index, row in df.iterrows():
        item = row['movie_id']
        feature = list(map(float, row['feature'][1:-1].split(",")))
        img_feature_dict[item] = feature
    return img_feature_dict


def read_genres(genres_csv):
    df = pd.read_csv(genres_csv, dtype={'movieId': int})
    genres_dict = {}
    for index, row in df.iterrows():
        item = row['movieId']
        genres = list(map(int, row['genres_onehot'][1:-1].split(',')))
        genres_dict[item] = genres
    return genres_dict


def serialize_user(user_set):
    user_set = set(user_set)
    user_idx = 0
    # key: user原始下标，value: user有序下标
    user_serialize_dict = {}
    for user in user_set:
        user_serialize_dict[user] = user_idx
        user_idx += 1
    return user_serialize_dict


# 输入user和item的set，输出user和item从1到n有序的字典
def serialize_item(item_set):
    item_set = set(item_set)
    item_idx = 0
    item_serialize_dict = {}
    for item in item_set:
        item_serialize_dict[item] = item_idx
        item_idx += 1
    return item_serialize_dict


class RatingDataset(torch.utils.data.Dataset):
    def __init__(self, train_csv, user_positive_movie_csv, img_features, genres, user_serialize_dict, item_pn_csv,
                 positive_number, negative_number):
        self.train_csv = train_csv
        # 读其他内容
        self.img_feature_dict = img_features
        self.genres_dict = genres
        self.user_pos_neg_movie_df = pd.read_csv(user_positive_movie_csv, dtype={'userId': np.int32, 'positive_movies': np.int32})
        self.item_pn_df = pd.read_csv(item_pn_csv)
        # print(self.item_pn_df)
        self.user_position_dict = read_user_positive_negative_movies(self.user_pos_neg_movie_df)
        self.user = self.train_csv["userId"]
        self.neg_user = self.train_csv['neg_user_id']
        self.item = self.train_csv["movieId"]
        self.rating = self.train_csv["rating"]
        # 序列化user和item
        self.user_serialize_dict = user_serialize_dict
        self.item_serialize_dict = serialize_item(self.item)
        # 返回个数时，返回全集的user数和训练集的item数
        self.user_number = len(user_serialize_dict)
        self.item_number = len(set(self.item))
        self.positive_number = positive_number
        self.negative_number = negative_number
        print("整个数据集的user个数为:", self.user_number, "train_set中的用户数目为:", len(set(self.user)))

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, index):
        user = self.user[index]
        item = self.item[index]
        neg_user = self.neg_user[index]
        # 处理 item genres
        genres = self.genres_dict.get(item)
        # 处理 item feature
        img_feature = self.img_feature_dict.get(item)
        # 处理 positive items
        # 直接存储df的位置，user -> 从哪里到哪里
        position_arr = self.user_position_dict.get(user)
        positive_segment = self.user_pos_neg_movie_df.loc[position_arr[0]: position_arr[1]]
        positive_movie_df = pd.DataFrame.sample(positive_segment, n=self.positive_number, replace=True)
        positive_movie_list = list(positive_movie_df['positive_movies'])
        negative_movie_list = positive_movie_df['negative_movies']
        self_negative_list = self.item_pn_df['negative_movies'][index]
        # self neg list 完成 序列化
        tmp_neg_list = list(map(int, self_negative_list[1:-1].split(",")))
        tmp_neg_ser_list = [self.item_serialize_dict.get(item) for item in tmp_neg_list]
        # 插入一条抽样
        self_neg_list = list(np.random.choice(tmp_neg_ser_list, self.negative_number, replace=True))
        # coll neg完成序列化
        neg_list = []
        for neg in negative_movie_list:
            tmp_neg_list = list(map(int, neg[1:-1].split(",")))
            tmp_neg_ser_list = [self.item_serialize_dict.get(item) for item in tmp_neg_list]
            # 插入一条抽样
            neg_list.append(list(np.random.choice(tmp_neg_ser_list, self.negative_number, replace=True)))
        # 对当前item进行抽样
        # user，item id进行序列化
        user = self.user_serialize_dict.get(user)
        neg_user = self.user_serialize_dict.get(neg_user)
        item = self.item_serialize_dict.get(item)
        # 序列化positive_movie_list
        positive_movie_list = [self.item_serialize_dict.get(item) for item in positive_movie_list]
        return torch.tensor(user), torch.tensor(item), torch.tensor(genres), torch.tensor(img_feature), \
               torch.tensor(neg_user), torch.tensor(positive_movie_list), torch.tensor(neg_list), \
               torch.tensor(self_neg_list)
