import random

import pandas as pd
import torch
import numpy as np
import time
import os
from myargs import get_args
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_random_user_rank_list(model, genres, image_feature, k):
    user_number = model.user_embedding.shape[0]
    user_list = list(range(0, user_number))
    res_list = []
    for i in range(k):
        res_list.append(random.sample(user_list, 1)[0])
    return res_list


def get_similar_user_speed(model, genres, image_feature, k):
    genres = genres.unsqueeze(dim=0)
    img_feature = image_feature.unsqueeze(dim=0)
    q_v_c = model(genres, img_feature, 1)
    user_emb = model.user_embedding
    ratings = torch.mul(user_emb, q_v_c).sum(dim=1)
    index = torch.argsort(-ratings)
    return index[0:k].cpu().detach().numpy().tolist()


def hr_at_k(item, recommend_users, item_user_dict, k):
    groundtruth_user = item_user_dict.get(item)
    recommend_users = recommend_users[0:k]
    inter = set(groundtruth_user).intersection(set(recommend_users))
    return len(inter)


def dcg_k(r):
    r = np.asarray(r)
    val = np.sum((np.power(2, r) - 1) / (np.log2(np.arange(1+1, r.size + 2))))
    return val


def ndcg_k(item, recommend_users, item_user_dict, k):
    groundtruth_user = item_user_dict.get(item)
    recommend_users = recommend_users[0:k]
    ratings = []
    ndcg = 0.0
    for u in recommend_users:
        if u in groundtruth_user:
            ratings.append(1.0)
        else:
            ratings.append(0.0)
    ratings_ideal = sorted(ratings, reverse=True)
    ideal_dcg = dcg_k(ratings_ideal)
    if ideal_dcg != 0:
        ndcg = (dcg_k(ratings) / ideal_dcg)
    return ndcg


class Validate:
    def __init__(self, validate_csv, user_serialize_dict, img, genres, category_num):
        print("validate class init")
        validate_csv = pd.read_csv(validate_csv)
        self.item = set(validate_csv['asin'])
        self.item_user_dict = {}
        # 构建完成 item->user dict
        for it in self.item:
            users = validate_csv[validate_csv['asin'] == it]['reviewerID']
            users = [user_serialize_dict.get(u) for u in users]
            self.item_user_dict[it] = users
        self.img_dict = img
        self.genres_dict = genres
        self.category_num = category_num

    def start_validate(self, model):
        # 开始评估
        hr_hit_cnt_5, hr_hit_cnt_10, hr_hit_cnt_20 = 0, 0, 0
        ndcg_sum_5, ndcg_sum_10, ndcg_sum_20 = 0.0, 0.0, 0.0
        max_k = 20
        it_idx = 0
        for it in self.item:
            # 输出
            model = model.to(device)  # move to cpu
            # 处理 item genres
            genres = torch.full((self.category_num, 1), -1)
            genres_index = self.genres_dict.get(it)
            genres[genres_index] = 1
            genres = genres.squeeze(dim=1)
            genres = torch.tensor(genres)
            image_feature = self.img_dict.get(it)
            image_feature = torch.tensor(image_feature)
            genres = genres.to(device)
            image_feature = image_feature.to(device)
            with torch.no_grad():
                recommend_users = get_similar_user_speed(model, genres, image_feature, max_k)
            # 计算hr指标
            hr_hit_cnt_5 += hr_at_k(it, recommend_users, self.item_user_dict, 5)
            hr_hit_cnt_10 += hr_at_k(it, recommend_users, self.item_user_dict, 10)
            hr_hit_cnt_20 += hr_at_k(it, recommend_users, self.item_user_dict, 20)
            # 计算NDCG指标
            ndcg_sum_5 += ndcg_k(it, recommend_users, self.item_user_dict, 5)
            ndcg_sum_10 += ndcg_k(it, recommend_users, self.item_user_dict, 10)
            ndcg_sum_20 += ndcg_k(it, recommend_users, self.item_user_dict, 20)
            # print("评估进度:", it_idx, "/", len(item))
            it_idx += 1
        item_len = len(self.item)
        hr_5 = hr_hit_cnt_5 / (item_len * 5)
        hr_10 = hr_hit_cnt_10 / (item_len * 10)
        hr_20 = hr_hit_cnt_20 / (item_len * 20)
        ndcg_5 = ndcg_sum_5/item_len
        ndcg_10 = ndcg_sum_10/item_len
        ndcg_20 = ndcg_sum_20/item_len
        print("hr@5:", "hr_10:", "hr_20:", 'ndcg@5', 'ndcg@10', 'ndcg@20')
        print(hr_5, ',', hr_10, ',', hr_20, ',', ndcg_5, ',', ndcg_10, ',', ndcg_20)
        return hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20


if __name__ == '__main__':
    # 参数解析器
    # 参数解析器
    import pickle
    from support import RatingDataset
    from model import CCFCRec
    args = get_args()
    # 提取user的原id: 序列化id的dict
    train_path = "data/train_withneg_rating.csv"
    vliad_path = 'data/test_rating.csv'
    train_df = pd.read_csv(train_path)
    load_dir = 'result/2022-10-14/'
    pkl_file = open(load_dir+'save_dict.pkl', 'rb')
    data = pickle.load(pkl_file)
    dataSet = RatingDataset(train_df, data['img_feature_dict'], data['asin_category_int_map'], data['category_ser_map_len'],
                            data['user_ser_dict'], args.positive_number, args.negative_number)
    args.user_number = dataSet.user_number
    args.item_number = dataSet.item_number
    validator = Validate(validate_csv=vliad_path, user_serialize_dict=data['user_ser_dict'], img=data['img_feature_dict'],
                         genres=data['asin_category_int_map'], category_num=data['category_ser_map_len'])
    myModel = CCFCRec(args)
    print('---------数据集加载完毕，开始测试----------------')
    test_result_name = 'test_result.csv'
    with open(test_result_name, 'a+') as f:
        f.write("p@5,p@10,p@20,ndcg@5,ndcg@10,ndcg@20\n")
    load_array = ['98', '99', '100']
    for model in load_array:
        myModel.load_state_dict(torch.load(load_dir+'/'+model+'.pt'))
        hr5, hr_10, hr_20, n_5, n_10, n_20 = validator.start_validate(myModel)
        with open(test_result_name, 'a+') as f:
            f.write("{},{},{},{},{},{}\n".format(p5, p_10, p_20, n_5, n_10, n_20))