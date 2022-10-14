import pandas as pd
import torch
import numpy as np
import time
import os
from myargs import get_args
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_similar_user_speed(model, genres, img, k):
    user_embedding = model.user_embedding
    user_idx = torch.tensor(list(range(user_embedding.shape[0])))
    user_idx = user_idx.to(device)
    # [138493*64]
    user_emb = user_embedding[user_idx]
    genres = genres.unsqueeze(dim=0)
    img = img.unsqueeze(dim=0)
    # 输入到模型中
    attr_present = model.attr_matrix(genres)
    attr_tmp1 = model.h(torch.matmul(attr_present, model.attr_W1.T) + model.attr_b1)
    attr_attention_b = model.softmax(torch.matmul(attr_tmp1, model.attr_W2))
    z_v = torch.matmul(attr_attention_b.transpose(1, 2), attr_present).squeeze()  # z_v是属性经过注意力加权融合后的向量
    p_v = torch.matmul(img, model.image_projection)  # item的图像嵌入向量
    q_v_a = torch.cat((z_v.unsqueeze(dim=0), p_v), dim=1)
    q_v_c = model.gen_layer2(model.h(model.gen_layer1(q_v_a)))
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
    def __init__(self, validate_csv, user_serialize_dict, img, genres):
        print("validate class init")
        validate_csv = pd.read_csv(validate_csv, dtype={'userId': int, 'movieId': int, 'rating': float})
        self.item = set(validate_csv['movieId'])
        self.item_user_dict = {}
        # 构建完成 item->user dict
        for it in self.item:
            users = validate_csv[validate_csv['movieId'] == it]['userId']
            users = [user_serialize_dict.get(u) for u in users]
            self.item_user_dict[it] = users
        self.img_dict = img
        self.genres_dict = genres

    def start_validate(self, model):
        # 开始评估
        hr_hit_cnt_5, hr_hit_cnt_10, hr_hit_cnt_20 = 0, 0, 0
        ndcg_sum_5, ndcg_sum_10, ndcg_sum_20 = 0.0, 0.0, 0.0
        max_k = 20
        it_idx = 0
        for it in self.item:
            # 输出
            model = model.to(device)  # move to cpu
            genres = torch.tensor(self.genres_dict.get(it))
            img_feature = torch.tensor(self.img_dict.get(it))
            genres = genres.to(device)
            img_feature = img_feature.to(device)
            with torch.no_grad():
                recommend_users = get_similar_user_speed(model, genres, img_feature, max_k)
            # 计算hr指标
            # 计算p@k指标
            hr_hit_cnt_5 += p_at_k(it, recommend_users, self.item_user_dict, 5)
            hr_hit_cnt_10 += p_at_k(it, recommend_users, self.item_user_dict, 10)
            hr_hit_cnt_20 += p_at_k(it, recommend_users, self.item_user_dict, 20)
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
    from support import read_img_feature, read_genres, serialize_user
    from model import CCFCRec
    args = get_args()
    train_df = pd.read_csv("data/train_rating.csv", dtype={'userId': int, 'movieId': int, 'neg_user_id': int})
    total_user_set = train_df['userId']
    user_serialize_dict = serialize_user(total_user_set)
    img_feature = read_img_feature('data/img_feature.csv')
    movie_onehot = read_genres('data/movies_onehot.csv')
    myModel = CCFCRec(args)
    validator = Validate(validate_csv='data/test_rating.csv', user_serialize_dict=user_serialize_dict,
                         img=img_feature, genres=movie_onehot)
    print('---------数据集加载完毕，开始测试----------------')
    test_result_name = 'test_result.csv'
    with open(test_result_name, 'a+') as f:
        f.write("hr@5,hr@10,hr@20,ndcg@5,ndcg@10,ndcg@20\n")
    load_dir = 'result/2022_10_14/'
    load_array = ['0batch_3000', '0batch_6000']
    for model in load_array:
        myModel.load_state_dict(torch.load(load_dir+'/epoch_'+model+'.pt'))
        p5, p_10, p_20, n_5, n_10, n_20 = validator.start_validate(myModel)
        with open(test_result_name, 'a+') as f:
            f.write("{},{},{},{},{},{}\n".format(p5, p_10, p_20, n_5, n_10, n_20))