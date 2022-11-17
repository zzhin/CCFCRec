import math
import os
import sys
import pickle
import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from preprocess import serial_asin_category
from extract_img_feature import get_img_feature_pickle
from support import RatingDataset
from tqdm import tqdm
import pandas as pd
import time
from support import serialize_user
from test import Validate
from myargs import get_args, args_tostring


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#torch.cuda.set_device(1)


# CCFCRec
class CCFCRec(nn.Module):
    def __init__(self, args):
        super(CCFCRec, self).__init__()
        self.args = args
        self.attr_matrix = torch.nn.Parameter(torch.FloatTensor(args.attr_num, args.attr_present_dim))
        # 定义属性attribute注意力层
        self.attr_W1 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, args.attr_present_dim))
        self.attr_b1 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, 1))
        self.attr_W2 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, 1))
        # 控制整个模型的激活函数
        self.h = nn.LeakyReLU()
        # 图像的映射矩阵
        self.image_projection = torch.nn.Parameter(torch.FloatTensor(4096, args.implicit_dim))
        self.sigmoid = torch.nn.Sigmoid()  # 将门控信号映射到[0, 1]之间
        # user和item的嵌入层，可用预训练的进行初始化
        if args.pretrain is True:
            if args.pretrain_update is True:
                self.user_embedding = nn.Parameter(torch.load('user_emb.pt'), requires_grad=True)
                self.item_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=True)
            else:
                self.user_embedding = nn.Parameter(torch.load('user_emb.pt'), requires_grad=False)
                self.item_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=False)
        else:
            self.user_embedding = nn.Parameter(torch.FloatTensor(args.user_number, args.implicit_dim))
            self.item_embedding = nn.Parameter(torch.FloatTensor(args.item_number, args.implicit_dim))
        # 定义生成层，将(q_v_a, u)的信息，共同生成 q_v_c， 生成包含协同信息的item嵌入
        self.gen_layer1 = nn.Linear(args.attr_present_dim*2, args.cat_implicit_dim)
        self.gen_layer2 = nn.Linear(args.attr_present_dim, args.attr_present_dim)
        # 参数初始化
        self.__init_param__()

    def __init_param__(self):
        nn.init.xavier_normal_(self.attr_matrix)
        nn.init.xavier_normal_(self.attr_W1)
        nn.init.xavier_normal_(self.attr_W2)
        nn.init.xavier_normal_(self.attr_b1)
        nn.init.xavier_normal_(self.image_projection)
        # 生成层初始化
        # user, item嵌入层的初始化, 没有预训练的情况下就初始化
        if self.args.pretrain is False:
            nn.init.xavier_normal_(self.user_embedding)
            nn.init.xavier_normal_(self.item_embedding)
        nn.init.xavier_normal_(self.gen_layer1.weight)
        nn.init.xavier_normal_(self.gen_layer2.weight)

    def forward(self, attribute, image_feature, batch_size):
        z_v = torch.matmul(torch.matmul(self.attr_matrix, self.attr_W1)+self.attr_b1.squeeze(), self.attr_W2)
        z_v_copy = z_v.repeat(batch_size, 1, 1)
        z_v_squeeze = z_v_copy.squeeze(dim=2).to(device)
        neg_inf = torch.full(z_v_squeeze.shape, -1e6).to(device)
        z_v_mask = torch.where(attribute != -1, z_v_squeeze, neg_inf)
        attr_attention_weight = torch.softmax(z_v_mask, dim=1)
        final_attr_emb = torch.matmul(attr_attention_weight, self.attr_matrix)
        image_norm = torch.nn.functional.normalize(image_feature, dim=1)
        p_v = torch.matmul(image_feature, self.image_projection)  # item的图像嵌入向量
        q_v_a = torch.cat((final_attr_emb, p_v), dim=1)
        q_v_c = self.gen_layer2(self.h(self.gen_layer1(q_v_a)))
        return q_v_c


def train(model, train_loader, optimizer, valida, args, model_save_dir):
    print("model start train!")
    test_save_path = model_save_dir + "/result.csv"
    print("model train at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # 写入超参数
    with open(model_save_dir + "/readme.txt", 'a+') as f:
        str_ = args_tostring(args)
        f.write(str_)
        f.write('\nsave dir:'+model_save_dir)
        f.write('\nmodel train time:'+(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    with open(test_save_path, 'a+') as f:
        f.write("loss,contrast_sum,hr@5,hr@10,hr@20,ndcg@5,ndcg@10,ndcg@20\n")
    save_index = 0
    for i_epoch in range(args.epoch):
        i_batch = 0
        batch_time = time.time()
        for user, item, item_genres, item_img_feature, neg_user, positive_item_list, negative_item_list, self_neg_list in tqdm(train_loader):
            optimizer.zero_grad()
            model.train()
            # allocate memory cpu to gpu
            model = model.to(device)
            user = user.to(device)
            item = item.to(device)
            item_genres = item_genres.to(device)
            item_img_feature = item_img_feature.to(device)
            neg_user = neg_user.to(device)
            positive_item_list = positive_item_list.to(device)
            negative_item_list = negative_item_list.to(device)
            # run model
            q_v_c = model(item_genres, item_img_feature, user.shape[0])
            q_v_c_unsqueeze = q_v_c.unsqueeze(dim=1)
            # compute contrast loss
            positive_item_emb = model.item_embedding[positive_item_list]
            pos_contrast_mul = torch.sum(torch.mul(q_v_c_unsqueeze, positive_item_emb), dim=2) / (
                    args.tau * torch.norm(q_v_c_unsqueeze, dim=2) * torch.norm(positive_item_emb, dim=2))
            pos_contrast_exp = torch.exp(pos_contrast_mul)  # shape = 1024*10
            # negative samples
            neg_item_emb = model.item_embedding[negative_item_list]
            q_v_c_un2squeeze = q_v_c_unsqueeze.unsqueeze(dim=1)
            neg_contrast_mul = torch.sum(torch.mul(q_v_c_un2squeeze, neg_item_emb), dim=3) / (
                    args.tau * torch.norm(q_v_c_un2squeeze, dim=3) * torch.norm(neg_item_emb, dim=3))
            neg_contrast_exp = torch.exp(neg_contrast_mul)
            neg_contrast_sum = torch.sum(neg_contrast_exp, dim=2)  # shape = [1024, 10]
            contrast_val = -torch.log(pos_contrast_exp / (pos_contrast_exp + neg_contrast_sum))  # shape = [1024*10]
            contrast_examples_num = contrast_val.shape[0] * contrast_val.shape[1]
            contrast_sum = torch.sum(torch.sum(contrast_val, dim=1), dim=0) / contrast_val.shape[1]  # 同一个batch求mean
            '''
            contrast self
            '''
            self_neg_item_emb = model.item_embedding[self_neg_list]
            self_neg_contrast_mul = torch.sum(torch.mul(q_v_c_unsqueeze, self_neg_item_emb), dim=2)/(
                args.tau*torch.norm(q_v_c_unsqueeze, dim=2)*torch.norm(self_neg_item_emb, dim=2))
            self_neg_contrast_sum = torch.sum(torch.exp(self_neg_contrast_mul), dim=1)
            item_emb = model.item_embedding[item]
            self_pos_contrast_mul = torch.sum(torch.mul(q_v_c, item_emb), dim=1) / (
                    args.tau * torch.norm(q_v_c, dim=1) * torch.norm(item_emb, dim=1))
            self_pos_contrast_exp = torch.exp(self_pos_contrast_mul)  # shape = 1024*1
            self_contrast_val = -torch.log(self_pos_contrast_exp/(self_pos_contrast_exp+self_neg_contrast_sum))
            self_contrast_sum = torch.sum(self_contrast_val)
            # rank loss
            user_emb = model.user_embedding[user]
            item_emb = model.item_embedding[item]
            neg_user_emb = model.user_embedding[neg_user]
            logsigmoid = torch.nn.LogSigmoid()
            y_uv = torch.mul(item_emb, user_emb).sum(dim=1)
            y_kv = torch.mul(item_emb, neg_user_emb).sum(dim=1)
            y_ukv = -logsigmoid(y_uv - y_kv).sum()
            # 使用属性生成item嵌入，再做一个bpr排序
            y_uv2 = torch.mul(q_v_c, user_emb).sum(dim=1)
            y_kv2 = torch.mul(q_v_c, neg_user_emb).sum(dim=1)
            y_ukv2 = -logsigmoid(y_uv2 - y_kv2).sum()
            total_loss = args.lambda1*(contrast_sum+self_contrast_sum) + (1-args.lambda1)*(y_ukv+y_ukv2)
            if math.isnan(total_loss):
                print("loss is nan!, exit.", total_loss)
                exit(255)
            total_loss.backward()
            optimizer.step()
            i_batch += 1
            if i_batch % args.save_batch_time == 0:
                model.eval()
                print("[{},/13931603]total_loss:,{},{},s".format(i_batch*1024, total_loss.item(), int(time.time()-batch_time)))
                with torch.no_grad():
                    hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20 = valida.start_validate(model)
                with open(test_save_path, 'a+') as f:
                    f.write("{},{},{},{},{},{},{},{}\n".format(total_loss.item(), contrast_sum, hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20))
                # 保存模型
                batch_time = time.time()
                save_index += 1
                torch.save(model.state_dict(), model_save_dir + '/' + str(save_index)+".pt")


if __name__ == '__main__':
    # result save dir
    save_dir = 'result/' + time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))
    os.makedirs(save_dir)
    # args
    args = get_args()
    print("progress start at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    train_path = "data/train_withneg_rating.csv"
    vliad_path = 'data/validate_rating.csv'
    train_df = pd.read_csv(train_path)
    total_user_set = train_df['reviewerID']
    user_ser_dict = serialize_user(total_user_set)
    asin_category_int_map, category_ser_map = serial_asin_category()
    img_feature_dict = get_img_feature_pickle()
    # write internal variable
    with open(save_dir+"/save_dict.pkl", "wb") as file:
        save_dict = {'img_feature_dict': img_feature_dict, 'asin_category_int_map': asin_category_int_map,
                     'category_ser_map_len': category_ser_map.__len__(), 'user_ser_dict': user_ser_dict}
        pickle.dump(save_dict, file)
    # load dataset
    dataSet = RatingDataset(train_df, img_feature_dict, asin_category_int_map, category_ser_map.__len__(),
                            user_ser_dict, args.positive_number, args.negative_number)
    args.user_number = dataSet.user_number
    args.item_number = dataSet.item_number
    train_loader = torch.utils.data.DataLoader(dataSet, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print("模型超参数:", args_tostring(args))
    myModel = CCFCRec(args)
    optimizer = torch.optim.Adam(myModel.parameters(), lr=args.learning_rate, weight_decay=0.1)
    validator = Validate(validate_csv=vliad_path, user_serialize_dict=user_ser_dict, img=img_feature_dict,
                         genres=asin_category_int_map, category_num=category_ser_map.__len__())
    train(myModel, train_loader, optimizer, validator, args, save_dir)

