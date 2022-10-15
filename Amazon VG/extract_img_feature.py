import numpy as np

def get_img_feature_pickle(img_feature_path='data/img_feature.npy'):
    # key: item的asin, value是item的图像特征
    return np.load(img_feature_path, allow_pickle=True).item()