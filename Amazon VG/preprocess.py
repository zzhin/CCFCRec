import pickle

def serial_asin_category(pkl_name='data/asin_int_category.pkl'):
    if os.path.exists(pkl_name) is True:
        pkl_file = open(pkl_name, "rb")
        data = pickle.load(pkl_file)
        # data['asin_category_int_map']， asin: category, category为经过顺序化后的属性
        # data['category_ser_map']， category: category_int_num, category对应的顺序编号
        return data['asin_category_int_map'], data['category_ser_map']
    asin_df = pd.read_csv("../data/asin.csv")
    category_set = set([])
    for idx, row in asin_df.iterrows():
        cat = row['category'].split(',')
        for i in cat:
            category_set.add(i)
    # 用顺序给category编上序号，把asin中的category字符串转换为数字
    idx = 0
    category_ser_map = {}
    for it in category_set:
        category_ser_map[it] = idx
        idx += 1
    asin_category_int_map = {}
    for idx, row in asin_df.iterrows():
        cat = row['category'].split(',')
        asin = row['asin']
        tmp_list = []
        for i in cat:
            tmp_list.append(category_ser_map.get(i))
        asin_category_int_map[asin] = tmp_list
    with open(pkl_name, "wb") as file:
        pickle.dump({'asin_category_int_map': asin_category_int_map, 'category_ser_map': category_ser_map}, file)