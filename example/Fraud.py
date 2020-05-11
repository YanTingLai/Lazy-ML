import pandas as pd
from collections import Counter
from LazyML.DataLoader import GetData
from LazyML.LazyModel import LazyClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def clean_data(train, test):
    """
    資料處理流程
    :param train: 測試集
    :param test: 驗證集
    :return: x_train, x_valid, y_train, y_valid, x_test, cols, user_id
    """
    y_train = train['class'].copy()
    train.drop(['class'], 1, inplace=True)

    user_id = test['user_id'].copy()

    # 將資料集合併做資料處理
    train['training'] = 1
    test['training'] = 0
    all_df = pd.concat([train, test])

    # 處理城市變數
    all_df.loc[pd.isna(all_df.country), 'country'] = 'unknown'
    # 小城市合併成其他
    country_list = [k for k, v in Counter(train.country.tolist()).items() if v < 5]
    all_df.loc[all_df.country.isin(country_list), 'country'] = 'others'

    # 處理device_id
    device_id = all_df.device_id.tolist()
    dup = []
    for k, v in Counter(device_id).items():
        if v > 1:
            dup += [k]
    device_id_dup = [id_ in dup for id_ in device_id]
    all_df['dup_device'] = device_id_dup

    # 產生衍生性變數:(purchase_time - signup_time) 註冊日與交易日天數差異
    all_df['diff_days'] = (all_df.purchase_time - all_df.signup_time).dt.seconds

    # 新增月份、小時、每周第幾天與交易天
    all_df['signup_month'] = all_df.signup_time.dt.month
    all_df['signup_hour'] = all_df.signup_time.dt.hour
    all_df['signup_dayofweek'] = all_df.signup_time.dt.dayofweek
    all_df['purchase_day'] = all_df.purchase_time.dt.day
    all_df['purchase_month'] = all_df.purchase_time.dt.month
    all_df['purchase_hour'] = all_df.purchase_time.dt.hour
    all_df['dayofweek'] = all_df.purchase_time.dt.dayofweek

    # 產生device_id單天消費次數
    # 有發現詐騙標註常發生在單日內
    # 因此算出device_id單日有多少交易量
    # 但此變數加入模型後表現變非常差
    # 還需再觀察

    # tt = all_df[['device_id', 'purchase_time', 'purchase_month', 'purchase_day']].copy()
    # aa = tt.groupby(['device_id', 'purchase_month', 'purchase_day'], as_index=False).count()
    # aa.columns = ['device_id', 'purchase_month', 'purchase_day', 'dup_purchase']
    # aa['dup_purchase'] = [0 if x_ == 1 else 1 for x_ in aa['dup_purchase'].tolist()]
    # # print(tt[tt['device_id'] == 'b00884bec1cbe'])
    # # print(aa[aa['device_id'] == 'b00884bec1cbe'])
    # all_df = pd.merge(left=all_df,
    #                   right=aa,
    #                   left_on=['device_id', 'purchase_month', 'purchase_day'],
    #                   right_on=['device_id', 'purchase_month', 'purchase_day']
    #                   )
    # all_df['dup_purchase_2'] = all_df['dup_purchase'] * all_df['dup_device']

    # 刪掉多餘變數
    all_df.drop(['device_id', 'signup_time', 'purchase_time', 'user_id'], 1, inplace=True)

    # 類別變數轉dummy
    # 日期衍伸變數不轉dummy，ensemble後F1更高。
    dummy_list = [
        'browser',
        'source',
        'country',
        # 'signup_month',
        # 'signup_hour',
        # 'signup_dayofweek',
        # 'purchase_month',
        # 'purchase_hour',
        # 'dayofweek',
        # 'dup_purchase'
    ]
    all_df = pd.get_dummies(all_df, prefix=dummy_list, columns=dummy_list)

    # 再將資料切為訓練、測試
    x_train = all_df[all_df.training == 1].copy()
    x_test = all_df[all_df.training == 0].copy()
    x_train.drop(['training'], 1, inplace=True)
    x_test.drop(['training'], 1, inplace=True)

    cols = x_train.columns
    # 標準化
    # Feature Scaling
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=0.20,
        random_state=5566
    )
    return x_train, x_valid, y_train, y_valid, x_test, cols, user_id

# 讀取現成檔案
data_set = GetData('fraud').load_data()

# 資料整理
x_train, x_valid, y_train, y_valid, x_test, cols, user_id = clean_data(data_set['train'], data_set['test'])

# 將資料集丟給LazyClassifier
clf = LazyClassifier(x_train, y_train, x_valid, y_valid, x_test)

# 讀取預設模型參數，可自行調整範圍
model_dict = clf.hyper_dict

# 訓練模型~
for k, v in model_dict.items():
    print()
    print(k)
    print('-' * 40)
    print()
    clf.fit(model_name=k, hyper_space=v['space'], hyper_objective=v['objective'], max_eval=20)
    print()
