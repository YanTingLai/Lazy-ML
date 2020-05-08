# -*- coding: utf-8 -*-
"""
Created on 2020/04/28 11:26:54

@author: 10356
"""
import os
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import warnings
import time
import numpy as np
from hyperopt import fmin, tpe, hp, Trials
from xgboost import XGBClassifier
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, fbeta_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# https://www.itread01.com/content/1545123188.html
# https://www.kaggle.com/eikedehling/tune-and-compare-xgb-lightgbm-rf-with-hyperopt
# https://districtdatalabs.silvrback.com/parameter-tuning-with-hyperopt
# scoring https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter


def get_data(data_cate: str):
    """
    讀取資料檔
    :param data_cate: 資料集類別，train, test
    :return: Pandas DataFrame
    """
    type_dict = {
        'train': {'user_id': 'str', 'device_id': 'str', 'age': 'int', 'sex': 'int', 'browser': 'str', 'source': 'str', 'country': 'str', 'purchase_value': 'int', 'signup_time': 'str', 'purchase_time': 'str', 'TXCNT': 'int', 'class': 'int'},
        'test': {'user_id': 'str', 'device_id': 'str', 'age': 'int', 'sex': 'int', 'browser': 'str', 'source': 'str', 'country': 'str', 'purchase_value': 'int', 'signup_time': 'str', 'purchase_time': 'str', 'TXCNT': 'int'}
    }
    parse_dates = ['signup_time', 'purchase_time']
    file_path = os.path.join(os.getcwd(), data_cate) + '.csv'
    data_set = pd.read_csv(file_path,
                           header=0,
                           keep_default_na=True,
                           dtype=type_dict[data_cate],
                           encoding='utf-8',
                           parse_dates=parse_dates)
    return data_set


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
    country_list = [k for k, v in Counter(train_df.country.tolist()).items() if v < 5]
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


def plot_result(y_t, y_p):
    """
    畫混淆矩陣與分類結果
    :param y_t: 實際y
    :param y_p: 預測y
    :return: None
    """
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_t, y_p))
    print()
    print("Classification Report:")
    print(classification_report(y_t, y_p))
    pass


def rf_objective(params):
    params = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'max_features': params['max_features'],
        "min_samples_split": params['min_samples_split'],
        "min_samples_leaf": int(params['min_samples_leaf']),
    }
    clf = RandomForestClassifier(n_jobs=8, class_weight='balanced', **params)
    score = cross_val_score(clf, x_train, y_train, scoring='f1', cv=StratifiedKFold()).mean()
    print("F1: {:.3f} params {}".format(score, params))
    return -score


def xgb_objective(params):
    params = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'learning_rate': params['learning_rate'],
        'min_child_weight': int(params['min_child_weight']),
        'colsample_bytree': params['colsample_bytree'],
        'gamma': params['gamma'],
    }
    clf = xgb.XGBClassifier(n_jobs=8, **params)
    score = cross_val_score(clf, x_train, y_train, scoring='f1', cv=StratifiedKFold()).mean()
    print("F1: {:.3f} params {}".format(score, params))
    return -score


def bgm_objective(params):
    params = {
        'n_estimators': int(params['n_estimators']),
        'num_leaves': int(params['num_leaves']),
        'colsample_bytree': params['colsample_bytree'],
        'learning_rate': params['learning_rate'],
    }
    clf = lgb.LGBMClassifier(**params)
    score = cross_val_score(clf, x_train, y_train, scoring='f1', cv=StratifiedKFold()).mean()
    print("F1: {:.5f} params {}".format(score, params))
    return -score


def hyper_tune(model_name, hyper_space, hyper_objective):
    trials = Trials()
    best = fmin(
        fn=hyper_objective,
        space=hyper_space,
        algo=tpe.suggest,
        max_evals=max_opt,
        trials=trials
    )
    print("{} Hyperopt estimated optimum {}".format(model_name, best))

    if model_name == 'lightGBM':
        best['n_estimators'] = int(best['n_estimators'])
        best['num_leaves'] = int(best['num_leaves'])
        hyper_model = lgb.LGBMClassifier(**best)
        hyper_model_ = lgb.LGBMClassifier()
    elif model_name == 'XGBoost':
        best['n_estimators'] = int(best['n_estimators'])
        best['max_depth'] = int(best['max_depth'])
        hyper_model = XGBClassifier(**best)
        hyper_model_ = XGBClassifier()
    elif model_name == 'RandomForest':
        best['n_estimators'] = int(best['n_estimators'])
        best['min_samples_leaf'] = int(best['min_samples_leaf'])
        hyper_model = RandomForestClassifier(**best)
        hyper_model_ = RandomForestClassifier()
    else:
        return 'Unknown Model: {}'.format(model_name)

    hyper_model.fit(x_train, y_train)
    hyper_model_.fit(x_train, y_train)
    predict_ = hyper_model.predict(x_valid)
    predict__ = hyper_model_.predict(x_valid)
    f_score = fbeta_score(y_valid, predict_, beta=1, average='binary')
    acc_score = accuracy_score(y_valid, predict_)
    print('f_score: {:2.4f}, acc_score: {:2.4f}'.format(f_score, acc_score))
    print()
    print('Default parameters.', '-'*20)
    plot_result(y_valid, predict_)
    print('Hyperopt parameters.', '-'*20)
    plot_result(y_valid, predict__)
    hyper_plot(hyper_space.keys(), trials, model_name)


def hyper_plot(parameters, trials, name):
    f, axes = plt.subplots(nrows=1, ncols=len(parameters), figsize=(15, 5))
    plt.title('model_' + name)
    cmap = plt.cm.jet
    for i, val in enumerate(parameters):
        xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
        ys = [-t['result']['loss'] for t in trials.trials]
        xs, ys = zip(*sorted(zip(xs, ys)))
        ys = np.array(ys)
        axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75, c=cmap(float(i) / len(parameters)))
        axes[i].set_title(val)

    plt.savefig('hyperopt_plt_' + name + '.png')
    plt.cla()


# 讀檔
train_df, test_df = get_data('train'), get_data('test')


# 清理資料
x_train, x_valid, y_train, y_valid, x_test, cols, user_id = clean_data(train_df, test_df)


# 模型調參

"""
新增模型流程：
1. model_dict加入模型名稱與參數範圍
2. 建立objective
3. hyper_tune中if增加判斷
"""

max_opt = 3
model_dict = {
    'lightGBM': {
        'space': {
            'n_estimators': hp.quniform('n_estimators', 5, 200, 5),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
            'num_leaves': hp.quniform('num_leaves', 8, 128, 2),
            'learning_rate': hp.loguniform('learning_rate', 1e-2, 5e-1),
        },
        'objective': bgm_objective,
        'trials': {},
    },
    'XGBoost': {
        'space': {
            'n_estimators': hp.quniform('n_estimators', 5, 200, 5),
            'learning_rate': hp.uniform('learning_rate', 1e-2, 5e-1),
            'max_depth': hp.quniform('max_depth', 1, 12, 1),
            'min_child_weight': hp.quniform('min_child_weight', 1, 12, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
            'gamma': hp.uniform('gamma', 0.0, 0.5),
        },
        'objective': xgb_objective,
        'trials': {},
    },
    'RandomForest': {
        'space': {
            'n_estimators': hp.quniform('n_estimators', 5, 200, 5),
            'max_depth': hp.quniform('max_depth', 1, 15, 1),
            'max_features': hp.uniform('max_features', 0.0, 0.5),
            "min_samples_split": hp.uniform('min_samples_split', 0.0, 0.5),
            "min_samples_leaf": hp.quniform('min_samples_leaf', 1, 10, 1),
        },
        'objective': rf_objective,
        'trials': {},
    },
}

for k, v in model_dict.items():
    print()
    print(k)
    print('-' * 40)
    print()
    hyper_tune(k, v['space'], v['objective'])
    print()
