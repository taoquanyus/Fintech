"""
特征组合：Dict+GroupBy+nlp
特征选择方式：Wrapper
参数寻优办法：hyperopt
模型：lightgbm
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from hyperopt import hp, fmin, tpe
from numpy.random import RandomState
from sklearn.metrics import roc_auc_score


def read_data(debug=True):
    """

    :param debug:
    :return:
    """
    print("read_data...")
    train = pd.read_csv("clean/train_encode_norm.csv")
    test = pd.read_csv("clean/test_encode_norm.csv")
    train['is_test'] = 1 - train['predict_train']
    train.drop(columns=['predict_train'], inplace=True)
    train.sort_values('is_test', ascending=False, inplace=True)
    train = train.reset_index(drop=True)
    train = train.iloc[:int(0.5 * len(train)),]
    print(train.columns.to_list())
    return train, test


def feature_select_wrapper(train, test):
    """

    :param train:
    :param test:
    :return:
    """
    print('feature_select_wrapper...')
    label = 'LABEL'
    is_test = 'is_test'
    features = train.columns.tolist()
    features.remove('CUST_UID')
    features.remove('LABEL')
    features.remove('is_test')
    # 配置模型的训练参数
    params_initial = {
        'num_leaves': 31,
        'learning_rate': 0.2,
        'boosting': 'gbdt',
        'min_child_samples': 20,
        'bagging_seed': 1998,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'feature_fraction': 0.8,
        'max_depth': -1,
        'metric': 'auc',
        'reg_alpha': 0,
        'reg_lambda': 1,
        'objective': 'binary'
    }
    ESR = 30
    NBR = 10000
    VBE = 300
    kf = KFold(n_splits=5, random_state=2022, shuffle=True)
    fse = pd.Series(0, index=features)
    for train_part_index, eval_index in kf.split(train[features], train[label]):
        # 模型训练
        train_part = lgb.Dataset(train[features].loc[train_part_index],
                                 train[label].loc[train_part_index], weight=train[is_test].loc[train_part_index])
        eval = lgb.Dataset(train[features].loc[eval_index],
                           train[label].loc[eval_index], weight=train[is_test].loc[eval_index])
        bst = lgb.train(params_initial, train_part, num_boost_round=NBR,
                        valid_sets=[train_part, eval],
                        valid_names=['train', 'valid'],
                        early_stopping_rounds=ESR, verbose_eval=VBE)
        fse += pd.Series(bst.feature_importance(), features)

    feature_select = ['CUST_UID'] + fse.sort_values(ascending=False).index.tolist()[:20]
    print(fse.sort_values(ascending=False))
    print('done')
    return train[feature_select + ['LABEL'] + ['is_test']], test[feature_select]


def params_append(params):
    """

    :param params:
    :return:
    """
    params['objective'] = 'binary'
    params['metric'] = 'auc'
    params['bagging_seed'] = 2022
    params['feature_pre_filter'] = False
    return params


def param_hyperopt(train):
    """

    :param train:
    :return:
    """
    label = 'LABEL'
    is_test = 'is_test'
    features = train.columns.tolist()
    features.remove('CUST_UID')
    print(features)
    features.remove('LABEL')
    features.remove('is_test')
    category_features = []
    for f in features:
        if f[:11] == 'Categorical':
            category_features.append(f)
    print(category_features)
    train_data = lgb.Dataset(train[features], train[label], weight=train[is_test], silent=True)

    def hyperopt_objective(params):
        """

        :param params:
        :return:
        """
        params = params_append(params)
        print(params)
        res = lgb.cv(params, train_data, 1000,
                     nfold=3,
                     stratified=False,
                     shuffle=True,
                     metrics='auc',
                     early_stopping_rounds=30,
                     verbose_eval=True,
                     show_stdv=False,
                     seed=2022)
        return max(res['auc-mean'])

    params_space = {
        'learning_rate': hp.uniform('learning_rate', 1e-3, 5e-1),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
        'num_leaves': hp.choice('num_leaves', list(range(10, 300, 10))),
        'reg_alpha': hp.randint('reg_alpha', 0, 10),
        'reg_lambda': hp.uniform('reg_lambda', 0, 10),
        'bagging_freq': hp.randint('bagging_freq', 1, 10),
        'min_child_samples': hp.choice('min_child_samples', list(range(1, 30, 5))),
        'max_drop': hp.randint('max_drop', 50, 200)
    }
    params_best = fmin(
        hyperopt_objective,
        space=params_space,
        algo=tpe.suggest,
        max_evals=30,
        rstate=np.random.default_rng(2022))
    return params_best


def train_predict(train, test, params):
    """

    :param train:
    :param test:
    :param params:
    :return:
    """
    label = 'LABEL'
    is_test = 'is_test'
    features = train.columns.tolist()
    features.remove('CUST_UID')
    features.remove('LABEL')
    features.remove('is_test')
    params = params_append(params)
    kf = KFold(n_splits=5, random_state=2022, shuffle=True)
    prediction_test = 0
    cv_score = []
    prediction_train = pd.Series()
    ESR = 30
    NBR = 1000
    VBE = 30
    for train_part_index, eval_index in kf.split(train[features], train[label]):
        # 模型训练
        train_part = lgb.Dataset(train[features].loc[train_part_index],
                                 train[label].loc[train_part_index], weight=train[is_test].loc[train_part_index])
        eval = lgb.Dataset(train[features].loc[eval_index],
                           train[label].loc[eval_index], weight=train[is_test].loc[eval_index])
        bst = lgb.train(params, train_part, num_boost_round=NBR,
                        valid_sets=[train_part, eval],
                        valid_names=['train', 'valid'],
                        early_stopping_rounds=ESR, verbose_eval=VBE)
        prediction_test += bst.predict(test[features])
        prediction_train = prediction_train.append(pd.Series(bst.predict(train[features].loc[eval_index]),
                                                             index=eval_index))
        eval_pre = bst.predict(train[features].loc[eval_index])
        score = roc_auc_score(train[label].loc[eval_index].values, eval_pre)
        cv_score.append(score)
    print(cv_score, sum(cv_score) / 5)
    pd.Series(prediction_train.sort_index().values).to_csv("preprocesses/train_lightgbm_norm.csv", index=False)
    pd.Series(prediction_test / 5).to_csv("preprocesses/test_lightgbm_norm.csv", index=False)
    test['LABEL'] = prediction_test / 5
    test[['CUST_UID', 'LABEL']].to_csv("result/submission_lightgbm_norm.txt", index=False, sep=' ', header=None)
    return


if __name__ == "__main__":
    train, test = read_data(debug=False)
    train, test = feature_select_wrapper(train, test)
    best_clf = param_hyperopt(train)
    train_predict(train, test, best_clf)
    # [0.9412433303130977, 0.9489010890071212, 0.9481599409312019, 0.9417089115106443, 0.9486615752999248] 0.9457349694123979
