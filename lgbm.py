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
from sklearn.metrics import mean_squared_error

def read_data(debug=True):
    """

    :param debug:
    :return:
    """
    print("read_data...")
    train = pd.read_csv("preprocesses/train_dict_encode.csv")
    test = pd.read_csv("preprocesses/test_dict_encode.csv")
    print("done")
    return train, test


def feature_select_wrapper(train, test):
    """

    :param train:
    :param test:
    :return:
    """
    print('feature_select_wrapper...')
    label = 'LABEL'
    features = train.columns.tolist()
    category_features = ['Categorical_0', 'Categorical_1', 'Categorical_2', 'Categorical_3', 'Categorical_4', 'Categorical_5']
    features.remove('CUST_UID')
    features.remove('LABEL')

    # 配置模型的训练参数
    params_initial = {
        'num_leaves': 31,
        'learning_rate': 0.2,
        'boosting': 'rf',
        'min_child_samples': 20,
        'bagging_seed': 2022,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'feature_fraction': 0.8,
        'max_depth': -1,
        'metric': 'auc',
        'reg_alpha': 0,
        'reg_lambda': 1,
        'objective': 'binary'
    }
    ESR = 100
    NBR = 10000
    VBE = 50
    kf = KFold(n_splits=10, random_state=2022, shuffle=True)
    fse = pd.Series(0, index=features)
    for train_part_index, eval_index in kf.split(train[features], train[label]):
        # 模型训练

        train_part = lgb.Dataset(train[features].loc[train_part_index],
                                 train[label].loc[train_part_index], feature_name=features, categorical_feature=category_features)
        eval = lgb.Dataset(train[features].loc[eval_index],
                           train[label].loc[eval_index], feature_name=features, categorical_feature=category_features)
        bst = lgb.train(params_initial, train_part, num_boost_round=NBR,
                        valid_sets=[train_part, eval],
                        valid_names=['train', 'valid'],
                        early_stopping_rounds=ESR, verbose_eval=VBE)
        fse += pd.Series(bst.feature_importance(), features)

    feature_select = ['CUST_UID'] + fse.sort_values(ascending=False).index.tolist()[:70]
    print('done')
    return train[feature_select + ['LABEL']], test[feature_select]


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
    features = train.columns.tolist()
    features.remove('CUST_UID')
    features.remove('LABEL')
    train_data = lgb.Dataset(train[features], train[label], silent=False)
    def hyperopt_objective(params):
        """

        :param params:
        :return:
        """
        params = params_append(params)
        print(params)
        res = lgb.cv(params, train_data, 1000,
                     nfold=10,
                     stratified=False,
                     shuffle=True,
                     metrics='auc',
                     early_stopping_rounds= 50,
                     verbose_eval=False,
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
        'min_child_samples': hp.choice('min_child_samples', list(range(1, 30, 5)))
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
    features = train.columns.tolist()
    features.remove('CUST_UID')
    features.remove('LABEL')
    category_features = ['Categorical_0', 'Categorical_1', 'Categorical_2', 'Categorical_3', 'Categorical_4',
                         'Categorical_5']
    params = params_append(params)
    kf = KFold(n_splits=10, random_state=2022, shuffle=True)
    prediction_test = 0
    cv_score = []
    prediction_train = pd.Series()
    ESR = 100
    NBR = 1000
    VBE = 50
    for train_part_index, eval_index in kf.split(train[features], train[label]):
        # 模型训练
        train_part = lgb.Dataset(train[features].loc[train_part_index],
                                 train[label].loc[train_part_index], feature_name=features, categorical_feature=category_features)
        eval = lgb.Dataset(train[features].loc[eval_index],
                           train[label].loc[eval_index], feature_name=features, categorical_feature=category_features)
        bst = lgb.train(params, train_part, num_boost_round=NBR,
                        valid_sets=[train_part, eval],
                        valid_names=['train', 'valid'],
                        early_stopping_rounds=ESR, verbose_eval=VBE)
        prediction_test += bst.predict(test[features])
        prediction_train = prediction_train.append(pd.Series(bst.predict(train[features].loc[eval_index]),
                                                             index=eval_index))
        eval_pre = bst.predict(train[features].loc[eval_index])
        score = np.sqrt(mean_squared_error(train[label].loc[eval_index].values, eval_pre))
        cv_score.append(score)
    print(cv_score, sum(cv_score) / 10)
    pd.Series(prediction_train.sort_index().values).to_csv("preprocesses/train_lightgbm.csv", index=False)
    pd.Series(prediction_test / 10).to_csv("preprocesses/test_lightgbm.csv", index=False)
    test['LABEL'] = prediction_test / 10
    test[['CUST_UID', 'LABEL']].to_csv("result/submission_lightgbm.txt", index=False, sep=' ', header=None)
    return

if __name__ == "__main__":
    train, test = read_data(debug=False)
    # train, test = feature_select_wrapper(train, test)
    best_clf = param_hyperopt(train)
    train_predict(train, test, best_clf)
# [3.686192535745703, 3.647032390847285, 3.706089838227353, 3.773664215095074, 3.5735473296458626] 3.677305261912256
