"""
特征组合：Dict+GroupBy
特征选择方式：Pearson
参数寻优办法：GridSearch
模型：randomforest

"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def read_data(debug=True):
    """
    读取数据
    :param debug:是否调试版，可以极大节省debug时间
    :return:训练集，测试集
    """

    print("read_data...")

    train = pd.read_csv("clean/train_encode_norm.csv")
    test = pd.read_csv("clean/test_encode_norm.csv")
    train['is_test'] = 1 - train['predict_train']
    train.drop(columns=['predict_train'], inplace=True)
    train.sort_values('is_test', ascending=False, inplace=True)
    train = train.reset_index(drop=True)
    train = train.iloc[:int(0.5 * len(train)), ]

    print(train.columns.to_list())

    return train, test

def feature_select_pearson(train, test):
    """
    利用pearson系数进行相关性特征选择
    :param train:训练集
    :param test:测试集
    :return:经过特征选择后的训练集与测试集
    """
    print('feature_select...')
    features = train.columns.tolist()
    features.remove("CUST_UID")
    features.remove("LABEL")
    features.remove("is_test")
    featureSelect = features[:]

    # 进行pearson相关性计算
    corr = []
    for fea in featureSelect:
        corr.append(abs(train[[fea, 'LABEL']].fillna(0).corr().values[0][1]))

    # 取top300的特征进行建模，具体数量可选
    se = pd.Series(corr, index=featureSelect).sort_values(ascending=False)
    print(se)
    feature_select = ['CUST_UID'] + se[:20].index.tolist()
    print('done')
    return train[feature_select + ['LABEL']], test[feature_select]

def param_grid_search(train):
    """
    网格搜索参数寻优
    :param train:训练集
    :return:最优的分类器模型
    """
    print('param_grid_search')
    features = train.columns.tolist()
    features.remove("CUST_UID")
    features.remove("LABEL")
    features.remode("is_test")
    parameter_space = {
        "n_estimators": [120],
        "min_samples_leaf": [30],
        "min_samples_split": [2],
        "max_depth": [9],
        "max_features": ["auto", 15]
    }

    print("Tuning hyper-parameters for auc")
    clf = RandomForestClassifier(
        criterion="gini",
        min_weight_fraction_leaf=0.,
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=True,
        n_jobs=4,
        random_state=2022,
        verbose=0,
        warm_start=False)
    grid = GridSearchCV(clf, parameter_space, cv=3, scoring="roc_auc")
    grid.fit(train[features].values, train['LABEL'].values)

    print("best_params_:")
    print(grid.best_params_)
    means = grid.cv_results_["mean_test_score"]
    stds = grid.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, grid.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print(grid.best_params_)
    return grid.best_estimator_


def train_predict(train, test, best_clf):
    """
    进行训练和预测输出结果
    :param train:训练集
    :param test:测试集
    :param best_clf:最优的分类器模型
    :return:
    """
    print('train_predict...')
    features = train.columns.tolist()
    features.remove("CUST_UID")
    features.remove("LABEL")

    prediction_test = 0
    cv_score = []
    prediction_train = pd.Series()
    kf = KFold(n_splits=5, random_state=2022, shuffle=True)
    for train_part_index, eval_index in kf.split(train[features], train['LABEL']):
        best_clf.fit(train[features].loc[train_part_index].values, train['LABEL'].loc[train_part_index].values)
        prediction_test += best_clf.predict_proba(test[features].values)[:, 1]
        eval_pre = best_clf.predict_proba(train[features].loc[eval_index].values)[:, 1]
        score = roc_auc_score(train['LABEL'].loc[eval_index], eval_pre)
        cv_score.append(score)
        print(score)
        prediction_train = prediction_train.append(
            pd.Series(best_clf.predict_proba(train[features].loc[eval_index])[:, 1],
                      index=eval_index))
    print(cv_score, sum(cv_score) / 5)
    pd.Series(prediction_train.sort_index().values).to_csv("preprocesses/train_randomforest.csv", index=False)
    pd.Series(prediction_test / 5).to_csv("preprocesses/test_randomforest.csv", index=False)
    test['LABEL'] = prediction_test / 5
    a = test[['CUST_UID', 'LABEL']]
    a.to_csv('result/rf_submission.txt', sep=' ', index=False, header=None)
    return


if __name__ == "__main__":
    # 获取训练集与测试集
    train, test = read_data(debug=False)

    # 获取特征选择结果
    train, test = feature_select_pearson(train, test)

    # 获取最优分类器模型
    best_clf = param_grid_search(train)

    # 获取结果
    train_predict(train, test, best_clf)
# [0.9421743179301318, 0.9511711066582075, 0.9514210868118008, 0.9440950864226438, 0.9521431957111568] 0.9482009587067882