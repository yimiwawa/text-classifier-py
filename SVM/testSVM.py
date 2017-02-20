#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random
import train_svm, predict_svm

# 特征选择保留词的比例
ratio = 0.4
# 对于所有特征值为0的样本是否删除,True or False
delete = True
# 是否进行SVM模型参数的搜索。True即为使用SVM模型grid.搜索，False即为不使用参数搜索。
param_select = False
# svm_param  用户自己设定的svm的参数,例如liblinear参数"-c 0.2 "
svm_param = "-s 1 -c 0.2"
# 全局权重的计算方式：有"one","idf","rf","chi"
global_fun = "idf"
# 对特征向量计算特征权重时需要设定的计算方式:x(i,j) = local(i,j)*global(i).可选的有tf,logtf,binary
local_fun = "logtf"


# 参数详细说明
"""
options:
	-s type : set type of solver (default 1) 选择正则和损失计算方法
		对于多类别分类：
			 0 -- L2-regularized logistic regression (primal)
			 1 -- L2-regularized L2-loss support vector classification (dual)
			 2 -- L2-regularized L2-loss support vector classification (primal)
			 3 -- L2-regularized L1-loss support vector classification (dual)
			 4 -- support vector classification by Crammer and Singer
			 5 -- L1-regularized L2-loss support vector classification
			 6 -- L1-regularized logistic regression
			 7 -- L2-regularized logistic regression (dual)
	-c cost : set the parameter C (default 1) 设置参数C，用于惩罚噪声点，C越大，噪声对目标函数影响越大
	-p epsilon : set the epsilon in loss function of SVR (default 0.1)
	-e epsilon : set tolerance of termination criterion 设置终止标准
			-s 0 and 2
				|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,
				where f is the primal function, (default 0.01)
			-s 11
				|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)
			-s 1, 3, 4, and 7
				Dual maximal violation <= eps; similar to liblinear (default 0.)
			-s 5 and 6
				|f'(w)|_inf <= eps*min(pos,neg)/l*|f'(w0)|_inf,
				where f is the primal function (default 0.01)
			-s 12 and 13
				|f'(alpha)|_1 <= eps |f'(alpha0)|,
				where f is the dual function (default 0.1)
		-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)
		-wi weight: weights adjust the parameter C of different classes (see README for details)
		            调整每个类别的权重，例如"-w1 10,-w2 20",那么类1权重为10*C，类2权重为20*C，其他类别权重不变为C
		-v n: n-fold cross validation mode
		        设置交叉验证模式
		-q : quiet mode (no outputs)
"""

# 用户自定义模型名称
config_name = "mostClass.liblinear.conf"
model_name = "mostClass.liblinear.model"
train_name = "mostClass.liblinear.train"
param_name = "mostClass.liblinear.param"

trainFile = "../data/mostClass.train.seg"


def train(traindatas):
    # 模型保存的路径
    main_save_path = "../data/"
    train_svm.train(traindatas, main_save_path,
                    config_name, model_name, train_name, param_name, svm_param,
                    ratio, delete, param_select, global_fun, local_fun)


def predict(testdatas):
    model_path = "../data/model/"
    predict_svm.batch_predict(testdatas, model_path, model_name, config_name)


if __name__ == "__main__":
    all_datas = [line.strip() for line in open(trainFile).readlines()]
    random.shuffle(all_datas)
    train_datas = all_datas[:3 * len(all_datas) / 5]
    test_datas = all_datas[3 * len(all_datas) / 5:]

    train(train_datas)

    predict(test_datas)