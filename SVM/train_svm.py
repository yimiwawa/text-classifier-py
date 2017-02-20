#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import utils
import feature_select
import measure
import grid_search_param
from dependence import liblinearutil

stopword_filename = '../data/stop_words.txt'


def train(train_docs, main_save_path,
          config_name, model_name, train_name, param_name, svm_param, ratio, delete,
          param_select, global_fun, local_fun):
    '''
    训练的自动化程序，分词,先进行特征选择，重新定义词典，根据新的词典，自动选择SVM最优的参数。
    然后使用最优的参数进行SVM分类，最后生成训练后的模型。
    需要保存的文件：（需定义一个主保存路径）
                 模型文件：词典.key+模型.model
                临时文件 ：svm分类数据文件.train
    '''

    print "-----------------创建模型文件保存的路径-----------------"
    if os.path.exists(main_save_path):
        if os.path.exists(os.path.join(main_save_path, "model")) is False:
            os.makedirs(os.path.join(main_save_path, "model"))
    if os.path.exists(main_save_path):
        if os.path.exists(os.path.join(main_save_path, "temp")) is False:
            os.makedirs(os.path.join(main_save_path, "temp"))

    #读取停用词文件
    if stopword_filename == "":
        stop_words_dic = dict()
    else:
        stop_words_dic = utils.read_dic(stopword_filename)

    print "-----------------现在正在进行特征选择---------------"
    dic_path = os.path.join(main_save_path, "model", "dic.key")
    feature_select.feature_select(train_docs, global_fun, dic_path, ratio, stop_words_dic)

    print "-----------------再根据特征选择后的词典构造新的SVM分类所需的训练样本------------------- "
    problem_save_path = os.path.join(main_save_path, "temp", train_name)
    label = cons_train_sample_for_cla(train_docs, measure.local_f(local_fun), dic_path, problem_save_path, delete)

    print"--------------------选择最优的c,g------------------------------"
    if param_select is True:
        search_result_save_path = os.path.join(main_save_path, "temp", param_name)

        coarse_c_range = (-5, 7, 2)
        coarse_g_range = (1, 1, 1)
        fine_c_step = 0.5
        fine_g_step = 0
        c, g = grid_search_param.grid(problem_save_path, search_result_save_path, coarse_c_range,
                                      coarse_g_range, fine_c_step, fine_g_step)
        svm_param = " -c " + str(c)

    print "-----------------训练模型，并将模型进行保存----------"
    model_save_path = os.path.join(main_save_path, "model", model_name)
    ctm_train_model(problem_save_path, svm_param, model_save_path)

    print "-----------------保存模型配置-----------------"
    f_config = file(os.path.join(main_save_path, "model", config_name), 'w')
    save_config(f_config, model_name, local_fun, global_fun, svm_param, label)
    f_config.close()

    print "-----------------训练结束---------------------"


def cons_train_sample_for_cla(train_docs, local_fun, dic_path, sample_save_path, delete):
    '''根据提供的词典，将指定文件中的指定位置上的内容构造成SVM所需的问题格式，并进行保存'''
    dic_list, global_weight = utils.read_dic_ex(dic_path, dtype=str)
    local_fun = measure.local_f(local_fun)
    label = set()

    fs = file(sample_save_path, 'w')
    for line in train_docs:
        y, string = line.strip().split("\t")
        x = utils.cons_pro_for_svm(string.strip().split(" "), dic_list, local_fun, global_weight)
        y = [float(y)]
        if delete is True and len(x[0]) == 0:
            continue
        save_dic_train_sample(fs, y, x)
        label.add(y[0])
    fs.close()
    return label


def save_dic_train_sample(f, y, x):
    '''
    将构造的svm问题格式进行保存
    y为list x为list里面为 词典。[ {} ]
    '''
    for i in range(len(y)):
        f.write(str(y[i]))
        sorted_keys = x[i].keys()
        sorted_keys.sort()
        for key in sorted_keys:
            f.write("\t" + str(key) + ":" + str(x[i][key]))
        f.write("\n")


def ctm_train_model(sample_save_path, param, model_save_path):
    '''训练模型，输入样本文件，训练的参数，模型的保存地址，最后会给出模型在训练样本上的测试结果。'''
    y, x = liblinearutil.svm_read_problem(sample_save_path)
    m = liblinearutil.train(y, x, param)
    liblinearutil.save_model(model_save_path, m)
    labels = {}.fromkeys(y).keys()
    if len(labels) > 2:
        pred_labels, (Micro, Macro, ACC), pred_values = liblinearutil.predict(y, x, m)
        print "(Micro=%g, Macro=%g, ACC=%g)" % (Micro, Macro, ACC)
    else:
        pred_labels, (f_score,recall,presion), pred_values = liblinearutil.predict(y, x, m)
        print "(f_score=%g,recall=%g,presion=%g)" % (f_score, recall, presion)
    return m


def save_config(f, model_name, local_fun, global_fun, svm_param, label):
    '''保存模型配置文件'''
    f.write("SvmType:" + str("liblinear").strip() + "\n")
    f.write("SvmParam:" + str(svm_param).strip() + "\n")
    f.write("DicName:" + str("dic.key").strip() + "\n")
    f.write("ModelName:" + str(model_name).strip() + "\n")
    f.write("LocalFun:" + str(local_fun).strip() + "\n")
    f.write("GlobalFun:" + str(global_fun).strip() + "\n")
    f.write("WordSeg:" + str("jieba").strip()+"\n")
    f.write("Date:" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())) + "\n")
    f.write("Labels:\n{\n")  # 将类标签写入，类标签会以"label,descr"进行存储
    for l in label:
        f.write(str(int(l)) + ",\n")
    f.write("}\n")