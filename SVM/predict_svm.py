#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import utils
import measure
from SVM.dependence import liblinearutil


def predict(text, model_dir, model_file, conf_file):
    '''一般形式的下得模型预测，即单个模型。'''
    dic, global_weight = utils.read_dic_ex(os.path.join(model_dir, "dic.key"), dtype=str)
    model = liblinearutil.load_model(os.path.join(model_dir, model_file))
    local_fun = load_conf(model_dir, conf_file)

    text_list = text.strip().split(" ")

    print "-----------------正在对样本进行预测-------------------"
    x = utils.cons_pro_for_svm(text_list, dic, local_fun, global_weight)
    p_lab, p_acc, p_sc = liblinearutil.predict("", x, model)

    label = p_lab[0]
    score = utils.classer_value(p_sc[0])

    return label, score


def batch_predict(testdocs, model_dir, model_file, conf_file):
    '''一般形式的下得模型预测，即单个模型。'''
    dic, global_weight = utils.read_dic_ex(os.path.join(model_dir, "dic.key"), dtype=str)
    model = liblinearutil.load_model(os.path.join(model_dir, model_file))
    local_fun = load_conf(model_dir, conf_file)

    right = 0
    total = 0
    for text in testdocs:
        terms = text.strip().split("\t")

        x = utils.cons_pro_for_svm(terms[1].strip().split(" "), dic, local_fun, global_weight)
        p_lab, p_acc, p_sc = liblinearutil.predict("", x, model)

        label = p_lab[0]
        score = utils.classer_value(p_sc[0])

        if float(terms[0]) == label:
            right += 1
        total += 1

    print "Total test docs: ", total
    print "Predict right docs: ", right
    print "Precision: ", right * 1.0 / total


def load_conf(model_dir, conf_file):
    f = file(os.path.join(model_dir, conf_file), 'r')
    for line in f.readlines():
        text = line.split(":")
        if text[0].strip() == "LocalFun":
            local_fun = measure.local_f(text[1].strip())
    return local_fun
