#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import measure

def read_dic(filename, dtype=str):
    '''
    这里的dic的是包括2列的，：term + index，用tab分割
    '''
    f = file(filename, 'r')
    dic = {}
    count = 0
    for line in f.readlines():
        line = line.split("\t")
        count += 1
        if len(line) < 1:
            continue
        if len(line) == 1:
            dic[dtype(line[0].strip())] = count
        else:
            dic[dtype(line[0].strip())] = int(float(line[1]))
    f.close()
    return dic


def read_dic_ex(filename, dtype=str):
    '''
    这里的dic的是包括3列的，：term + index + global_weight，用tab分割
    dic的主键是index,
    '''
    f = file(filename, 'r')
    dic = {}
    global_weight = {}
    count = 0
    for line in f.readlines():
        line = line.split("\t")
        count += 1
        if len(line) < 1:
            continue
        if len(line) == 2:
            dic[dtype(line[0].strip())] = count
            global_weight[int(float(line[1]))] = 1
        else:
            dic[dtype(line[0].strip())] = int(float(line[1]))
            global_weight[int(float(line[1]))] = float(line[2])
    f.close()
    return dic,global_weight


def cons_pro_for_svm(text, dic, local_fun=measure.tf, global_weight=dict()):
    '''
    根据构造的输入的类标签和以及经过分词后的文本和词典，SVM分类所用的输入格式，会对特征向量进行归一化
        注意：这个实现已经去除了全局因子的影响，意味着特征权重直接使用词频。
    x begin from 1
    '''
    x = {}
    if len(global_weight) < 1:
        for i in range(len(dic) + 1):
            global_weight[i] = 1

    #构造特征向量
    for term in text:
        term = term.strip()
        if dic.has_key(term):
            index = int(dic.get(term))
            if x.has_key(index):
                x[index] += 1.0
            else:
                x[index] = 1.0
    # 计算特征向量的特征权重
    for key in x.keys():
        x[key] = local_fun(x[key]) * global_weight.get(key)

    #计算特征向量的模
    vec_sum = 0.0
    for key in x.keys():
        if x[key] != 0:
            vec_sum += x[key] ** 2.0
    #对向量进行归一化处理。
    vec_length = math.sqrt(vec_sum)
    if vec_length != 0:
        for key in x.keys():
            x[key] = float(x[key]) / vec_length
    return [x]


def classer_value(values):
    '''计算类得隶属度,libsvm采用的为one-against-one算法。
    liblinear采用的为oen-against-rest算法。因此在计算最终的隶属度分数上有所区别.
    计算公式为：sum(vi)/(2*k)+k/(2*n):n为所有参数类得总数,对libsvm为all-1,liblinear为1，k为支持该类的数,vi为支持该类的value
    '''
    n = len(values)
    max = 0
    for i in range(1, n - 1):
        if values[i] > values[max]:
            max = i
    size = 1
    k = 1
    init_score = values[max]
    return float(init_score)/(2.0*k)+float(k)/(2.0*size)