#!/usr/bin/python
#_*_ coding: utf-8 _*_
#author:张知临 zhzhl202@163.com
#Filename: feature_select.py

'''利用卡方进行特征选择的程序'''

import math
import measure

def feature_select(train_docs, global_fun, dic_save_path, ratio, stop_words_dic):
    '''
    特征选择的主函数，输入的训练样本，内容的index，特征选择后的词典保存的路径及名称，卡方选择的比例默认为40%.
    最终会选择top $ratio 的作为最终的词典。
    '''
    dic, cat_num_dic, rows = cons_dic(train_docs, stop_words_dic)
    chi_score = chi_max_score(dic, cat_num_dic, rows)
    sorted_keys = sorted(chi_score.items(), key=lambda chi_score : chi_score[1], reverse=True)
    global_weight = measure.global_f(global_fun)(dic, cat_num_dic, rows)
    save_keys(sorted_keys[0:int(len(chi_score) * ratio)], global_weight, dic_save_path)

def cons_dic(train_docs, stop_words_dic):
    '''
    构造词典，主要是为了计算卡方的特征值。
       词典的形式：{term:{label:n(term)}} 每个term对应的是一个词典，其中词典中存储了每个类别中包含该term 的样本数。
       label 为int型
    '''
    dic = dict()  # 词典
    cat_num_dic = dict()  # 各个类别的个数
    
    rows = 0  # 总的样本个数
    count = 0
    for line in train_docs:
        # sample = line.strip().split(tc_splitTag)
        label, string = line.strip().split("\t")
        count += 1
        rows += 1  # 统计样本的总行数
        #统计各个样本的个数
        label = int(float(label))
        if cat_num_dic.has_key(label) is True:
            cat_num_dic[label] += 1.0
        else:
            cat_num_dic[label] = 1.0
        #计算包含term的样本数      
        temp_dic = dict() 
        for term in string.strip().split(" "):
            if len(term.strip()) == 0:
                continue
            term = term.strip()
            if stop_words_dic.has_key(term) is True:
                continue
            temp_dic[term] = 1.0  # 只要出现一次就代表着该样本包含该词。
            
        for key in temp_dic.keys():
            if dic.has_key(key) is True:
                if dic[key].has_key(label):
                    dic[key][label] += 1.0
                else:
                    dic[key][label] = 1
            else:
                dic[key] = {}
                dic[key][label] = 1
    for key in dic.keys():
        for cat in cat_num_dic.keys():
            if dic[key].has_key(cat) is False:
                dic[key][cat] = 0.0
    return dic, cat_num_dic, rows

def chi_max_score(dic, cat_num_dic, rows):
    '''利用卡方公式计算每个term的分值'''
    term_score = dict()
    for term in dic.keys():
        term = term.strip()
        all_num_term = float(sum(dic[term].values()))
        all_num_cat = float(sum(cat_num_dic.values()))
        chi_score = 0.0
        for cat in cat_num_dic.keys():
            A = float(dic[term][cat])  # 既在该类也包含该term
            B = float(cat_num_dic[cat] - A)  # 在该类中但不包含该term
            C = float(all_num_term - A)  # 不在该类但包含该term
            D = float(all_num_cat - all_num_term - cat_num_dic[cat] + A)  # 既不包含term也不在该类中。
            if (A + C) * (B + D) * (A + B) * (C + D) == 0:
                chi_score = 0
            else:
                chi_score = max(chi_score, rows * math.pow(A * D - B * C, 2) / ((A + C) * (B + D) * (A + B) * (C + D)))
        term_score[term] = chi_score
    return term_score


def save_keys(keys_truple, global_weight, save_path):
    '''the format of the key :keys,index,frequant'''  
    f = file(save_path, 'w')
    count = 0
    for truple in keys_truple:
        count += 1
        f.write(truple[0].strip() + "\t" + str(count) + "\t" + str(global_weight[truple[0].strip()]) + "\n")
    f.close()

