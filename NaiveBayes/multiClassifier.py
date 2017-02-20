#! -*- coding:utf-8 -*-

# __author__ = "houlisha"

import json
import math
import numpy as np

stopwords = set([line.strip() for line in open("../data/stop_words.txt").readlines()])


def featureSelecter(textdatas, method):
    def chi(textdatas):
        '''
            A: 包含t    属于c     的文档数
            B: 不包含t  属于c     的文档数
            C: 包含t    不属于c   的文档数
            D: 不包含t  不属于c   的文档数
            kafang(t,c) = N * (AD-BC)^2 / (A+C)(B+D)(A+B)(C+D)
        '''
        term_cate_dict = {}  # 词t属于类别c的文档数，A
        cate_doc_dict = {}   # 属于类别c中的文档数， A+B
        n_docs = len(textdatas)  # 文档总数， N
        for line in textdatas:
            cate, words = line.split("\t")
            cate_doc_dict[cate] = cate_doc_dict.get(cate, 0.0) + 1.0
            terms = set(words.split(" "))
            for t in terms:
                if t not in stopwords:
                    tc_dict = term_cate_dict.get(t, {})
                    tc_dict[cate] = tc_dict.get(cate, 0.0) + 1.0
                    term_cate_dict[t] = tc_dict

        # 计算
        term_score = {}
        for term, tc_dict in term_cate_dict.items():
            term_in = sum(tc_dict.values())
            cate_in = cate_doc_dict[cate]
            chi_score_max = 0.0
            for cate, A in tc_dict.items():
                B = cate_in - A
                C = term_in - A
                D = float(n_docs) - cate_in - term_in + A
                tmp = (A + C) * (B + D) * (A + B) * (C + D)
                if tmp != 0.0:
                    chi_score = n_docs * math.pow(A * D - B * C, 2) / tmp
                chi_score_max = max(chi_score, chi_score_max)
            term_score[term] = chi_score_max

        # sort and select
        term_score_sorted = sorted(term_score.items(), key=lambda x: x[1], reverse=True)
        limit = len(term_score_sorted) * 2 / 5
        return dict(term_score_sorted[:limit])

    def tf(textdatas):
        features = {}
        for line in textdatas:
            words = line.strip().split('\t')[1].split(' ')
            for word in words:
                if word not in stopwords:
                    features[word] = features.get(word, 0) + 1
        features_sorted = sorted(features.items(), key=lambda x: x[1], reverse=True)
        limit = len(features_sorted) * 2 / 5
        return dict(features_sorted[:limit])

    if method == "chi":
        return chi(textdatas)

    if method == "tf":
        return tf(textdatas)


def train(textdatas, features):
    # output:
    #   feature_log_prob: array-like, shape = [n_features, n_categorys]
    categorys = list(set([line.split('\t')[0] for line in textdatas]))

    def cate2id(label):
        return categorys.index(label)

    n_docs = len(textdatas)
    n_categorys = len(categorys)
    n_features = len(features)

    category_counts = np.zeros(n_categorys)             # 每类包含文章数量
    feature_counts = {}        # 每个特征词在每个类别中出现的次数
    total_feature_counts = np.zeros(n_categorys)    # 每个类别中所有特征词个数
    for line in textdatas:
        cate, words = line.split('\t')
        cate_index = cate2id(cate)
        category_counts[cate_index] += 1
        for w in words.split(' '):
            if features.has_key(w):
                fc = feature_counts.get(w, np.zeros(n_categorys))
                fc[cate_index] += 1
                feature_counts[w] = fc
                total_feature_counts[cate_index] += 1

    category_counts /= float(n_docs)

    # 拉普拉斯平滑，得到（特征词|类别）条件概率
    alpha = 1.0
    for w, fc in feature_counts.items():
        feature_counts[w] = (fc + alpha) / (total_feature_counts + n_features * alpha)

    return feature_counts, category_counts, categorys


def save_model(feature_prob, category_prior, categorys, modelFile):
    fw = open(modelFile, 'w')
    fw.write(json.dumps(categorys) + '\n')
    fw.write(json.dumps(category_prior.tolist()))
    for w, fc in feature_prob.items():
        fw.write('\n' + w + '\t')
        fw.write(json.dumps(fc.tolist()))
    fw.close()


def save_features(feature_words, featureFile):
    fw = open(featureFile, 'w')
    for term, score in feature_words.items():
        fw.write(term + '\t' + str(score) + '\n')
    fw.close()


def load_model(modelFile):
    fr = open(modelFile)
    categorys = eval(fr.readline())
    category_prior = eval(fr.readline())
    feature_prob = {}
    for line in fr:
        w, fc = line.strip().split('\t')
        feature_prob[w] = eval(fc)
    fr.close()
    return feature_prob, category_prior, categorys


def predict(feature_prob, category_prior, categorys, textdatas):
    rightCount = 0
    totalCount = 0
    for line in textdatas:
        true_cate, words = line.split('\t')
        pred_score = np.log(category_prior)
        for w in words.split(' '):
            if feature_prob.has_key(w):
                pred_score += np.log(feature_prob[w])
        pred_score = pred_score.tolist()
        pred_cate = categorys[pred_score.index(max(pred_score))]
        if pred_cate == true_cate:
            rightCount += 1
        totalCount += 1
        print totalCount

    print "Total test docs: ", totalCount
    print "Predict right docs: ", rightCount
    print "Precision: ", rightCount * 1.0 / totalCount

