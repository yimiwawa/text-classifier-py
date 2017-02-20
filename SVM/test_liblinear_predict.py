#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")

from classifier import predict_svm

# 训练文本所在的文件名,每行为一个文本，每个文本的字段之间用'\t'间隔
filename = "D:\\GitLab\\xh-nlp\\test\\liblinear_model\\multi.test"
#
is_seg = True

texts = file(filename, 'r').readlines()
right = 0
wrong = 0
for text in texts:
    terms = text.strip().split("\t")
    label, score = predict_svm.predict(terms[1], is_seg)
    print "src:" + terms[0] + "\t" + "pre:" + str(label) + "\t" + "score:" + str(score)
    if float(terms[0]) == label:
        right += 1
    else:
        wrong += 1
print "right:" + str(right)
print "wrong:" + str(wrong)


