#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author: houlisha
@file: fasttext_utils.py
@time: 2019/01/{DAY}
@desc:
"""
import fasttext

def train(input_file, output, label_prefix='__label__', lr=0.1, lr_update_rate=100,
          dim=100, ws=5, epoch=5, min_count=1, neg=5, word_ngrams=1, loss='softmax',
          bucket=0, minn=0, maxn=0, thread=12, t=0.0001, silent=1, encoding='utf-8',
          pretrained_vectors=""):
    '''
    :param input_file     			training file path (required)
    :param output         			output file path (required)
    :param label_prefix   			label prefix ['__label__']
    :param lr             			learning rate [0.1]
    :param lr_update_rate 			change the rate of updates for the learning rate [100]
    :param dim            			size of word vectors [100]
    :param ws             			size of the context window [5]
    :param epoch          			number of epochs [5]
    :param min_count      			minimal number of word occurences [1]
    :param neg            			number of negatives sampled [5]
    :param word_ngrams    			max length of word ngram [1]
    :param loss           			loss function {ns, hs, softmax} [softmax]
    :param bucket         			number of buckets [0]
    :param minn           			min length of char ngram [0]
    :param maxn           			max length of char ngram [0]
    :param thread         			number of threads [12]
    :param t              			sampling threshold [0.0001]
    :param silent         			disable the log output from the C++ extension [1]
    :param encoding       			specify input_file encoding [utf-8]
    :param pretrained_vectors		pretrained word vectors (.vec file) for supervised learning []
    :return: 
    '''
    classifier = fasttext.supervised(input_file, output, label_prefix=label_prefix, lr=lr, lr_update_rate=lr_update_rate,
          dim=dim, ws=ws, epoch=epoch, min_count=min_count, neg=neg, word_ngrams=word_ngrams, loss=loss,
          bucket=bucket, minn=minn, maxn=maxn, thread=thread, t=t, silent=silent, encoding=encoding,
          pretrained_vectors=pretrained_vectors)


def evaluation(eval_file, model_path, label_prefix='__label__', k=1):
    '''
    :param eval_file: evaluating file path(required)
    :param model_path: model file path(.bin file)(required)
    :param label_prefix: label prefix ['__label__']
    :param k: 
    :return: 
    '''
    classifier = fasttext.load_model(model_path, label_prefix=label_prefix)
    result = classifier.test(eval_file, k=k)
    return {'precision': result.precision, 'recall':result.recall, 'nexamples':result.nexamples}


def predict(texts, model_path, label_prefix, with_proba, k):
    '''
    :param texts: test sentences list(required)
    :param model_path: model file path(.bin file)(required)
    :param label_prefix: label prefix ['__label__']
    :param with_proba: get with probability or not
    :param k: get top k-best labels
    :return: 
    '''
    classifier = fasttext.load_model(model_path, label_prefix=label_prefix)
    if with_proba:
        labels = classifier.predict_proba(texts, k=k)
        return labels
    else:
        labels =classifier.predict(texts, k=k)
        return labels


def get_sent_vector():
    # $ ./fasttext print-sentence-vectors model.bin < text.txt
    pass


if __name__ == '__main__':
    pass