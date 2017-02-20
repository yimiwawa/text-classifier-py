#! -*- coding:utf-8 -*-

# __author__ = "houlisha"

import random
import multiClassifier as nb

trainFile = "../data/mostClass.train.seg"
featureFile = "../data/mostClass.features"
modelFile = "../data/mostClass.model"


if __name__ == "__main__":
    all_datas = [line.strip() for line in open(trainFile).readlines()]
    random.shuffle(all_datas)
    train_datas = all_datas[:3 * len(all_datas) / 5]
    test_datas = all_datas[3 * len(all_datas) / 5:]

    print "特征选择..."
    feature_words = nb.featureSelecter(train_datas, "chi")
    print "共选择特征数量:", len(feature_words)

    nb.save_features(feature_words, featureFile)

    feature_prob, category_prior, categorys = nb.train(train_datas, feature_words)

    nb.save_model(feature_prob, category_prior, categorys, modelFile)
    feature_prob, category_prior, categorys = nb.load_model(modelFile)

    nb.predict(feature_prob, category_prior, categorys, test_datas)