代码目的：
    实现基于liblinear的SVM实现文本分类算法，以及对未知文本类别进行预测。

主要函数调用方法：
    见testSVM.py代码，主要方法有：

    train_svm.train(): 输入分类器。输入的训练数据格式为list<string>，每行代表一篇分词后的文档。
    predict_svm.predict(): 预测单条文档的类别（分词后的文档）。
    predict_svm.batch_predict(): 预测批量文档的类别，格式同训练数据。