代码目的：
    实现基于朴素贝叶斯的文本多类别分类器，和对未知文本的类别预测。

代码函数以及作用：
featureSelecter():对输入文本数据做特征选择，目前实现chi和tf两种方法。
    输入: textdatas    分词后的文本数据，list<string>，每行为一篇文档
          method       使用的特征选择方法，“chi”或者“tf”
    输出: dict类型的特征词典，key为特征词，value为特征值，按特征值从大到小排序

train():输入训练数据和特征词典，训练朴素贝叶斯模型。
    输入: textdatas   分词后的文本数据，list<string>，每行为一篇文档
          features    特征词典
    输出: feature_counts  dict类型，特征词对类别的概率
          category_counts   dict类型，每个类型的先验
          categorys     list类型，类标

save_model():存储训练模型
    输入: feature_prob    dict类型，特征词对类别的概率
          category_prior  dict类型，每个类别的先验
          categorys       list类型，类标
          modelFile       模型文件存储路径

save_features():存储特征词
    输入: feature_words   dict类型，特征词典
          featureFile     特征词文件存储路径

load_model():加载模型文件
    输入: modelFile   模型文件存储路径
    输出: feature_prob    dict类型，特征词对类别的概率
          category_prior  dict类型，每个类别的先验
          categorys       list类型，类标

predict():对未知文本预测类别
    输入: feature_prob    dict类型，特征词对类别的概率
          category_prior  dict类型，每个类别的先验
          categorys       list类型，类标
          textdatas     分词后的文本数据，list<string>，每行为一篇文档
    输出: 直接打印预测结果以及准确率
