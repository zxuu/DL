import numpy as np
import re
import random

def textParse(input_string):
    '''
    将文本数据转换成单词list
    :param input_string: 传进来的文本数据
    :return: [一个个单词]
    '''
    listofTokens = re.split(r'\W+',input_string)
    return [tok.lower() for tok in listofTokens if len(listofTokens)>2]

def creatVocablist(doclist):
    '''
    得到所有唯一的单词
    :param doclist: 传进来的邮件list
    :return: 返回所有邮件所有不一样的单词（语料表）
    '''
    vocabSet = set([])
    for document in doclist:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWord2Vec(vocablist, inputSet):
    '''

    :param vocablist: 语料表
    :param inputSet: 当前的训练邮件
    :return: 标记list，邮件中的单词出现在语料表中就标记为1
    '''
    returnVec = [0]*len(vocablist)
    #遍历邮件中的单词
    for word in inputSet:
        if word in vocablist:
            returnVec[vocablist.index(word)] = 1
    return returnVec


def trainNB(trainMat, trainClass):
    '''
    训练模型
    :param trainMat: 训练向量
    :param trainClass: 标签
    :return: 正常邮件单词的频率, 垃圾邮件单词的频率, 垃圾邮件的概率
    '''
    numTrainDocs = len(trainMat) #邮件个数
    numWords = len(trainMat[0]) #所有邮件的所有单词总数(语料表的长度)
    p1 = sum(trainClass)/float(numTrainDocs) #垃圾邮件的概率
    p0Num = np.ones((numWords)) #做了一个平滑处理，默认为1
    p1Num = np.ones((numWords)) #拉普拉斯平滑
    p0Denom = 2
    p1Denom = 2 #所有垃圾邮件的单词总数，通常情况下都是设置成类别个数

    #计算P(d1|h+)*P(d2|h+)*P(d3|h+)*P(d4|h+)***
    # 计算P(d1|h-)*P(d2|h-)*P(d3|h-)*P(d4|h-)***
    for i in range(numTrainDocs): #遍历邮件
        if trainClass[i] == 1: #垃圾邮件
            p1Num += trainMat[i] #统计垃圾邮件中的单词出现的次数
            p1Denom += sum(trainMat[i]) #分母是：多少个单词出现在语料库当中
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])

    #概率值较小，给放大。因为只需要比价大小，真实值不重要
    p1Vec = np.log(p1Num/p1Denom)
    p0Vec = np.log(p0Num/p0Denom)
    return p0Vec, p1Vec, p1


def classifyNB(wordVec, p0Vec, p1Vec, p1_class):
    '''
    分类模块
    :param wordVec: 测试邮件单词向量[0,1,0,1,......]
    :param p0Vec: 正常邮件的单词概率
    :param p1Vec: 垃圾邮件的单词概率
    :param p1_class: 垃圾邮件的概率
    :return:
    '''
    p1 = np.log(p1_class) + sum(wordVec*p1Vec)
    p0 = np.log(1 - p1_class) + sum(wordVec * p0Vec)
    if p0>p1:
        return 0 #是正常邮件
    else:
        return 1 #是垃圾邮件


def spam():
    doclist = []    #邮件list
    classlist = []   #标签list
    for i in range(1,26):    #一个一个遍历邮件
        wordlist = textParse(open('email/spam/%d.txt'%i,'r').read())
        doclist.append(wordlist)
        classlist.append(1) #1表示垃圾邮件

        wordlist = textParse(open('email/ham/%d.txt' % i, 'r').read())
        doclist.append(wordlist)
        classlist.append(0)  # 0表示正常邮件

    #得到语料表
    vocablist = creatVocablist(doclist)

    #数据集切分
    trainSet = list(range(50))    #训练集放40个邮件
    testSet = [] #测试机放10个邮件
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainSet)))
        testSet.append(trainSet[randIndex])
        del (trainSet[randIndex])

    #构建训练数据向量和标签向量
    trainMat = []
    trainClass = []
    for docIndex in trainSet:
        trainMat.append(setOfWord2Vec(vocablist, doclist[docIndex]))
        trainClass.append(classlist[docIndex])
    
    #开始训练
    p0Vec, p1Vec, p1 = trainNB(np.array(trainMat),np.array(trainClass))

    #测试模块
    errorCount = 0
    for docIndex in testSet:
        wordVec = setOfWord2Vec(vocablist,doclist[docIndex])
        #分类模块
        if classifyNB(np.array(wordVec), p0Vec, p1Vec, p1) != classlist[docIndex]:
            errorCount += 1
    print('当前10个测试样本，错了：',errorCount)


    if __name__ == '__main__':
        spam()

