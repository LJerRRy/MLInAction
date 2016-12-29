from numpy import *
import operator
from os import listdir


# NumPy系统是Python的一种开源的数值计算扩展。
# 可用来存储和处理大型矩阵，比Python自身的嵌套列表（nested list structure)结构要高效的多
# （该结构也可以用来表示矩阵（matrix））

# inX输入向量，dataSet训练集， 标签向量labels，选择最近邻居的数目k

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # tile函数是位于模块numpy.lib.shape_base中，它的功能为重复某个数组
    # 计算已知类别数据中的点与当前点之间的距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # itemgetter(1)表示按照第二个关键字排序，然后此次排序为逆序
    # 注意在Python3中dict已经没有iteritems()方法了，应该用items，表示以列表方式返回字典中 的键值对
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 处理文本数据

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    # 创建返回的Numpy矩阵
    returnMat = zeros((numberOfLines, 3))
    index = 0
    classLabelVector = []
    for line in arrayOLines:
        line = line.strip()
        # spilt()返回一个list
        listFromLine = line.split('\t')
        returnMat[index:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    # min(0)取每列中最小的值，min(1)取每行最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    # 获取数据集的行数,即训练集个数
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    # 测试集的数目
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    # 文件中前numTestVecs个为测试集，剩下的为训练集
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        # print("the classifier came back with: %d, the real answer is: %d" %
        #       (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
            print("the classifier came back with: %d, the real answer is: %d" %
                  (classifierResult, datingLabels[i]))
    print("the total error rate is: %f\nthe error numbers is: %f " %
          (errorCount / float(numTestVecs), errorCount))


def classifyPerson():
    resultList = ['didn\'tLike', 'in small doses', 'in large doses']
    percentTats = float(input("percent of time spent playing video games?"))
    icecream = float(input("liters of ice cream consumed per year?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, icecream])
    classifierResult = classify0(((inArr - minVals) / ranges), normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])


def img2vector(filename):
    # 将32*32的文本数字转换为1*1024的向量
    returnVector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0, 32 * i + j] = int(lineStr[j])
    return returnVector


# 手写数字测试
# 算法执行效率不高，有1024个维度浮点数
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        # print("the classifier came back with: %d, the real answer is : %d" % (classNumStr, classifierResult))
        if classifierResult != classNumStr:
            errorCount += 1
            print("the classifier came back with: %d, the real answer is : %d" % (classifierResult, classNumStr))
            print(fileNameStr)
    print("the total error rate is: %f" % (errorCount / float(mTest)))
    print("the error number is: %d" % errorCount)


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
