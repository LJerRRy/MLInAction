import operator
from math import log


# /**
#  * 决策树是一个预测模型，他代表的是对象属性与对象值之间的一种映射关系。
#  *
#  * Entropy = 系统的凌乱程度，使用算法ID3, C4.5和C5.0生成树算法使用熵。这一度量是基于信息学理论中熵的概念
#  * 本代码是根据ID3算法来构造决策树
#  *
#  * 决策点，是对几种可能方案的选择，即最后选择的最佳方案。如果决策属于多级决策，
#  * 则决策树的中间可以有多个决策点，以决策树根部的决策点为最终决策方案。通常用矩形框来表示
#  *
#  * 状态节点，代表备选方案的经济效果（期望值），通过各状态节点的经济效果的对比，按照一定的决策标准就可以选出最佳方案。
#  * 由状态节点引出的分支称为概率枝，概率枝的数目表示可能出现的自然状态数目每个分枝上要注明该状态出现的概率。通常用圆圈来表示
#  * 状态节点，比如给你1000万元来投资建厂，到底建大厂还是小厂，这里的大厂和小厂分别就是状态节点，以这两个节点开始的分支分别属于两种方案
#  *
#  *  结果节点，将每个方案在各种自然状态下取得的损益值标注于结果节点的右端。通常用三角形来表示
#  *
#  *  ID3算法以原始集合{S} S作为根节点开始。在算法的每次迭代中，它遍历集合{S} S的每个未使用的属性，并计算熵{H(S)}(或信息增益{IG （S）}）。
#  *  然后选择具有最小熵（或最大信息增益）值的属性。
#  *  然后，通过所选择的属性（例如年龄小于50，年龄在50和100之间，年龄大于100）来划分集合{S}以产生数据的子集。
#  *  算法继续在每个子集上递归，仅考虑之前从未选择的属性。
#  *  子集中的递归可以在以下情况下停止：
#  *  1.子集中的每个元素都属于同一个类（+或 - ），然后将该节点变成一个叶子并用例子的类
#  *  2.没有更多的属性要被选择，但是示例仍然不属于同一个类（一些是+，一些是 - ），然后节点变成一个叶子，并标记了最常见的类的例子在子集
#  *  3.在子集中没有示例，当没有发现父集合中的示例匹配所选属性的特定值时发生，例如如果没有age> = 100的示例。然后创建叶，并且标记为父集中的最常见的示例类。
#  *    在整个算法中，决策树被构造为每个非终端节点表示其上分割了数据的所选属性，并且终端节点表示该分支的最后子集的类标签。
#  * Created by Jerry on 2016/12/27.
#  */


# 类的标签，即指该类可能的取值
def createTree(dataSet, labels):
    classList = [ex[-1] for ex in dataSet]
    # 递归终止条件一： 所有样本都属于同一类，直接返回
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 递归终止条件二：只有一个类，选择列表中的多数来分类
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeature]
    myTree = {bestFeatLabel: {}}  # 最终要返回的决策树
    del (labels[bestFeature])  # 从标签中删除已经决策过的标签
    featureValues = [ex[bestFeature] for ex in dataSet]
    uniqueVals = set(featureValues)
    for value in uniqueVals:
        subLabel = labels[:]
        # 递归构造决策树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabel)
    return myTree


# 选择列表中的多数
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 将字典里的值排序
    sortedCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedCount[0][0]


# 选择最优划分的属性，根据ID3.0方法来选择，即计算各个属性的信息增益，取最大的信息增益
# 函数返回，属性的数组下标
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 总的属性个数
    baseEntropy = calcShannoEnt(dataSet)  # 信息熵
    bestFeature = -1  # 最大信息增益的属性的下标
    bestInfoGain = 0.0  # 最大信息增益
    for i in range(numFeatures):
        featureList = [ex[i] for ex in dataSet]  # 遍历属性i的所有取值（标签）
        uniqueVals = set(featureList)  # 去掉重复的取值
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannoEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        print(newEntropy)
        print(infoGain)
        # 求最大信息增益
        if infoGain > bestInfoGain:
            print(bestInfoGain)
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 按照给定特征划分数据集，输入参数为带划分数据集、划分数据集的特征（这里是数组下标）、需要返回的特征的值
# 注意要申明一个新的变量保存划分后的数据，不能改变参数里的dataSet，Python中参数如数组字典都是按引用调用
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 将该特征值抽取掉
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 计算信息熵
def calcShannoEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 计算属性值的“固定值”
def calcShannoEnt2(dataSet, featV):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[featV]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'],
               [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def readDataSet(fileName, splitStr):
    fr = open(fileName, encoding="utf-8")
    labels = fr.readline().strip().split(splitStr)
    dataSet = [inst.strip().split(splitStr) for inst in fr.readlines()]
    return dataSet, labels
