import trees
import treePlotter

# dataSet, labels = trees.createDataSet()
# shonnon = trees.calcShannoEnt(dataSet)
# print(dataSet)
# print(labels)
# print(shonnon)
#
# print(trees.calcShannoEnt(trees.splitDataSet(dataSet, 0, 1)) * 0.6)
# print(trees.chooseBestFeatureToSplit(dataSet))

# print(treePlotter.retrieveTree(1))
# print(treePlotter.retrieveTree(0))
# myTree = trees.createTree(dataSet, labels)
# print(myTree)
# print(treePlotter.getNumLeafs(myTree))
# print(treePlotter.getTreeDepth(myTree))

"""
treePlotter.createPlot(myTree)
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
print(lenses)
lensesTree = trees.createTree(lenses, lensesLabels)
print(lensesTree)
treePlotter.createPlot(lensesTree)
"""


dataSet, labels = trees.readDataSet("waterlem.txt", ',')
myTreeWater = trees.createTree(dataSet, labels)
print(myTreeWater)
treePlotter.createPlot(myTreeWater)
