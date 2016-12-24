import kNN
import importlib
import matplotlib.pyplot as plt
from numpy import array
from os import listdir

group, labels = kNN.createDataSet()
kNN.classify0([0, 0], group, labels, 3)
importlib.reload(kNN)
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0*array(datingLabels))
# plt.show()
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
print(normMat)

kNN.datingClassTest()

# kNN.classifyPerson()

kNN.handwritingClassTest()
