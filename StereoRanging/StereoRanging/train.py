import os
import numpy as np
#import tensorflow
import yaml
from easydict import EasyDict as edict

path = os.path.dirname(__file__) + '/data/20170810/train1_CorrPoints.yaml'
file = open(path, 'r')
pointsDict = edict(yaml.load(file))

K1 = np.array(np.zeros((3, 3)))
K1[0, :] = pointsDict.K1.data[0:3]
K1[1, :] = pointsDict.K1.data[3:6]
K1[2, :] = pointsDict.K1.data[6:9]

K2 = np.array(np.zeros((3, 3)))
K2[0, :] = pointsDict.K2.data[0:3]
K2[1, :] = pointsDict.K2.data[3:6]
K2[2, :] = pointsDict.K2.data[6:9]

R = np.array(np.zeros((3, 3)))
R[0, :] = pointsDict.R.data[0:3]
R[1, :] = pointsDict.R.data[3:6]
R[2, :] = pointsDict.R.data[6:9]

t = np.array(pointsDict.t.data)

print(pointsDict.ImageCount)
