import os
import numpy as np
import tensorflow as tf
import yaml
from easydict import EasyDict as edict
import time

# get data
file = open('train1_CorrPoints.yaml', 'r')
pointsDict = edict(yaml.load(file))

# get K1, K2, R, t
K1_np = np.array(np.zeros((3, 3)))
K1_np[0, :] = pointsDict.K1.data[0:3]
K1_np[1, :] = pointsDict.K1.data[3:6]
K1_np[2, :] = pointsDict.K1.data[6:9]

K2_np = np.array(np.zeros((3, 3)))
K2_np[0, :] = pointsDict.K2.data[0:3]
K2_np[1, :] = pointsDict.K2.data[3:6]
K2_np[2, :] = pointsDict.K2.data[6:9]

R_np = np.array(np.zeros((3, 3)))
R_np[0, :] = pointsDict.R.data[0:3]
R_np[1, :] = pointsDict.R.data[3:6]
R_np[2, :] = pointsDict.R.data[6:9]

t_np = np.array(np.zeros((3, 1)))
t_np[0, :] = pointsDict.t.data[0]
t_np[1, :] = pointsDict.t.data[1]
t_np[2, :] = pointsDict.t.data[2]

# get image count
imgCount = pointsDict.ImageCount

# get points count
corrPointsCount = pointsDict.CorrPointsCount

# get corrsponding point coordinates and ground truth
lPoints = []
rPoints = []
groundTruth = []
for i in range(1, imgCount + 1):
    lPoints.append(pointsDict['L' + str(i)])
    rPoints.append(pointsDict['R' + str(i)])
    groundTruth.append(pointsDict['GroundTruth' + str(i)])

# prepare training data
m_train = []
n_train = []
range_train = []
for img in range(imgCount):
    for (lPoint, rPoint) in zip(lPoints[img], rPoints[img]):
        temp_m = np.array([
            [-1, 0, lPoint[0], 0, 0, 0],
            [0, -1, lPoint[1], 0, 0, 0],
            [0, 0, 0, -1, 0, rPoint[0]],
            [0, 0, 0, 0, -1, rPoint[1]]
        ])
        temp_n = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, rPoint[0]],
            [0, 0, 0, 0, -1, rPoint[1]]
        ])
        m_train.append(temp_m)
        n_train.append(temp_n)
        range_train.append(groundTruth[img])

# basic elements
K1 = tf.get_variable("K1", shape=[3, 3], initializer=tf.constant_initializer(K1_np))
K2 = tf.get_variable("K2", shape=[3, 3], initializer=tf.constant_initializer(K2_np))
R = tf.get_variable("R", shape=[3, 3], initializer=tf.constant_initializer(R_np))
t = tf.get_variable("t", shape=[3, 1], initializer=tf.constant_initializer(t_np))

# projection
K1t = tf.Variable(tf.zeros([3, 1], dtype=tf.float32), trainable=False, dtype=tf.float32)
K2R = tf.matmul(K2, R, name="K2R")
K2t = tf.matmul(K2, t, name="K2t")

# model
# Ax=B (A = MP, B = NQ)
M = tf.placeholder(tf.float32, [None, 4, 6])
N = tf.placeholder(tf.float32, [None, 4, 6])
groundTruth = tf.placeholder(tf.float32, [None])

P = tf.concat([K1, K2R], 0, name="P")
Q = tf.concat([K1t, K2t], 0, name="Q")

# batch
total_loss = 0
for i in range(corrPointsCount):
    A = tf.matmul(M[i], P, name="A")
    B = tf.matmul(N[i], Q, name="B")

    # A_geninv = (A'A)^-1A'
    A_geninv = tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(A), A)), tf.transpose(A))
    X = tf.matmul(A_geninv, B)
    l_squr = tf.matmul(tf.transpose(X), X)
    RtX = tf.add(tf.matmul(R, X), t)
    r_squr = tf.matmul(tf.transpose(RtX), RtX)
    b_squr = tf.matmul(tf.transpose(t), t)
    h = 0.5 * tf.sqrt(2 * l_squr + 2 * r_squr - b_squr) / 1000

    # loss
    total_loss = total_loss + tf.squared_difference(h, groundTruth[i])

total_loss = total_loss / corrPointsCount

# params
learningRate = 0.011
iterations = 100000
previous_loss = 0
current_loss = 0

# initialization
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

# optimizer
train = tf.train.GradientDescentOptimizer(learningRate).minimize(total_loss)

# timing
tik = time.time()

# run session
for i in range(iterations):
    # pass previous loss
    previous_loss = current_loss

    # train
    current_train, current_loss = sess.run([train, total_loss],
                                           feed_dict={M: m_train, N: n_train, groundTruth: range_train})

    # print loss after several iters
    if (i + 1) % 20 == 0:
        # timing
        tok = time.time()
        time_past = tok - tik
        print("Iteration " + str(i + 1) + ": loss = " + str(current_loss) + ", " + str(time_past) + "s")
        # timing
        tik = time.time()

    # when loss is stable, break iteration
    if abs(current_loss - previous_loss) < 1e-6:
        print("Early termination: loss is stable.")
        break

print("Final result:")
print("K1 = ")
print(sess.run(K1))
print("K2 = ")
print(sess.run(K2))
print("R = ")
print(sess.run(R))
print("t = ")
print(sess.run(t))

sess.close()