from read_file import *
import numpy as np
X_train_file = read_file('X_train.txt')
x_train_all= string_to_float(X_train_file)
x_train_all = x_train_all[0:len(x_train_all)-1]
Y_train_file = read_file('y_train -306_ 31_ 40.txt')
y_train_all= string_to_float(Y_train_file)
y_train_all =y_train_all[0:len(y_train_all)-1]

X_tst_file = read_file('X_tst.txt')
x_tst_all = string_to_float(X_tst_file)
x_tst_all = x_tst_all[0:len(x_tst_all)-1]
Y_tst_file = read_file('y_tst_ 48_31 _ 40.txt')
y_tst_all = string_to_float(Y_tst_file)
y_tst_all = y_tst_all[0:len(y_tst_all)-1]

x_train =[[[0 for i in range(40)] for j in range(336)] for k in range(306)]
for x in range(0, 306):
    for y in range(0,336):
        x_train[x][y] = x_train_all[x*336+y]

x_tst =[[[0 for i in range(40)] for j in range(336)] for k in range(48)]
for x in range(0, 48):
    for y in range(0,336):
        x_tst[x][y] = x_tst_all[x*336+y]

y_train = [[[0 for i in range(40)] for j in range(31)] for k in range(306)]
for x in range(0, 306):
    for y in range(0,31):
        y_train[x][y] = y_train_all[x*31+y]


y_tst = [[[0 for i in range(40)] for j in range(31)] for k in range(48)]

for x in range(0, 48):
    for y in range(0,31):
        y_train[x][y] = y_train_all[x*31+y]
