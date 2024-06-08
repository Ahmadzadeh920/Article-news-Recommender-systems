from initial_run import *
data_size = (None, 1, 1, 40)  # Batch size x Img Channels x Height x Width
from theano import tensor as T
import theano
input_var = T.tensor4('input')
output_size = (1,32)  # We will run the example in mnist - 10 digits
target_var = T.row('targets')
import lasagne
theano.config.exception_verbosity = 'high'

net = {}
net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)
net['conv1'] = lasagne.layers.Conv2DLayer(net['data'], num_filters=6, filter_size=(1, 30))
net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=(1, 2))
net['conv2'] = lasagne.layers.Conv2DLayer(net['pool1'], num_filters=6, filter_size=(1, 5))
net['out'] = lasagne.layers.DenseLayer(net['conv2'], num_units=32,
                                           nonlinearity=lasagne.nonlinearities.softmax)
attention = lasagne.layers.DenseLayer(net['out'], num_units=32, nonlinearity=lasagne.nonlinearities.sigmoid)
model = lasagne.layers.ElemwiseMergeLayer((net['out'], attention), T.mul)
############
lr = 1e-2
weight_decay = 1e-5
prediction = lasagne.layers.get_output(model)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()
weightsl2 = lasagne.regularization.regularize_network_params(model , lasagne.regularization.l2)
loss += weight_decay * weightsl2
params = lasagne.layers.get_all_params(model , trainable=True)
updates = lasagne.updates.sgd(loss, params, learning_rate=lr)
train_fn = theano.function([input_var, target_var], loss, updates=updates)
test_prediction = lasagne.layers.get_output(model, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
val_fn = theano.function([input_var, target_var], test_loss)
get_preds = theano.function([input_var], test_prediction)
train_model = []
import numpy as np

train_model = []
for k in range(0,306):
    input_cnn=[[[[0.0 for w in range(336)] for e in range(1)]for r in range(1)]for t in range(40)]
    out_cnn=[[0.0 for m in range(31)]for n in range(40)]
    trans_x_train = np.array(x_train[k]).T
    out_cnn = np.array(y_train[k]).T
    for h in range(0, 40):
        input_cnn[h][0][0] = trans_x_train[h]
    train_model.append(train_fn(input_cnn,out_cnn)) # This is where the model gets updated

loss_tst_list = []
acc_tst_list = []
predict_tst_list = []
for j in range(0, 48):
    input_cnn = [[[[0.0 for w in range(0, 336)] for e in range(0, 1)] for r in range(0, 1)] for t in range(0, 40)]
    out_cnn = [[0.0 for m in range(0, 31)] for n in range(0, 40)]
    trans_x = np.array(x_tst[k]).T
    out_cnn = np.array(y_tst[k]).T
    for e in range(0, 40):
        input_cnn[e][0][0] = trans_x[e]
    loss_tst = val_fn(input_cnn, out_cnn)
    loss_tst_list.append(loss_tst)
    predict_tst = get_preds(input_cnn)  # This is where the model gets updated
    predict_tst_list.append(predict_tst)
