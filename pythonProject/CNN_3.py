# def CNN(number_data_train,number_data_tst,x_train,y_train,x_tst,y_tst):
def CNN_3(x_train, y_train, x_tst, y_tst):
    data_size = (None, 1, 1, 40)  # Batch size x Img Channels x Height x Width
    from theano import tensor as T
    import theano
    input_var = T.tensor4('input')
    output_size = (1, 32)  # We will run the example in mnist - 10 digits
    target_var = T.row('targets')
    import lasagne
    theano.config.exception_verbosity = 'high'
    theano.config.optimizer = "fast_compile"

    net = {}
    net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)
    net['conv1'] = lasagne.layers.Conv2DLayer(net['data'], num_filters=6, filter_size=(1, 30))
    net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=(1, 2))
    net['out'] = lasagne.layers.DenseLayer(net['pool1'], num_units=32,
                                           nonlinearity=lasagne.nonlinearities.softmax)
    lr = 1e-2
    weight_decay = 1e-5
    prediction = lasagne.layers.get_output(net['out'])
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    weightsl2 = lasagne.regularization.regularize_network_params(net['out'], lasagne.regularization.l2)
    loss += weight_decay * weightsl2
    params = lasagne.layers.get_all_params(net['out'], trainable=True)
    updates = lasagne.updates.sgd(loss, params, learning_rate=lr)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    test_prediction = lasagne.layers.get_output(net['out'], deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
    val_fn = theano.function([input_var, target_var], test_loss)
    get_preds = theano.function([input_var], test_prediction)
    train_model = []
    import numpy as np
    for batch in range(0, len(x_train)):
        x_batch = [[[]]]
        x_batch[0][0].append(x_train[batch])
        x_batch = np.array(x_batch)
        y_batch = []
        y_batch.append(y_train[batch])
        y_batch = np.array(y_batch)
        train_model.append(train_fn(x_batch, y_batch))  # This is where the model gets updated
    loss_tst_list = []
    acc_tst_list = []
    predict_tst_list = []
    for j in range(0, len(x_tst)):
        x_batch_tst = [[[]]]
        x_batch_tst[0][0].append(x_tst[j])
        x_batch_tst = np.array(x_batch_tst)
        y_batch_tst = []
        y_batch_tst.append(y_tst[j])
        loss_tst = val_fn(x_batch_tst, y_batch_tst)
        loss_tst_list.append(loss_tst)
        "acc_tst_list.append(acc_tst)"""
        predict_tst = get_preds(x_batch_tst)  # This is where the model gets updated
        predict_tst_list.append(predict_tst)

    return train_model, loss_tst_list, predict_tst_list
