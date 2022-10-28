import numpy as np
import tensorflow as tf
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True
from six.moves import cPickle as pickle
from six.moves import range

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0]) 

def performance(pred, labels, regCost):
    Acc = (100.0 * np.sum(np.argmax(pred, 1) == np.argmax(labels, 1)) / pred.shape[0]) 
    Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred) +
            tf.nn.l2_loss(regCost)).eval()
    return Acc,Loss

def reformat(dataset, labels, num_feats, num_labels):
    dataset = dataset.reshape((-1, num_feats)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = np.squeeze((np.arange(num_labels) == labels[:,None]).astype(np.float32))
    # labels = np.squeeze((np.arange(num_labels) == labels).astype(np.float32))
    return dataset, labels

def buildFNmodel3layers(Data,Labels,filename,num_steps=50000,numHiddenNodes = [5, 4, 3],regCost = 0.5):
    
    N = Labels.size
    
    if 0:
        HH = np.vstack((1-Data[:,1].copy(),Data[:,1].copy())).transpose().astype(np.float32)
        HH[HH==0.0] = 0.000001
        HH[HH==1.0] = 0.999999
        HH = np.log(HH)

    Means = np.mean(Data,axis=0)
    Std = np.std(Data,axis=0)
    
    Data = (Data - Means) / Std

    np.random.seed(42)
    isTestSet = np.random.uniform(size=N) > 0.8
    test_dataset = Data[isTestSet,:]
    test_labels = Labels[isTestSet]
    train_dataset = Data[isTestSet==False,:]
    train_labels = Labels[isTestSet==False]

    num_feats = train_dataset.shape[1]
    num_labels = 2

    train_dataset, train_labels = reformat(train_dataset, train_labels, num_feats, num_labels)

    test_dataset, test_labels = reformat(test_dataset, test_labels, num_feats, num_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    batch_size = 25
    
    learningRate = 0.0002
    dropoutRate = 0.75
    num_steps += 1


    graph = tf.Graph()
    with graph.as_default():    

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_feats))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_test_dataset = tf.constant(test_dataset)
        tf_valid_dataset = tf.constant(train_dataset)
   
        # Variables.
        weights = tf.Variable(
                tf.truncated_normal([num_feats, numHiddenNodes[0]]))
        biases = tf.Variable(tf.zeros([numHiddenNodes[0]]))

        weights2 = tf.Variable(
                tf.truncated_normal([numHiddenNodes[0], numHiddenNodes[1]]))
        biases2 = tf.Variable(tf.zeros([numHiddenNodes[1]]))
    
        weights3 = tf.Variable(
                tf.truncated_normal([numHiddenNodes[1], numHiddenNodes[2]]))
        biases3 = tf.Variable(tf.zeros([numHiddenNodes[2]]))
    
        weights4 = tf.Variable(
                tf.truncated_normal([numHiddenNodes[2], num_labels]))
        biases4 = tf.Variable(tf.zeros([num_labels]))    
      
        # Training computation.
        logits = tf.matmul(tf.nn.dropout(tf.nn.relu(tf.matmul(tf.nn.dropout(tf.nn.relu(tf.matmul(tf.nn.dropout(tf.nn.relu(tf.matmul(tf_train_dataset, weights) + biases),
                                                dropoutRate),weights2) + biases2), dropoutRate),weights3) + biases3), dropoutRate),weights4) + biases4
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits) +
                tf.nn.l2_loss(regCost))
      
        # Optimizer.
        #optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
        #optimizer = tf.train.RMSPropOptimizer(learningRate,decayRate).minimize(loss)  
        optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)  
    
        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)

        valid_probs = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights) + biases),
            weights2) + biases2), weights3) + biases3), weights4) + biases4)
        test_probs = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights) + biases),
                                                weights2) + biases2), weights3) + biases3), weights4) + biases4)

    # now run the optimization    

    with tf.Session(graph=graph,config=config) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run(
              [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 1000 == 0):
                trainAcc,trainLoss = performance(valid_probs.eval(), train_labels, regCost)
                testAcc,testLoss = performance(test_probs.eval(), test_labels, regCost)
                
                #acc,lss = performance(HH, np.squeeze((np.arange(num_labels) == Labels[:,None]).astype(np.float32)), regCost)

                print('{:d}: Loss training/test: {:.3f} / {:.3f}'.format(step,trainLoss,testLoss))
                print('{:d}: Accuracy tra/test: {:.1f} / {:.1f}'.format(step,trainAcc,testAcc))

         
        Weights = weights.eval()
        Weights2 = weights2.eval()
        Weights3 = weights3.eval()
        Weights4 = weights4.eval()
        Biases = biases.eval()
        Biases2 = biases2.eval()
        Biases3 = biases3.eval()
        Biases4 = biases4.eval()
        Model = {
            "Weights": Weights,
            "Weights2": Weights2,
            "Weights3": Weights3,
            "Weights4": Weights4,
            "Biases":Biases,
            "Biases2":Biases2,
            "Biases3":Biases3,
            "Biases4":Biases4,
            "Means":Means,
            "Std":Std
            }
        f = open(filename+'.npz', "wb")
        pickle.dump(Model, f)
        f.close()
        #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
        trainingProbs = valid_probs.eval()[:,1]  
        testProbs = test_probs.eval()[:,1]
        #testProbsAll = test_prediction.eval()
  
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2)
    nBins = 40
    probBins = np.linspace(0,1,nBins)
    Pd = np.zeros(len(probBins))
    Pfa = np.zeros(len(probBins))
    PdTr = np.zeros(len(probBins))
    PfaTr = np.zeros(len(probBins))
    Inds1Tr = np.where(train_labels[:,1]==1)
    Inds0Tr = np.where(train_labels[:,0]==1)
    Inds1 = np.where(test_labels[:,1]==1)
    Inds0 = np.where(test_labels[:,0]==1)
    for j,p in enumerate(probBins):
        Pfa[j] = 100*np.sum(testProbs[Inds0] > p)/len(Inds0[0])
        Pd[j] = 100*np.sum(testProbs[Inds1] > p)/len(Inds1[0])
        PfaTr[j] = 100*np.sum(trainingProbs[Inds0Tr] > p)/len(Inds0Tr[0])
        PdTr[j] = 100*np.sum(trainingProbs[Inds1Tr] > p)/len(Inds1Tr[0])    
    
    ax[0].plot(Pfa, Pd,'b',label='Test set')
    ax[0].plot(PfaTr,PdTr,'r:',label='Training set')

    ax[0].legend(loc="lower right")
    ax[0].set_title(filename)

    ax[0].set_ylabel("Pd")
    ax[0].set_xlabel("Pfa")
    ax[0].set_xlim([0 ,100])
    ax[0].set_ylim([0 ,100])

    H1 = np.histogram(testProbs[Inds1],bins=probBins)[0]
    H2 = np.histogram(testProbs[Inds0],bins=probBins)[0]
    errMidBins = 0.5 * (probBins[0:(nBins-1)] + probBins[1:nBins]) 
    ax[1].plot(errMidBins,H1)
    ax[1].plot(errMidBins,H2)
    ax2 = ax[1].twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(errMidBins,H1/(H1 + H2),color='tab:green')
    #ax[1].plot(probBins, Pd,'b',label='Test set')

    plt.show()

def buildFNmodel1layer(Data,Labels,filename,num_steps=50000,numHiddenNodes = [3],regCost = 0.5):
    N = Labels.size
    
    Means = np.mean(Data,axis=0)
    Std = np.std(Data,axis=0)
    
    Data = (Data - Means) / Std

    np.random.seed(42)
    isTestSet = np.random.uniform(size=N) > 0.8
    test_dataset = Data[isTestSet,:]
    test_labels = Labels[isTestSet]
    train_dataset = Data[isTestSet==False,:]
    train_labels = Labels[isTestSet==False]

    num_feats = train_dataset.shape[1]
    num_labels = 2

    train_dataset, train_labels = reformat(train_dataset, train_labels, num_feats, num_labels)

    test_dataset, test_labels = reformat(test_dataset, test_labels, num_feats, num_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    batch_size = 25
  
    learningRate = 0.0002
    dropoutRate = 0.75
    num_steps += 1


    graph = tf.Graph()
    with graph.as_default():    

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_feats))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_test_dataset = tf.constant(test_dataset)
        tf_valid_dataset = tf.constant(train_dataset)
   
        # Variables.
        weights = tf.Variable(
                tf.truncated_normal([num_feats, numHiddenNodes[0]]))
        biases = tf.Variable(tf.zeros([numHiddenNodes[0]]))
 
        weights2 = tf.Variable(
                tf.truncated_normal([numHiddenNodes[0], num_labels]))
        biases2 = tf.Variable(tf.zeros([num_labels]))    
      
        # Training computation.
        logits = tf.matmul(tf.nn.dropout(tf.nn.relu(tf.matmul(tf_train_dataset, weights) + biases),dropoutRate),weights2) + biases2
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits) +
                tf.nn.l2_loss(regCost))
      
        # Optimizer.
        optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)  
    
        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)

        tf_valid_dataset

        valid_probs = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights) + biases),weights2) + biases2)
        test_probs = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights) + biases),weights2) + biases2)

    # now run the optimization    

    with tf.Session(graph=graph,config=config) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run(
              [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 1000 == 0):
                trainAcc,trainLoss = performance(valid_probs.eval(), train_labels, regCost)
                testAcc,testLoss = performance(test_probs.eval(), test_labels, regCost)
                print('{:d}: Loss training/test: {:.3f} / {:.3f}'.format(step,trainLoss,testLoss))
                print('{:d}: Accuracy tra/test: {:.1f} / {:.1f}'.format(step,trainAcc,testAcc))

         
        Weights = weights.eval()
        Weights2 = weights2.eval()
        Biases = biases.eval()
        Biases2 = biases2.eval()
        Model = {
            "Weights": Weights,
            "Weights2": Weights2,
            "Biases":Biases,
            "Biases2":Biases2,
            "Means":Means,
            "Std":Std
            }
        f = open(filename+'.npz', "wb")
        pickle.dump(Model, f)
        f.close()
        #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
        trainingProbs = valid_probs.eval()[:,1]  
        testProbs = test_probs.eval()[:,1]
        #testProbsAll = test_prediction.eval()
  
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2)
    nBins = 40
    probBins = np.linspace(0,1,nBins)
    Pd = np.zeros(len(probBins))
    Pfa = np.zeros(len(probBins))
    PdTr = np.zeros(len(probBins))
    PfaTr = np.zeros(len(probBins))
    Inds1Tr = np.where(train_labels[:,1]==1)
    Inds0Tr = np.where(train_labels[:,0]==1)
    Inds1 = np.where(test_labels[:,1]==1)
    Inds0 = np.where(test_labels[:,0]==1)
    for j,p in enumerate(probBins):
        Pfa[j] = 100*np.sum(testProbs[Inds0] > p)/len(Inds0[0])
        Pd[j] = 100*np.sum(testProbs[Inds1] > p)/len(Inds1[0])
        PfaTr[j] = 100*np.sum(trainingProbs[Inds0Tr] > p)/len(Inds0Tr[0])
        PdTr[j] = 100*np.sum(trainingProbs[Inds1Tr] > p)/len(Inds1Tr[0])    
    
    ax[0].plot(Pfa, Pd,'b',label='Test set')
    ax[0].plot(PfaTr,PdTr,'r:',label='Training set')

    ax[0].legend(loc="lower right")
    ax[0].set_title(filename)

    ax[0].set_ylabel("Pd")
    ax[0].set_xlabel("Pfa")
    ax[0].set_xlim([0 ,100])
    ax[0].set_ylim([0 ,100])

    H1 = np.histogram(testProbs[Inds1],bins=probBins)[0]
    H2 = np.histogram(testProbs[Inds0],bins=probBins)[0]
    errMidBins = 0.5 * (probBins[0:(nBins-1)] + probBins[1:nBins]) 
    ax[1].plot(errMidBins,H1)
    ax[1].plot(errMidBins,H2)
    ax2 = ax[1].twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(errMidBins,H1/(H1 + H2),color='tab:green')
    #ax[1].plot(probBins, Pd,'b',label='Test set')

    plt.show()

def buildFNmodel0layers(Data,Labels,filename,num_steps=50000,regCost = 0.5):
    N = Labels.size
    
    Means = np.mean(Data,axis=0)
    Std = np.std(Data,axis=0)
    
    Data = (Data - Means) / Std

    np.random.seed(42)
    isTestSet = np.random.uniform(size=N) > 0.8
    test_dataset = Data[isTestSet,:]
    test_labels = Labels[isTestSet]
    train_dataset = Data[isTestSet==False,:]
    train_labels = Labels[isTestSet==False]

    num_feats = train_dataset.shape[1]
    num_labels = 2

    train_dataset, train_labels = reformat(train_dataset, train_labels, num_feats, num_labels)

    test_dataset, test_labels = reformat(test_dataset, test_labels, num_feats, num_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    batch_size = 25
   
    learningRate = 0.0002
    dropoutRate = 0.75
    num_steps += 1


    graph = tf.Graph()
    with graph.as_default():    

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_feats))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_test_dataset = tf.constant(test_dataset)
        tf_valid_dataset = tf.constant(train_dataset)
   
        # Variables.
        weights = tf.Variable(
                tf.truncated_normal([num_feats, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))
      
        # Training computation.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits) +
                tf.nn.l2_loss(regCost))
      
        # Optimizer.
        optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)  
    
        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)

        tf_valid_dataset

        valid_probs = tf.matmul(tf_valid_dataset, weights) + biases
        test_probs = tf.matmul(tf_test_dataset, weights) + biases

    # now run the optimization    

    with tf.Session(graph=graph,config=config) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run(
              [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 1000 == 0):
                trainAcc,trainLoss = performance(valid_probs.eval(), train_labels, regCost)
                testAcc,testLoss = performance(test_probs.eval(), test_labels, regCost)
                print('{:d}: Loss training/test: {:.3f} / {:.3f}'.format(step,trainLoss,testLoss))
                print('{:d}: Accuracy tra/test: {:.1f} / {:.1f}'.format(step,trainAcc,testAcc))

         
        Weights = weights.eval()
        Biases = biases.eval()
        Model = {
            "Weights": Weights,
            "Biases":Biases,
            "Means":Means,
            "Std":Std
            }
        f = open(filename+'.npz', "wb")
        pickle.dump(Model, f)
        f.close()
        #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
        trainingProbs = valid_probs.eval()[:,1]  
        testProbs = test_probs.eval()[:,1]
        #testProbsAll = test_prediction.eval()
  
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3)
    nBins = 40
    probBins = np.linspace(0,1,nBins)
    Pd = np.zeros(len(probBins))
    Pfa = np.zeros(len(probBins))
    PdTr = np.zeros(len(probBins))
    PfaTr = np.zeros(len(probBins))
    Inds1Tr = np.where(train_labels[:,1]==1)
    Inds0Tr = np.where(train_labels[:,0]==1)
    Inds1 = np.where(test_labels[:,1]==1)
    Inds0 = np.where(test_labels[:,0]==1)
    for j,p in enumerate(probBins):
        Pfa[j] = 100*np.sum(testProbs[Inds0] > p)/len(Inds0[0])
        Pd[j] = 100*np.sum(testProbs[Inds1] > p)/len(Inds1[0])
        PfaTr[j] = 100*np.sum(trainingProbs[Inds0Tr] > p)/len(Inds0Tr[0])
        PdTr[j] = 100*np.sum(trainingProbs[Inds1Tr] > p)/len(Inds1Tr[0])    
    
    ax[0].plot(Pfa, Pd,'b',label='Test set')
    ax[0].plot(PfaTr,PdTr,'r:',label='Training set')

    ax[0].legend(loc="lower right")
    ax[0].set_title(filename)

    ax[0].set_ylabel("Pd")
    ax[0].set_xlabel("Pfa")
    ax[0].set_xlim([0 ,100])
    ax[0].set_ylim([0 ,100])

    H1 = np.histogram(testProbs[Inds1],bins=probBins)[0]
    H2 = np.histogram(testProbs[Inds0],bins=probBins)[0]
    errMidBins = 0.5 * (probBins[0:(nBins-1)] + probBins[1:nBins]) 
    ax[1].plot(errMidBins,H1)
    ax[1].plot(errMidBins,H2)
    ax2 = ax[1].twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(errMidBins,H1/(H1 + H2),color='tab:green')
    #ax[1].plot(probBins, Pd,'b',label='Test set')




    plt.show()







