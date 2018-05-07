## SELECTED, WORKING IMPLEMENTATION OF mbGD with ESE for AMC038
#  import packages
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import sys
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def getSigElecList(sigThresh):
    allElecScores = sio.loadmat(dataPath + allElecScoresFile)['scoreMat']
    idxSigElec = (allElecScores[:, 3] >= sigThresh) & (allElecScores[:, 4] == 0)
    sigElecList = allElecScores[idxSigElec, 0:3]
    return sigElecList


def buildSupergrid(sigElecList):
    nElec = sigElecList.shape[0]
    patientCode = 'AMC{:03g}'.format(sigElecList[0, 0])
    runCode = 'TheWall1_run{:g}'.format(sigElecList[0, 1])
    nSamp = sio.loadmat('{}{}/{}_{}_preprocessed.mat'.format(dataPath, patientCode, patientCode, runCode))['resp'].shape[0]
    supergrid = np.zeros((nSamp, nElec))
    for idxElec in range(nElec):
        patientCode = 'AMC{:03g}'.format(sigElecList[idxElec, 0])
        runCode = 'TheWall1_run{:g}'.format(sigElecList[idxElec, 1])
        if (idxElec == 0) or ((idxElec > 0) and (sigElecList[idxElec, 0] != sigElecList[idxElec-1, 0])):  # load datafile only if different patient, otherwise computationally unefficient-+-
            dataTMP = sio.loadmat('{}{}/{}_{}_preprocessed.mat'.format(dataPath, patientCode, patientCode, runCode))['resp']
        supergrid[:, idxElec] = dataTMP[:, int(sigElecList[idxElec, 2])-1]
    return supergrid


def preprocessing(modelType, stim, resp, badEpochs = np.nan, dataTime = np.nan):
    t = time.time()
    # load data
    if ~np.any(np.isnan(dataTime)):
        for i in range(badEpochs.shape[0]):
            resp[np.argmin(np.abs((dataTime - badEpochs[i, 0]))):np.argmin(np.abs((dataTime - badEpochs[i, 1]))), :] = np.nan
    if modelType == 'encoding':
        features = stim
        target = resp[:, targetYidx]
    else:
        features = resp
        target = stim[:, targetYidx]
    nSamp = features.shape[0] - nLags
    nFeatYs = features.shape[1]
    nFeat = nFeatYs * nLags
    # create lag matrix
    XLagMat = np.zeros((nSamp, nFeat))
    for i in range(nSamp):
        XLagMat[i, :] = np.reshape(features[i:i + nLags, :], (1, nFeatYs * nLags))
    if modelType == 'encoding':
        target = target[nLags:]
    else:
        target = target[:-nLags]
    XLagMat = XLagMat[~np.isnan(target), :]
    target = target[~np.isnan(target)]
    target = target[~np.isnan(XLagMat).any(axis=1)]
    XLagMat = XLagMat[~np.isnan(XLagMat).any(axis=1), :]
    features = XLagMat
    nSamp = features.shape[0]
    target = np.reshape(target, [nSamp, 1])
    # create groups
    nGroups = int(np.floor(nSamp / fs / groupL))
    idxArray = np.repeat(range(nGroups), fs * groupL)
    extraTimePoints = nSamp - len(idxArray)
    if extraTimePoints < groupMinL * fs:
        idxArray = np.concatenate((idxArray, np.repeat(idxArray[-1], extraTimePoints)))
    else:
        idxArray = np.concatenate((idxArray, np.repeat(idxArray[-1] + 1, extraTimePoints)))
    # groups = np.unique(idxArray)
    # create train, early stop and test sets
    nGroupsVal = int(np.round(valSize * nGroups))
    nGroupsTest = int(np.round(testSize * nGroups))
    nGroupsTrain = nGroups - (nGroupsVal + nGroupsTest)
    nGroupsSets = [nGroupsTrain, nGroupsVal, nGroupsTest]
    groups = np.arange(nGroups)
    print('Data import and preprocessing - {:s} model - time elapsed: {:.3f}'.format(modelType, time.time() - t))
    # sungPart = np.array([[14.9, 24.6, 33.9, 43.9, 56, 63.9, 73.6], [19.4, 29.5, 39, 49, 59.1, 69, 78.6]]) * fs
    # sungIdx = np.concatenate([np.arange(sungPart[0, i], sungPart[1, i]) for i in range(sungPart.shape[1])])
    return features, target, nFeatYs, nFeat, groups, idxArray, nGroupsSets


def splitNscale(idxFold):
    t = time.time()
    np.random.shuffle(groups)
    idxTrain = np.where(np.in1d(idxArray, groups[:nGroupsSets[0]]))[0]
    idxVal = np.where(np.in1d(idxArray, groups[nGroupsSets[0]:np.sum(nGroupsSets[:2])]))[0]
    idxTest = np.where(np.in1d(idxArray, groups[np.sum(nGroupsSets[:2]):]))[0]
    train_x = features[idxTrain]
    train_y = target[idxTrain]
    val_x = features[idxVal]
    val_y = target[idxVal]
    test_x = features[idxTest]
    test_y = target[idxTest]
    # Feature standardization
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)
    print('SplitNscale for resample #{:d} - time elapsed: {:.3f}'.format(idxFold, time.time() - t))
    return idxTrain, idxVal, idxTest, train_x, train_y, val_x, val_y, test_x, test_y


def fitModel():
    t = time.time()
    tf.reset_default_graph()
    if visualTrig == 1:
        if modelType == 'decoding':
            extentTuple = (0, nLags * 10, 1, nFeatYs+1)
        else:
            extentTuple = (-nLags * 10, 0, 1, nFeatYs+1)
        plt.ion()
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(131)
        STRF = ax1.imshow(np.zeros((nFeatYs, nLags)), aspect='auto', origin='lower', cmap='jet', extent=extentTuple)
        plt.xlabel('time lag (ms)')
        plt.ylabel('feature Y index')
        ax2 = fig.add_subplot(132)
        trainHistory, = ax2.plot([], [])
        plt.xlabel('# iterations')
        plt.ylabel('training loss')
        ax3 = fig.add_subplot(133)
        valHistory, = ax3.plot([], [])
        valFit, = ax3.plot([], [])
        plt.xlabel('# iterations')
        plt.ylabel('validation loss')
    train_loss_history = np.empty(shape=[0])
    val_loss_history = np.empty(shape=[0])
    best_weights = np.zeros([nFeat, 1])
    best_bias = 0
    best_loss = 0
    best_iteration = 0
    X = tf.placeholder(tf.float32, shape=(None, nFeat))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    W = tf.Variable(tf.zeros((nFeat, 1)))
    b = tf.Variable(tf.constant((train_y.mean(),)))
    # b = tf.Variable(tf.zeros((1,)))
    yPred = tf.add(tf.matmul(X, W), b)
    loss = tf.reduce_mean(tf.square(y_ - yPred))
    opt_operation = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)
    init = tf.global_variables_initializer()
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.49)
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:  # sess = tf.InteractiveSession()
        sess.run(init)
        counter = 0
        iteration = 0
        while counter < maxPatience:
            _, train_loss = sess.run([opt_operation, loss], feed_dict={X: train_x, y_: train_y})
            train_loss_history = np.append(train_loss_history, train_loss)
            val_loss = sess.run(loss, feed_dict={X: val_x, y_: val_y})
            val_loss_history = np.append(val_loss_history, val_loss)
            currentWeights = sess.run(W)
            if iteration == 0:
                best_loss = val_loss
            else:
                if val_loss_history[-1] >= val_loss_history[-2]:
                    counter += 1
                else:
                    counter = 0
                    best_weights = currentWeights
                    best_bias = sess.run(b)
                    best_loss = val_loss
                    best_iteration = iteration
            if iteration % 20 == 0:
                print('Iteration #{:d} - Validation loss = {:.3g} - Counter = {:d}'.format(iteration, val_loss, counter))
            if visualTrig == 1:
                if iteration % 20 == 0:
                    weights2d = np.reshape(currentWeights, [nLags, nFeatYs]).T
                    ax1.set_title('iteration #{:d}'.format(iteration))
                    STRF.set_data(weights2d)
                    trainHistory.set_data(range(len(train_loss_history)), train_loss_history)
                    valHistory.set_data(range(len(val_loss_history)), val_loss_history)
                    time.sleep(0.001)
                    STRF.set_clim(weights2d.min(), weights2d.max())
                    ax2.relim()
                    ax2.autoscale_view(True, True, True)
                    ax3.relim()
                    ax3.autoscale_view(True, True, True)
                    fig.canvas.draw()
            iteration += 1
            if iteration > maxIter:
                break
            if np.isnan(val_loss) == True:
                break
        print('Best validation loss: {:.3g} at iteration #{:d}'.format(best_loss, best_iteration))
        yPred = np.matmul(test_x, best_weights) + best_bias
        score = np.corrcoef(yPred, test_y, rowvar=False)[0, 1]
        print('Pearson\'s r computed on test set: {:.3g}'.format(score))
        print('Model definition and training - time elapsed: {:.3f}'.format(time.time() - t))
    weights2d = np.reshape(best_weights, [nLags, nFeatYs]).T
    if visualTrig == 1:
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.plot(val_loss_history)
        plt.plot(best_iteration, best_loss, 'r*')
        plt.legend(['validation error', 'best model'])
        plt.xlabel('number of iterations')
        plt.ylabel('loss')
        plt.title('best loss = {:.3g} after {:d} iterations - learn rate {:.1g} - patience {:d}\n'
                  'test pred acc = {:.3g} - {:.3f} sec elapsed'
                  .format(best_loss, best_iteration, learningRate, maxPatience, score, time.time() - t), loc='left')
        plt.subplot(122)
        margin = np.max(np.abs([weights2d.min(), weights2d.max()]))
        plt.imshow(weights2d, aspect='auto', origin='lower', cmap='jet', extent=extentTuple, clim=(-margin, margin))
        plt.colorbar()
        plt.xlabel('time lag (ms)')
        plt.ylabel('feature Y index')
        plt.show()
    return weights2d, score, yPred


# GridEngine interfacing
argin = sys.argv[1]
# argin = 'supergrid-16'
patientCode = argin.split('-')[0]

t1 = time.time()
nLags = 75
groupL = 3  # 15#10#3#1
groupMinL = 2  # 13#7.5#1.5#0.75
fs = 100
valSize = 0.2
testSize = 0.2

if patientCode == 'supergrid':
    # other SGE ARGIN imports
    targetYidx = int(argin.split('-')[1]) - 1
    nLags = 50
    fileSuffix = 'TheWall1'
    dataPath = '/home/knight/lbellier/DataWorkspace/_projects/PinkFloyd/'
    allElecScoresFile = 'PF_allElecScores.mat'
    stimFname = 'stimuli/thewall1_stim32.mat'
    sigThresh = 0.05
    modelType = 'decoding'
    targetYprefix = 'f'
    sigElecList = getSigElecList(sigThresh)
    supergrid = buildSupergrid(sigElecList)
    stim = sio.loadmat(dataPath + stimFname)['stim']
    nTargetYs = stim.shape[1]
    [features, target, nFeatYs, nFeat, groups, idxArray, nGroupsSets] = preprocessing(modelType, stim, supergrid)
else:
    fileSuffix = argin.split('-')[1]
    targetYidx = int(argin.split('-')[2]) - 1
    modelType = argin.split('-')[3]
    dataPath = '/home/knight/lbellier/DataWorkspace/_projects/PinkFloyd/{}/'.format(patientCode)
    dataFile = '{}_{}_preprocessed.mat'.format(patientCode, fileSuffix)
    fnameIn = dataPath + dataFile
    data = sio.loadmat(fnameIn)
    stim = data['stim']
    resp = data['resp']
    if modelType == 'decoding':
        targetYprefix = 'f'
        nTargetYs = 32
        nLags = 50
        sigThresh = 0.1
        allElecScoresFile = 'PF_allElecScores.mat'
        allElecScores = sio.loadmat(dataPath[:-7] + allElecScoresFile)['scoreMat']
        idxSigElec = (allElecScores[:, 0] == int(patientCode[3:])) & (allElecScores[:, 3] >= sigThresh) & (allElecScores[:, 4] == 0)
        sigElecList = allElecScores[idxSigElec, 2]-1
        resp = resp[:, sigElecList.astype('int')]
    else:
        targetYprefix = 'e'
        nTargetYs = sio.loadmat(fnameIn)['resp'].shape[1]
    badEpochs = data['patientInfo']['badEpochs'][0][0][0][int(fileSuffix[-1]) - 1]
    dataTime = data['patientInfo']['time'][0][0][0][0][0, :]
    [features, target, nFeatYs, nFeat, groups, idxArray, nGroupsSets] = preprocessing(modelType, stim, resp, badEpochs, dataTime)

nFolds = 50
if modelType == 'encoding':
    # ## AMC038 STRF best params
    learningRate = 0.0005
    # batchSize = 2048
else:
    ## AMC038 decoding best params - WIP
    # learningRate = 0.001
    # batchSize = 1024
    ## AMC009 decoding best params - WIP
    learningRate = 0.001  #0.0001 #0.0005
    # batchSize = 2048

maxPatience = 5
maxIter = 200
visualTrig = 0
saveWeights2d = np.zeros((nFolds, nFeatYs, nLags))
saveScore = np.zeros((nFolds))
saveIdxTest = []
saveYPred = []
idxFold = 0
nanCounter = 0
while idxFold < nFolds:
    print('Processing resample #{:g}/{:g}'.format(idxFold+1, nFolds))
    [idxTrain, idxVal, idxTest, train_x, train_y, val_x, val_y, test_x, test_y] = splitNscale(idxFold)
    [saveWeights2d[idxFold, :, :], saveScore[idxFold], yPred] = fitModel()  # (learningRate, batchSize, maxPatience, linearThresh, fitMemory, maxIter, visualTrig)
    if ~np.isnan(saveScore[idxFold]):
        saveIdxTest.append(idxTest), saveYPred.append(yPred[:,0])
        idxFold += 1
    else:
        nanCounter += 1
        print('Warning: nan score - model didn\'t converge')
# idxPlot = 18
# plt.plot(target[saveIdxTest[idxPlot]]), plt.plot(saveYPred[idxPlot])

score = np.mean(saveScore)
# weights2d = saveWeights2d.mean(axis=0) / saveWeights2d.std(axis=0, ddof=1)
# # margin = np.max(np.abs([weights2d.min(), weights2d.max()]))
# margin = 5
# plt.figure()
# plt.imshow(weights2d, aspect='auto', origin='lower', cmap='jet', interpolation='spline36',
#            extent=[0, nLags * 10, 1, nFeatYs], clim=(-margin, margin))
# plt.colorbar()
# plt.xlabel('time lag (ms)')
# plt.ylabel('electrode')
# plt.title('AMC009 e{:d} - TF_mbGDwithESE - {:d} resamples - batch size {:g}\n'
#           'learn rate {:.1g} - patience {:d} - linearThresh {:g} - fitMemory {:d}\n'
#           'score={:.3g}'.format(targetYidx+1, nFolds, batchSize, learningRate, maxPatience, linearThresh, fitMemory,
#                                 score))
# plt.show()
print('Total time elapsed: {:.3f} - prediction accuracy: {:.3f} - nanCounter: {:g}'.format(time.time() - t1, score, nanCounter))


# mdict = {'alphaList': alphaList, 'saveScoreAll': saveScoreAll, 'saveAlphaRs': saveAlphaRs, 'saveCoefRs': saveCoefRs,
#          'saveScoreRs': saveScoreRs}
# destname = '/home/knight/lbellier/DataWorkspace/_projects/PinkFloyd/{}/'.format(patientCode)
if modelType == 'encoding':
    fnameOut = '{}_{}_{}Mod_{}{}-{}folds'.format(patientCode, fileSuffix, modelType, targetYprefix, targetYidx+1, nFolds)
    sio.savemat(dataPath + fnameOut,
                mdict={'saveWeights2d': saveWeights2d, 'saveScore': saveScore, 'totalTime': time.time() - t1, 'nanCounter': nanCounter,
                       'valSize': valSize, 'testSize': testSize, 'learningRate': learningRate, 'maxPatience': maxPatience, 'maxIter': maxIter})
elif patientCode == 'supergrid':
    fnameOut = '{}/{}_{}_{}Mod_{}{}-{}folds'.format(patientCode, patientCode, fileSuffix, modelType, targetYprefix, targetYidx+1, nFolds)
    sio.savemat(dataPath + fnameOut,
            mdict={'saveWeights2d': saveWeights2d, 'saveScore': saveScore, 'saveIdxTest': saveIdxTest, 'saveYPred': saveYPred, 'totalTime': time.time() - t1, 'nanCounter': nanCounter,
                   'valSize': valSize, 'testSize': testSize, 'learningRate': learningRate, 'maxPatience': maxPatience, 'maxIter': maxIter})
else:
    fnameOut = '{}_{}_{}Mod_{}{}-{}folds'.format(patientCode, fileSuffix, modelType, targetYprefix, targetYidx + 1, nFolds)
    sio.savemat(dataPath + fnameOut,
                mdict={'saveWeights2d': saveWeights2d, 'saveScore': saveScore, 'saveIdxTest': saveIdxTest, 'saveYPred': saveYPred, 'totalTime': time.time() - t1, 'nanCounter': nanCounter,
                       'valSize': valSize, 'testSize': testSize, 'learningRate': learningRate, 'maxPatience': maxPatience, 'maxIter': maxIter})