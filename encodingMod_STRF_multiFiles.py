# Importation of packages
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import time
import pickle
import sys


# GridEngine interfacing
argin = sys.argv[1]  # argin = 'AMC009-TheWall1_run1-56'
patientCode = argin.split('-')[0]
fileSuffix = argin.split('-')[1].split('/')
elec = int(argin.split('-')[2])


# Definition of variables
dataPath = '/home/knight/lbellier/DataWorkspace/_projects/PinkFloyd/{}/'.format(patientCode)
nLags = 75
fs = 100
nSplitsRs = 20
nSplitsCV = 5
splitRsTestSize = 0.2
splitCVtestSize = 0.25
groupLength = 3
groupMinLength = 1.5
alphaList = [0.0001, 0.01, 1, 3, 10, 30, 100, 1000, 100000]  # coarse
if len(argin.split('-'))>3:
    alphaList = [0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 0.7, 1, 3, 7, 10, 30, 70, 100, 300, 700, 1000, 3000, 10000, 100000, 1000000]  # fine
# alphaList = [0.0001, 0.01, 1, 10, 100, 10000, 1000000]
# alphaList = [0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 7, 10, 30, 70, 100, 300, 1000, 3000, 10000, 100000, 1000000]


# Loading of data
nFiles = len(fileSuffix)
Xblocks = list()
yBlocks = list()
for idxFile in range(nFiles):
    dataFile = '{}_{}_preprocessed.mat'.format(patientCode, fileSuffix[idxFile])
    data = sio.loadmat(dataPath + dataFile)
    if fileSuffix[idxFile] == 'ShortStories':
        for idxStory in range(4):
            Xblocks.append(np.array(data['stim'])[idxStory, 0])
            yBlocks.append(np.array(data['resp'])[idxStory, 0])
    else:
        Xblocks.append(np.array(data['stim']))
        yBlocks.append(np.array(data['resp']))


# Creation of the meta lag matrix using all provided files
saveXLagMat = list()
saveGroup = list()
saveY = list()
groupInit = 0
for idxBlock in range(len(Xblocks)):
    X = Xblocks[idxBlock]
    y = yBlocks[idxBlock]
    y = y[:, elec-1]
    nBins = X.shape[1]

    # X = X / np.nanstd(X)
    # y = y / np.nanstd(y)

    # Correction of possible mismatch between X and y lengths
    LX = X.shape[0]
    Ly = y.shape[0]
    if Ly < LX:
        X = X[0:Ly, :]
    elif Ly > LX:
        y = y[0:LX]
    L = X.shape[0] - nLags


    # Creation of the lag matrix
    XLagMat = np.zeros((L, nBins*nLags))
    for i in range(L):
        XLagMat[i, :] = np.reshape(X[i:i+nLags, :], (1, nBins*nLags))
    y = y[nLags:]
    L = XLagMat.shape[0]


    # Creation of groups
    nGroups = int(np.floor(L / fs / groupLength))
    groups = np.repeat(range(nGroups), 100*groupLength) + groupInit
    extraTimePoints = L - len(groups)
    if extraTimePoints < groupMinLength * fs:
        groups = np.concatenate((groups, np.repeat(groups[-1], extraTimePoints)))
    else:
        groups = np.concatenate((groups, np.repeat(groups[-1] + 1, extraTimePoints)))
    groupInit = groups.max() + 1

    # Artifact removal
    saveXLagMat.append(XLagMat[np.where(~np.isnan(y))[0], :])
    saveGroup.append(groups[np.where(~np.isnan(y))[0]])
    saveY.append(y[np.where(~np.isnan(y))[0]])

XLagMat = np.vstack(saveXLagMat)
groups = np.hstack(saveGroup)
y = np.hstack(saveY)


# Preparation of saving lists and split generators
saveScoreAll = list()
saveAlphaRs = list()
saveCoefRs = list()
saveScoreRs = list()
gssplitRs = GroupShuffleSplit(n_splits=nSplitsRs, test_size=splitRsTestSize)
gssplitCV = GroupShuffleSplit(n_splits=nSplitsCV, test_size=splitCVtestSize)
gssplitRsList = list(gssplitRs.split(XLagMat, y, groups))


## Core
tTotal = time.time()
for idxFold in range(nSplitsRs):
    tFold = time.time()
    print('Resampling #{} / {}'.format(idxFold + 1, nSplitsRs))

    idxTrain, idxTest = gssplitRsList[idxFold]
    rMatCV = np.zeros((nSplitsCV, len(alphaList)))
    for idxCV in range(nSplitsCV):
        tCV = time.time()
        print('   CV #{} / {}'.format(idxCV + 1, nSplitsCV))

        idxTrainCV, idxTestCV = list(gssplitCV.split(idxTrain, groups=groups[idxTrain]))[idxCV]
        idxTrainCV, idxTestCV = idxTrain[idxTrainCV], idxTrain[idxTestCV]

        alphaRescale = len(idxTrainCV)
        scaler = StandardScaler()
        scaler.fit(XLagMat[idxTrainCV, :])
        XtrainCV = scaler.transform(XLagMat[idxTrainCV, :])
        XtestCV = scaler.transform(XLagMat[idxTestCV, :])

        rListCV = list()
        for idxAlpha in range(len(alphaList)):
            tAlpha = time.time()
            mod = Ridge(alpha=alphaList[idxAlpha]*alphaRescale, fit_intercept=False)
            mod.fit(XtrainCV, y[idxTrainCV])
            yPred = mod.predict(XtestCV)
            r = np.corrcoef(yPred, y[idxTestCV])[0, 1]
            rListCV.append(r)
            print('      alpha{} = {:.2f} - r = {:.4f} - {:.2f} sec elapsed'.format(idxAlpha + 1, alphaList[idxAlpha], r , time.time() - tAlpha))

        rMatCV[idxCV, :] = rListCV
        print('   {:.2f} sec elapsed'.format(time.time() - tCV))

    idxBestAlpha = int(np.argmax(rMatCV.mean(axis=0)))
    bestAlpha = alphaList[idxBestAlpha]
    alphaRescale = len(idxTrain)
    scaler = StandardScaler()
    scaler.fit(XLagMat[idxTrain, :])
    Xtrain = scaler.transform(XLagMat[idxTrain, :])
    Xtest = scaler.transform(XLagMat[idxTest, :])

    bestMod = Ridge(alpha=bestAlpha*alphaRescale, fit_intercept=False)
    bestMod.fit(Xtrain, y[idxTrain])
    yPred = bestMod.predict(Xtest)
    r = np.corrcoef(yPred, y[idxTest])[0, 1]

    saveScoreAll.append(rMatCV)
    saveAlphaRs.append(bestMod.alpha / alphaRescale)
    saveCoefRs.append(bestMod.coef_)
    saveScoreRs.append(r)

    print('{:.2f} sec elapsed\n'.format(time.time() - tTotal))


## Save modeling results
mdict={'alphaList': alphaList, 'saveScoreAll': saveScoreAll, 'saveAlphaRs': saveAlphaRs, 'saveCoefRs': saveCoefRs, 'saveScoreRs': saveScoreRs}
destname = '/home/knight/lbellier/DataWorkspace/_projects/PinkFloyd/{}/'.format(patientCode)
TheWallRuns = [s for s in fileSuffix if 'TheWall1' in s]
nTheWallRuns = len(TheWallRuns)
if nTheWallRuns > 1:
    for idxRun in range(nTheWallRuns-1):
        fileSuffix[idxRun+1] = fileSuffix[idxRun+1][9:]
fileSuffixOut = '-'.join(fileSuffix)
if len(argin.split('-'))>3:
    fname = '{}_{}_encodingMod_e{}-{}Rs{}CVfine.pkl'.format(patientCode, fileSuffixOut, elec, nSplitsRs, nSplitsCV)
else:
    fname = '{}_{}_encodingMod_e{}-{}Rs{}CV.pkl'.format(patientCode, fileSuffixOut, elec, nSplitsRs, nSplitsCV)
fh = open(destname + fname, 'wb')
pickle.dump(mdict, fh)
fh.close()