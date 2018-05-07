#### Import packages ####
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import time
import pickle
import sys


#### GridEngine interfacing ####
argin = sys.argv[1]  # argin = 'AMC037-TheWall1_run1-120'
patientCode = argin.split('-')[0]
fileSuffix = argin.split('-')[1]
freqBin = int(argin.split('-')[2])


#### User-define variables ####
nLags = 40
fs = 100  # sampling rate (Hz)
nSplitsRs = 20
nSplitsCV = 5
splitRsTestSize = 0.2
splitCVtestSize = 0.25
groupLength = 3
groupMinLength = 1.5
threshSigElec = 0.1
# alphaCenter = 10
# alphaList = [0.00001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 10000000]
alphaList = [0.0001, 0.01, 1, 3, 10, 30, 100, 1000, 100000]  # coarse
dataPath = '/home/knight/lbellier/DataWorkspace/_projects/PinkFloyd/{}/'.format(patientCode)
dataFile = '{}_{}_preprocessed.mat'.format(patientCode, fileSuffix)


#### Load data ####
data = sio.loadmat(dataPath + dataFile)
X = np.array(data['resp'])
idxNan = np.where(np.isnan(X))[0]  # nan correction
idxOK = np.where(~np.isnan(X))[0]  # nan correction
X[idxNan, :] = X[idxOK, :].mean(axis=0)  # nan correction
# y = np.array(data['stim'])
# y = y[:, freqBin - 1]
del data

stim = sio.loadmat('/home/knight/lbellier/DataWorkspace/_projects/PinkFloyd/stimuli/thewall1_stim128.mat')
y = np.array(stim['stim'])
y = y[:, freqBin - 1]


### Load scores ###
scores = sio.loadmat(dataPath + '{}_{}_encodingMod_{}Rs{}CV_scores.mat'.format(patientCode, fileSuffix, nSplitsRs, nSplitsCV))
scores = scores['score'][0, :]
idxElecSig = np.where(scores > threshSigElec)
X = X[:, idxElecSig[0]]
nElecs = X.shape[1]


#### Correct possible X and y lengths mismatch ####
LX = X.shape[0]
Ly = y.shape[0]
if Ly < LX:
    X = X[0:Ly, :]
elif Ly > LX:
    y = y[0:LX]
L = X.shape[0] - nLags


#### Assemble lag matrix ####
XLagMat = np.zeros((L, nElecs*nLags))
for i in range(L):
    XLagMat[i, :] = np.reshape(X[i:i+nLags, :], (1, nElecs*nLags))
y = y[:L]
L = XLagMat.shape[0]


#### Define groups ####
nGroups = int(np.floor(L / fs / groupLength))
groups = np.repeat(range(nGroups), 100*groupLength)
extraTimePoints = L - len(groups)
if extraTimePoints < groupMinLength * fs:
    groups = np.concatenate((groups, np.repeat(groups[-1], extraTimePoints)))
else:
    groups = np.concatenate((groups, np.repeat(groups[-1] + 1, extraTimePoints)))

y = y[~np.isnan(XLagMat).any(axis=1)]
groups = groups[~np.isnan(XLagMat).any(axis=1)]
XLagMat = XLagMat[~np.isnan(XLagMat).any(axis=1), :]


#### Prepare saving variables and split generators ####
saveScoreAll = list()
saveAlphaRs = list()
saveCoefRs = list()
saveScoreRs = list()
saveYPredRs = list()
saveIdxTest = list()
gssplitRs = GroupShuffleSplit(n_splits=nSplitsRs, test_size=splitRsTestSize, random_state=7)
gssplitCV = GroupShuffleSplit(n_splits=nSplitsCV, test_size=splitCVtestSize)
gssplitRsList = list(gssplitRs.split(XLagMat, y, groups))


#### Core ####
tTotal = time.time()
for idxRs in range(nSplitsRs):
    tRs = time.time()
    print('Resampling #{} / {}'.format(idxRs + 1, nSplitsRs))

    idxTrain, idxTest = gssplitRsList[idxRs]
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
        for idxAlphaCV in range(len(alphaList)):
            tAlpha = time.time()
            mod = Ridge(alpha=alphaList[idxAlphaCV]*alphaRescale, fit_intercept=False)
            mod.fit(XtrainCV, y[idxTrainCV])
            yPred = mod.predict(XtestCV)
            r = np.corrcoef(yPred, y[idxTestCV])[0, 1]
            rListCV.append(r)
            print('      alpha{} = {:.2f} - r = {:.4f} - {:.2f} sec elapsed'.format(idxAlphaCV + 1, alphaList[
                idxAlphaCV], r, time.time() - tAlpha))

        rMatCV[idxCV, :] = rListCV
        print('   {:.2f} sec elapsed'.format(time.time() - tCV))

    idxBestAlpha = int(np.nanargmax(rMatCV.mean(axis=0)))
    bestAlpha = alphaList[idxBestAlpha]
    alphaRescale = len(idxTrain)
    scaler = StandardScaler()
    scaler.fit(XLagMat[idxTrain, :])
    Xtrain = scaler.transform(XLagMat[idxTrain, :])
    Xtest = scaler.transform(XLagMat[idxTest, :])

    bestMod = Ridge(alpha=bestAlpha * alphaRescale, fit_intercept=False)
    bestMod.fit(Xtrain, y[idxTrain])
    yPred = bestMod.predict(Xtest)
    r = np.corrcoef(yPred, y[idxTest])[0, 1]

    saveScoreAll.append(rMatCV)
    saveAlphaRs.append(bestMod.alpha / alphaRescale)
    saveCoefRs.append(bestMod.coef_)
    saveScoreRs.append(r)
    saveYPredRs.append(yPred)
    saveIdxTest.append(idxTest)

    print('{:.2f} sec elapsed'.format(time.time() - tRs))

print('{:.2f} sec elapsed total\n'.format(time.time() - tTotal))


## Save modeling results
mdict={'alphaList': alphaList, 'saveScoreAll': saveScoreAll, 'saveAlphaRs': saveAlphaRs, 'saveCoefRs': saveCoefRs,
       'saveScoreRs': saveScoreRs, 'saveYPredRs': saveYPredRs, 'saveIdxTest': saveIdxTest, 'stim': y}
destname = '/home/knight/lbellier/DataWorkspace/_projects/PinkFloyd/{}/'.format(patientCode)
fname = '{}_{}_decodingMod_f{}-{}Rs{}CV.pkl'.format(patientCode, fileSuffix, freqBin, nSplitsRs, nSplitsCV)
fh = open(destname + fname, 'wb')
pickle.dump(mdict, fh)
fh.close()