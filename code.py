import scipy.io
import numpy as np
from scipy.fft import fft
from scipy import signal
from PyEMD import EEMD
from scipy import stats
import matplotlib.pyplot as plt
import csv

def preprocess(X, Y):
    # Time Domain Features
    f1 = np.ptp(X)
    f2 = np.mean(X)
    f3 = np.var(X)
    f4 = stats.kurtosis(X)
    f5 = np.std(X)
    f6 = stats.median_abs_deviation(X)
    f7 = max(X) / (np.mean(np.sqrt(abs(X))))**2

    # Frequency domain features
    f8 = np.mean(Y**2)
    f9 = np.sum([-x*np.log2(x) for x in Y**2])
    rn = np.arange(1, len(Y)+1,1)
    f10 = np.sum(np.multiply(Y,rn))/np.sum(Y)
    f11 = np.sum(np.multiply(Y**2,rn))/np.sum(Y**2)
    f, t, Zxx = signal.stft(X)
    f12 = np.sum(abs(Zxx)**2)
    f = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, 1]
    return f

if __name__ == "__main__":
    Xtrain = []
    mat = scipy.io.loadmat('97.mat')
    FE = mat['X097_FE_time']
    bin_size = 300  ## hyperparam
    FE = FE[:len(FE) - len(FE)%bin_size]
    FE = np.reshape(FE, (-1, bin_size))
    for i in range(len(FE)):
        if i%2 == 0:
            print(i)
        fft_x = np.abs(fft(FE[i]))
        eemd = EEMD()
        eIMFs = eemd.eemd(FE[i])
        corr = []
        for i in range(eIMFs.shape[0]):
            cor = stats.pearsonr(FE[i], eIMFs[i])
            corr.append(cor[0])

        max_index = corr.index(max(corr))
        imf = eIMFs[i]
        f = preprocess(imf, fft_x)
        Xtrain.append(f)

    with open("train1.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(Xtrain)
        