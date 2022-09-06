from KitNET.KitNET import KitNET
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from scipy.stats import ttest_ind

class Kitsune:
    def __init__(self, NumFeatures = 4, max_autoencoder_size = 10, AD_grace_period = 40000, FM_grace_period = 2000, learning_rate = 0.1, hidden_ratio = 0.75):
        self.NumFeatures = NumFeatures
        self.AnomDetector = KitNET(self.NumFeatures, max_autoencoder_size, FM_grace_period, AD_grace_period, learning_rate, hidden_ratio)
        self.count = 0
        self.Adaptive_Grace = AD_grace_period + FM_grace_period
        # Flag to turn on adaptive mode
        self.ENABLE_ADAPTIVE = 0
        # Using fixed interval adaptive
        self.Adaptive_interval = 10000
        self.prevCount = 0

    def proc_next_vector(self, row):
        # Flag for determining when to start adaptive training
        if self.count > self.Adaptive_Grace and self.ENABLE_ADAPTIVE == 0:
            self.ENABLE_ADAPTIVE = 1
        
        # If row vector is empty, return invalid RMSE
        if len(row) == 0:
            return -1

        self.count += 1
        if self.count%1000 == 0:
            print(f'{self.count} records processed')
        # process KitNET
        self.rmse = self.AnomDetector.process(row)

        # If Adaptive is enabled, we introduce periodic retraining
        if self.ENABLE_ADAPTIVE and self.count - self.prevCount > self.Adaptive_interval and self.rmse < 0.75:
            # Implemented hard thresholding right now
            # Have to replace it by auto one
            self.prevCount = self.count
            self.AnomDetector.train(row)
        # Calculating stats
        # self.rmse_var = ((self.rmse_var * self.rmse_mean)**2 * self.counter + self.rmse * self.rmse) / (self.counter + 1)
        # self.rmse_mean = (self.counter * self.rmse_mean + self.rmse) / (self.counter + 1)
        # self.rmse_var -= self.rmse_mean**2
        # alert = 0
        # if self.counter > self.Z_Grace:
        #     zval = (self.rmse - self.rmse_mean) / math.sqrt(self.rmse_var + 1)
        #     if zval > self.zscore or zval < -self.zscore:
        #         alert = 1
        return self.rmse

if __name__ == "__main__":
    RMSE = []
    path = './Healthy Data/h30hz0.txt'
    df = pd.read_csv(path, sep='\t', names=['acc1', 'acc2', 'acc3', 'acc4', 'acc5'], header=None)
    cols = ['acc1', 'acc2', 'acc3', 'acc4']
    df = df[cols]
    df = df[:60000]
    df = pd.DataFrame(np.abs(fft(df)), columns=cols)
    clasifier = Kitsune()
    
    for i in range(len(df)):
        RMSE.append(clasifier.proc_next_vector(df.iloc[i]))
    
    b_RMSE = []

    print('Broken Tooth')
    path = './BrokenTooth Data/b30hz0.txt'
    df = pd.read_csv(path, sep='\t', names=['acc1', 'acc2', 'acc3', 'acc4', 'acc5'], header=None)
    cols = ['acc1', 'acc2', 'acc3', 'acc4']
    df = df[cols]
    df = df[:18000]
    df = df[:18000]
    df = np.abs(fft(df))
    df = pd.DataFrame(np.abs(fft(df)), columns=cols)
    for i in range(len(df)):
        b_RMSE.append(clasifier.proc_next_vector(df.iloc[i]))
    
    t, p = ttest_ind(RMSE, b_RMSE, equal_var=False)
    print(t, p)

    x = RMSE[4200:] + b_RMSE
    plt.plot(x)
    plt.savefig('test.png')
    # mu = np.mean(RMSE)
    # sigma = np.std(RMSE)   
    # count, bins, ignored = plt.hist(RMSE, 30, density=True)
    # plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')

    # mu = np.mean(b_RMSE)
    # sigma = np.std(b_RMSE)   
    # count, bins, ignored = plt.hist(b_RMSE, 30, density=True)
    # plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='b')
    
    # plt.show()
    # # plt.plot(RMSE)
    # plt.savefig('test.png')