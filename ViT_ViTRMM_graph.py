#%%
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
import os
import numpy as np
import statistics as stat
import xlwt
from xlwt import Workbook
import matplotlib.pyplot as  plt



def evaluate_performance(conf): 
    tp = np.trace(conf)
    fp_fn = np.sum(np.sum(conf))-np.trace(conf)
    acsa = round(100*tp/(tp+fp_fn),2)
    acsp = round(100*tp/(tp+fp_fn),2)
    acsf = round(2*acsa*acsp/(acsa+acsp),2)
    return acsa,acsp,acsf

outputpath = './output/'
datasets = ['Colorectal Histology','ISIC18','CBIS_DDSM','Chestxray','Fundus','PBC'] 
title = ['Colorectal Histology','ISIC18','CBIS_DDSM','Chestxray','Fundus','PBC'] 
fontsize = 18
epoch = 100
acsa = np.zeros(epoch)
acsp = np.zeros(epoch)
acsf = np.zeros(epoch)

kernel_size = 3
kernel = np.ones(kernel_size) / kernel_size
color = ['#F17720','#0474BA']
count = 1
for d,t in zip(datasets,title):
    ax = plt.subplot(1,1,1)
    folders = ['ViT','ViT-R-MM']
    count = 0
    for i in folders:
        a = np.load(os.path.join(outputpath,d,i,'Record.npz'))
        for k in range(0,epoch):
            acsa[k],acsp[k],acsf[k] = evaluate_performance(a['confMatSaveTr'][k])
        ax.plot(range(0,epoch-kernel_size+1),np.convolve(acsf, kernel, mode='valid'),'-',color = color[count],linewidth = 2.5)
        for k in range(0,epoch):
            acsa[k],acsp[k],acsf[k] = evaluate_performance(a['confMatSaveTs'][k])
        ax.plot(range(0,epoch-kernel_size+1),np.convolve(acsf, kernel, mode='valid'),'--',color = color[count],linewidth = 2.5)
        count = count+1
    ax.set_ylabel("ACSF (%)",fontsize = fontsize)
    ax.set_xlabel("Epochs",fontsize = fontsize)
    ax.tick_params(axis='x', labelsize = 15)
    ax.tick_params(axis='y', labelsize = 15)
    # plt.legend(["VIT_Train","VIT_Test","VIT_reverse_Train","VIT_reverse_Test","VIT_Reverse_MLPmixer_Train","VIT_Reverse_MLPmixer_Test"])
    ax.set_title(t,fontsize = fontsize)
    ax.autoscale(enable=True, axis='x', tight=True)
    # plt.ylim([50,95])
    count = count + 1
    plt.show()
plt.legend(["ViT Train","ViT Test","ViT-R-MM Train","ViT-R-MM Test"],loc='upper center', bbox_to_anchor=(0.5, -0.2),fontsize = fontsize,ncol = 4)
plt.show()
