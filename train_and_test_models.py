######### Load Pacakges #######
import numpy as np
import random
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.python.keras import backend as K
import os
import time
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from load_data import load_data
from load_model import load_model
tf.config.run_functions_eagerly(True)
# tf.compat.v1.enable_eager_execution()

#%% set paths and parameters
dataset = 'Colorectal Histology'  #  ['Colorectal Histology','CBIS_DDSM','Chestxray','Fundus','ISIC18','PBC']
model_name = 'ResNet-ViT-R-MM' # ['ViT','ViT-R','ViT-MM', 'ViT-R-MM','ResNet50','ResNet-ViT','ResNet-ViT-R-MM','Aux-ResNet-Vit-R-MM']

seed = 0

if model_name in ['ViT','ViT-R','ViT-MM', 'ViT-R-MM']:
    epochs = 100
else:
    epochs = 50
batch_size = 16
base_output_path = './output/'
weight_path = base_output_path + dataset + '/'+ model_name 
fileEnd ='.h5'

if not os.path.exists(weight_path):
    os.makedirs(weight_path)  

#%% define functions

def reset_random_seeds(seed):
   os.environ['PYTHONHASHSEED']=str(seed)
   os.environ['TF_DETERMINISTIC_OPS'] = '1'
   tf.random.set_seed(seed)
   np.random.seed(seed)
   random.seed(seed)
   
def exp_decay(epoch):
    initial_lrate = 0.0002
    k = 0.1#0.025
    lrate = initial_lrate * np.exp(-k*epoch)
    print('lr:',lrate)
    return lrate

def evaluate_performance(conf): 
    n = conf.shape[0]
    acsa = 0
    acsp = 0
    acsf = 0
    acc_list = []
    for i in range(0,n):
        acsa = acsa + conf[i,i]/sum(conf[i,:])
        acc_list.append(round(100*conf[i,i]/sum(conf[i,:]),2))
        acsp = acsp + conf[i,i]/sum(conf[:,i])
        acsf = acsf +  2* (conf[i,i]/sum(conf[i,:]))*(conf[i,i]/sum(conf[:,i]))/((conf[i,i]/sum(conf[i,:]))+(conf[i,i]/sum(conf[:,i])))
    acsa = round(100*acsa/n,2)
    acsp = round(100*acsp/n,2)
    acsf = round(100*acsf/n,2)
    return acsa,acsp,acsf

#%% load data  

reset_random_seeds(seed)
trainS, labelTr, testS, labelTs = load_data(dataset)
no_class = len(np.unique(labelTr))
labelsCat=to_categorical(labelTr)

shuffleIndex=np.random.choice(np.arange(labelTr.shape[0]), size=(labelTr.shape[0],), replace=False)
trainS=trainS[shuffleIndex]
labelTr=labelTr[shuffleIndex]
labelsCat=labelsCat[shuffleIndex]

weights = class_weight.compute_class_weight('balanced', np.unique(labelTr),labelTr)
classes = list(np.unique(labelTr))
class_weights = {classes[i]: weights[i] for i in range(len(classes))}

#%% load, train nd test model
model = load_model(model_name, no_class, weights)

acsaSaveTr, acspSaveTr, acsfSaveTr=np.zeros((epochs,),dtype = np.float32), np.zeros((epochs,),dtype = np.float32), np.zeros((epochs,),dtype = np.float32)
acsaSaveTs, acspSaveTs, acsfSaveTs=np.zeros((epochs,),dtype = np.float32), np.zeros((epochs,),dtype = np.float32), np.zeros((epochs,),dtype = np.float32)
confMatSaveTr,confMatSaveTs=np.zeros((epochs, no_class, no_class),dtype = np.float32), np.zeros((epochs, no_class, no_class),dtype = np.float32)
lossSaveTr, lossSaveTs = np.zeros((epochs,),dtype = np.float32),np.zeros((epochs,),dtype = np.float32)

epoch = 0
while epoch<epochs:
    start_time = time.time()

    K.set_value(model.optimizer.learning_rate, exp_decay(epoch))  # set new learning_rate

    model.fit(trainS, labelsCat,batch_size=batch_size, verbose=1, class_weight = class_weights)
    lossSaveTr[epoch] = model.history.history['loss'][0]
    print("training loss :", model.history.history['loss'][0])
    lossSaveTs[epoch] = model.evaluate(testS,to_categorical(labelTs))
    print("test loss :", lossSaveTs[epoch])

    pLabel=np.argmax(model.predict(trainS,batch_size= batch_size), axis=1)
    confMat=confusion_matrix(labelTr, pLabel)
    acsa, acsp, acsf = evaluate_performance(confMat)
    print('Train: epoch: ', epoch, 'ACSA: ', np.round(acsa, 2), 'ACSP: ', np.round(acsp, 2), 'ACSF: ', np.round(acsf, 2))
    acsaSaveTr[epoch], acspSaveTr[epoch], acsfSaveTr[epoch]=acsa, acsp, acsf
    confMatSaveTr[epoch]=confMat
        
    pLabel=np.argmax(model.predict(testS,batch_size= batch_size), axis=1)
    confMat=confusion_matrix(labelTs, pLabel)
    acsa, acsp, acsf = evaluate_performance(confMat)
    print('Test: epoch: ', epoch, 'ACSA: ', np.round(acsa, 2), 'ACSP: ', np.round(acsp, 2), 'ACSF: ', np.round(acsf, 2))
    acsaSaveTs[epoch], acspSaveTs[epoch], acsfSaveTs[epoch]=acsa, acsp, acsf
    confMatSaveTs[epoch]=confMat

    epoch = epoch+1
    end_time = time.time() 
    print("total time taken ", epoch, " loop: ", end_time - start_time)
    
    if epoch ==50:
        model.save(weight_path+'/model_49'+fileEnd)
        
model.save(weight_path+'/model'+fileEnd)

recordSave=weight_path+'/Record'
np.savez(recordSave, acsaSaveTr=acsaSaveTr, acspSaveTr=acspSaveTr, acsfSaveTr=acsfSaveTr,acsaSaveTs=acsaSaveTs, acspSaveTs=acspSaveTs, acsfSaveTs=acsfSaveTs, confMatSaveTr=confMatSaveTr, confMatSaveTs=confMatSaveTs,lossSaveTr=lossSaveTr, lossSaveTs=lossSaveTs)
