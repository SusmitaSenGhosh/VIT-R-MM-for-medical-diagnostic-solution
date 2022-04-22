#%% BN  #%%
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import random
import my_visualize
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
from load_data import load_data
from load_model import load_model
from sklearn.utils import class_weight
tf.config.run_functions_eagerly(True)
# tf.compat.v1.enable_eager_execution()

# %% set paths and parameters
data_name = 'Colorectal Histology'  #  ['Colorectal Histology','CBIS_DDSM','Chestxray','Fundus','ISIC18','PBC']
model_name =  ['ViT', 'ViT-R-MM']
batch_size = 16 
input_size = 224
seed = 0
fileEnd ='.h5'
path = './output/' + data_name
weight_path1 = path+'/ViT/model.h5' 
weight_path2 = path +'/ViT-R-MM/model.h5'

savePath = path+'/attention_maps_ViT_ViTRMM/'

if not os.path.exists(savePath):
    os.makedirs(savePath)

def reset_random_seeds(seed):
   os.environ['PYTHONHASHSEED']=str(seed)
   os.environ['TF_DETERMINISTIC_OPS'] = '1'
   tf.random.set_seed(seed)
   np.random.seed(seed)
   random.seed(seed)
   
   #%%
reset_random_seeds(0)
trainS, labelTr, testS, labelTs = load_data(data_name)
no_class = len(np.unique(labelTr))
labelsCat=to_categorical(labelTr)

shuffleIndex=np.random.choice(np.arange(labelTr.shape[0]), size=(labelTr.shape[0],), replace=False)
trainS=trainS[shuffleIndex]
labelTr=labelTr[shuffleIndex]
labelsCat=labelsCat[shuffleIndex]


weights = class_weight.compute_class_weight('balanced', np.unique(labelTr),labelTr)
classes = list(np.unique(labelTr))
class_weights = {classes[i]: weights[i] for i in range(len(classes))}
no_class = len(np.unique(labelTr))

if data_name == 'CBIS_DDSM':
    scaling_factor = 255
else:
    scaling_factor = 1

if data_name == 'CBIS_DDSM':
    alpha  = 0.25
else:
    alpha = 0.5
    
no_sample = 50

#%% load model for ResNet50
model = load_model(model_name[0], no_class,weights)
model.load_weights(weight_path1)

x = model.layers[0].layers[-1].output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(128, activation = tfa.activations.gelu)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(32, activation = tfa.activations.gelu)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(no_class, 'softmax')(x)
model1 = Model(inputs=model.layers[0].layers[0].input, outputs=x) # second input provided to model

for i in range(1,7):
    model1.layers[-i].set_weights(model.layers[-i].get_weights())

print(model.predict(np.expand_dims(trainS[5],axis = 0)))
print(model1.predict(np.expand_dims(trainS[5],axis = 0)))
del model
#%%

model = load_model(model_name[1], no_class,weights)
model.load_weights(weight_path2)

x = model.layers[0].layers[-1].output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(128, activation = tfa.activations.gelu)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(32, activation = tfa.activations.gelu)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(no_class, 'softmax')(x)
model2 = Model(inputs=model.layers[0].layers[0].input, outputs=x) # second input provided to model

for i in range(1,7):
    model2.layers[-i].set_weights(model.layers[-i].get_weights())

print(model.predict(np.expand_dims(trainS[5],axis = 0)))
print(model2.predict(np.expand_dims(trainS[5],axis = 0)))
del model

#%%
        
alpha = 0.75
for i in range(0,no_class):
    count1 = 0
    indices = np.where(labelTs==i)
    count2 = 0 
    while(True):
        index = indices[0][count1]
        img =testS[index]
        img_array = np.expand_dims(img,axis = 0)
        class_index = int(labelTs[index])
        
        predicted_class_index1 = np.argmax(model1.predict(img_array))
        predicted_class_index2 = np.argmax(model2.predict(img_array))

        if (predicted_class_index1 == class_index and predicted_class_index2 == class_index ):
            count2 = count2+1
            print(index,class_index,predicted_class_index1)#,predicted_class_index2,predicted_class_index3,predicted_class_index4,predicted_class_index5,predicted_class_index6)
    
            heatmap1,_ = my_visualize.attention_map(model1, img_array,alpha,gridSize = 14)        
            heatmap2,_ = my_visualize.attention_map(model2, img_array,alpha,gridSize = 14)        
            
            
            img = img/scaling_factor
            superimposed1 = heatmap1*alpha+img
            superimposed1 = superimposed1/superimposed1.max()
            
            superimposed2 = heatmap2*alpha+img
            superimposed2 = superimposed2/superimposed2.max()

            fig, axs = plt.subplots(1, 1)
            axs.imshow(img)
            axs.axis('off')
            plt.savefig(savePath+'Train_class'+str(i)+'_sample'+str(count2)+'_image.jpeg',bbox_inches='tight',pad_inches = 0)
            plt.show()
        
            fig, axs = plt.subplots(1, 1)
            axs.imshow(superimposed1)
            axs.axis('off')
            plt.savefig(savePath+'Train_class'+str(i)+'_sample'+str(count2)+'_vit_'+str(class_index)+'_'+str(predicted_class_index1)+'.jpeg',bbox_inches='tight',pad_inches = 0)
            plt.show()
        
        
            fig, axs = plt.subplots(1, 1)
            axs.imshow(superimposed2)
            axs.axis('off')
            plt.savefig(savePath+'Train_class'+str(i)+'_sample'+str(count2)+'_vit_r_mm_'+str(class_index)+'_'+str(predicted_class_index2)+'.jpeg',bbox_inches='tight',pad_inches = 0)
            plt.show()
            

        count1 = count1+1
        if count2 == no_sample or len(indices[0])-1==count1:
            break
