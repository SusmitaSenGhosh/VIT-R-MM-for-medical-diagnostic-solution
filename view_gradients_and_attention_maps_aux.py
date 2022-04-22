#%% BN  #%%
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import random
import my_visualize_aux
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from load_data import load_data
from load_model import load_model
from sklearn.utils import class_weight
tf.config.run_functions_eagerly(True)
# tf.compat.v1.enable_eager_execution()
# %% set paths and parameters
data_name = 'Colorectal Histology'  #  ['Colorectal Histology','CBIS_DDSM','Chestxray','Fundus','ISIC18','PBC']
model_name =  ['Aux-ResNet-ViT-R-MM']
batch_size = 16 
input_size = 224
seed = 0
fileEnd ='.h5'
path = './output/' + data_name
weight_path = path+'/Aux-ResNet-ViT-R-MM/model.h5' 


savePath = path+'/attention_maps_Hybrid_Aux/'

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


#%%
model = load_model(model_name[0], no_class,weights)
model.load_weights(weight_path)

# x = model.layers[0].layers[-1].output
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dense(128, activation = tfa.activations.gelu)(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dense(32, activation = tfa.activations.gelu)(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dense(no_class, 'softmax')(x)
# model5 = Model(inputs=model.layers[0].layers[0].input, outputs=x) # second input provided to model

# for i in range(1,7):
#     model5.layers[-i].set_weights(model.layers[-i].get_weights())

# print(model.predict(np.expand_dims(trainS[5],axis = 0)))
print(model.predict(np.expand_dims(trainS[5],axis = 0)))
#%%
def make_gradcam_heatmap2(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # if pred_index is None:
        #     pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    #heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = (heatmap-tf.math.reduce_min(heatmap))/ (tf.math.reduce_max(heatmap)-tf.math.reduce_min(heatmap))

    return heatmap.numpy()
def make_gradcam_heatmap3(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # if pred_index is None:
        #     pred_index = tf.argmax(preds[0])
        class_channel = preds[2][:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    #heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = (heatmap-tf.math.reduce_min(heatmap))/ (tf.math.reduce_max(heatmap)-tf.math.reduce_min(heatmap))

    return heatmap.numpy()
def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):


    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    jet_heatmap = jet_heatmap/jet_heatmap.max()
    #print(jet_heatmap.max(),jet_heatmap.min())

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap* alpha + img
    superimposed_img = superimposed_img/superimposed_img.max()
    #superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
   # superimposed_img.save(cam_path)

    # Display Grad CAM
    return jet_heatmap,superimposed_img


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

        predicted_class_index = np.argmax(model.predict(img_array)[-1])

        # print(index,class_index,predicted_class_index2)
        # print(index,class_index,predicted_class_index3)
        if (predicted_class_index == class_index):
            count2 = count2+1
            print(index,class_index,predicted_class_index)

            img = img/scaling_factor

            last_conv_layer_name = "conv5_block2_out"
            model.layers[-1].activation = None
            preds = model.predict(img_array)[-1]
            heatmap_1 = make_gradcam_heatmap3(img_array, model, last_conv_layer_name, pred_index = class_index)
            heatmap_1, _ = save_and_display_gradcam(img, heatmap_1, cam_path="cam.jpg", alpha=0.5)
            
            heatmap_21,heatmap_22,heatmap_2,_,_,_ = my_visualize_aux.attention_map(model, img_array,alpha,gridSize = 7)        
    
            
            superimposed_1 = heatmap_1*alpha+img
            superimposed_1 = superimposed_1/superimposed_1.max()
            
            superimposed_21 = heatmap_21*alpha+img
            superimposed_21 = superimposed_21/superimposed_21.max()
    
            superimposed_22 = heatmap_22*alpha+img
            superimposed_22 = superimposed_22/superimposed_22.max()
        
            superimposed_2 = heatmap_2*alpha+img
            superimposed_2 = superimposed_2/superimposed_2.max()
        
            comb_heatmap = (heatmap_1+heatmap_2)/2 
            comb_heatmap = comb_heatmap/comb_heatmap.max()
            superimposedcomb = comb_heatmap*alpha+img
            superimposedcomb = superimposedcomb/superimposedcomb.max()
            
            
            fig, axs = plt.subplots(1, 1)
            axs.imshow(img)
            axs.axis('off')
            plt.savefig(savePath+'Train_class'+str(i)+'_sample'+str(count2)+'_image.jpeg',bbox_inches='tight',pad_inches = 0)
            plt.show()

            fig, axs = plt.subplots(1, 1)
            axs.imshow(superimposedcomb)
            axs.axis('off')
            plt.savefig(savePath+'Train_class'+str(i)+'_sample'+str(count2)+'_TransResNet_reverse_mlpmixer_Aux_'+str(class_index)+'_'+str(predicted_class_index)+'.jpeg',bbox_inches='tight',pad_inches = 0)
            plt.show()
            
            
            fig, axs = plt.subplots(1, 1)
            axs.imshow(superimposed_1)
            axs.axis('off')
            plt.savefig(savePath+'Train_class'+str(i)+'_sample'+str(count2)+'_TransResNet_reverse_mlpmixer_Aux_conv_'+str(class_index)+'_'+str(predicted_class_index)+'.jpeg',bbox_inches='tight',pad_inches = 0)
            plt.show()
            
            fig, axs = plt.subplots(1, 1)
            axs.imshow(superimposed_2)
            axs.axis('off')
            plt.savefig(savePath+'Train_class'+str(i)+'_sample'+str(count2)+'_TransResNet_reverse_mlpmixer_Aux_vit_'+str(class_index)+'_'+str(predicted_class_index)+'.jpeg',bbox_inches='tight',pad_inches = 0)
            plt.show()
    
            fig, axs = plt.subplots(1, 1)
            axs.imshow(superimposed_21)
            axs.axis('off')
            plt.savefig(savePath+'Train_class'+str(i)+'_sample'+str(count2)+'_TransResNet_reverse_mlpmixer_Aux_vit1_'+str(class_index)+'_'+str(predicted_class_index)+'.jpeg',bbox_inches='tight',pad_inches = 0)
            plt.show()
    
            fig, axs = plt.subplots(1, 1)
            axs.imshow(superimposed_22)
            axs.axis('off')
            plt.savefig(savePath+'Train_class'+str(i)+'_sample'+str(count2)+'_TransResNet_reverse_mlpmixer_Aux_vit2_'+str(class_index)+'_'+str(predicted_class_index)+'.jpeg',bbox_inches='tight',pad_inches = 0)
            plt.show()

        count1 = count1+1
        if count2 == no_sample or len(indices[0])-1==count1:
            break
