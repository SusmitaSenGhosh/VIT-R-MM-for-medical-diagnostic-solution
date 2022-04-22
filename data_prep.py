import os
import numpy as np
import csv
import cv2
#%% ISIC18

path = './ISIC2018_Task3_Training_Input' # add data path here
csv_path = './ISIC_grounfthruth.csv' # add csv path here
save_path = './data/ISIC18' # add path to save the data

if not os.path.exists(save_path):
    os.makedirs(save_path)  
    
classes = [0,1,2,3,4]
file_name = []
y = []
with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        print(line_count)
        if line_count !=0:
            file_name.append(row[0])
            img = np.asarray(cv2.resize(cv2.imread(os.path.join(path,row[0]+'.jpg')),(224,224))/255, dtype =np.float32)
            y.append(np.argmax((row[1::])))
            if line_count ==1:
                x=np.expand_dims(img,axis = 0)
            else:
                x = np.concatenate((x,np.expand_dims(img,axis = 0)),axis = 0)
        line_count += 1
        
np.savez(save_path+'/sample.npz', x, y,file_name)


#%% PBC
import time
import cv2

path = './PBC_dataset_normal_DIB/'  # add data path here
save_path = './data/PBC' # add path to save the data
if not os.path.exists(save_path):
    os.makedirs(save_path) 
    
x = []
y = []
dirs = os.listdir(path)
count = 0
class_name = 0
for folder in dirs:
    print(count)
    file_list = os.listdir(os.path.join(path,folder))
    
    for file in file_list:
        print(count)
        since = time.time()
        img = cv2.imread(os.path.join(path,folder,file))
        img = np.asarray(cv2.resize(img,(224,224)),dtype = np.uint8)
        if count == 0:
            x=np.expand_dims(img,axis = 0)
            y.append(class_name)
        else:
            x = np.concatenate((x,np.expand_dims(img,axis = 0)),axis = 0)
            y.append(class_name)
        count = count+1
    class_name = class_name+1


np.savez(save_path+'/sample.npz', x, y)

#%% Fundus

import time
import cv2

csv_path = './regular_fundus_images/regular-fundus-training/regular-fundus-training.csv' # add csv path here
base_path = './regular_fundus_images/regular-fundus-training/Images' # add data path here

save_path = './data/Fundus' # add path to save the data
if not os.path.exists(save_path):
    os.makedirs(save_path) 
    
x = []
y = []
count = 0
with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    next(csv_reader)
    for row in csv_reader:
        print(count)
        file_path = base_path+row[2][24::]
        img = cv2.imread(file_path)
        img = np.asarray(cv2.resize(img,(224,224)),dtype = np.uint8)
        if row[1][-2]=='l':
            class_name = int(row[4])
        else:
            class_name = int(row[5])
            
        if count == 0:
            x = np.expand_dims(img,axis = 0)
            y.append(class_name)
        else:
            x = np.concatenate((x,np.expand_dims(img,axis = 0)),axis = 0)
            y.append(class_name)
        count = count+1
        
np.savez(save_path+'/train.npz', x, y)

csv_path = './regular_fundus_images/regular-fundus-validation/regular-fundus-validation.csv' # add csv path here
base_path = './regular_fundus_images/regular-fundus-validation/Images' # add data path here

save_path = './data/Fundus' # add path to save the data
if not os.path.exists(save_path):
    os.makedirs(save_path) 
    
x = []
y = []
count = 0
with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    next(csv_reader)
    for row in csv_reader:
        print(count)
        file_path = base_path+row[2][26::]
        img = cv2.imread(file_path)
        img = np.asarray(cv2.resize(img,(224,224)),dtype = np.uint8)
        if row[1][-2]=='l':
            class_name = int(row[4])
        else:
            class_name = int(row[5])
            
        if count == 0:
            x = np.expand_dims(img,axis = 0)
            y.append(class_name)
        else:
            x = np.concatenate((x,np.expand_dims(img,axis = 0)),axis = 0)
            y.append(class_name)
        count = count+1
        
np.savez(save_path+'/test.npz', x, y)
#%% Colorectal Histology

import numpy as np
from PIL import Image 
import os

path = './Kather_texture_2016_image_tiles_5000' # add data path here
save_path = './data/Colorectal Histolgy' # add path to save the data
if not os.path.exists(save_path):
    os.makedirs(save_path) 
    
x = []
y = []
dirs = os.listdir(path)
count = 0
class_name = 0
for folder in dirs:
    print(count)
    file_list = os.listdir(os.path.join(path,folder))
    for file in file_list:
        img = Image.open(os.path.join(path,folder,file))
        img = np.asarray(img.resize((224,224)),dtype = np.float32)/255
        if count == 0:
            x=np.expand_dims(img,axis = 0)
            y.append(class_name)
        else:
            x = np.concatenate((x,np.expand_dims(img,axis = 0)),axis = 0)
            y.append(class_name)
        count = count+1
    class_name = class_name+1


np.savez(save_path+'/sample.npz', x, y)
#%% Chestxray

import os
import numpy as np
import csv
import cv2


classes = [0,1]
y = []
count = 0
class_count = 0
path = './chestxray/train' # add train data path here
save_path = './data/Chestxray' # add path to save the data
if not os.path.exists(save_path):
    os.makedirs(save_path) 
    
folders = ['NORMAl','PNEUMONIA']
for folder in folders:
    file_name = os.listdir(os.path.join(path,folder))
    for file in file_name:
        img = np.asarray(cv2.resize(cv2.imread(os.path.join(path,folder,file)),(224,224))/255,dtype = np.float32)
        y.append(classes[class_count])
        if count ==0:
            x=np.expand_dims(img,axis = 0)
        else:
            x = np.concatenate((x,np.expand_dims(img,axis = 0)),axis = 0)
        count += 1
        print(count)
    class_count += 1
np.savez(save_path+'/train.npz', x, y,file_name)


classes = [0,1]
y = []
count = 0
class_count = 0
path = './chestxray/test' # add test data path here
folders = ['NORMAl','PNEUMONIA']
for folder in folders:
    file_name = os.listdir(os.path.join(path,folder))
    for file in file_name:
        img = np.asarray(cv2.resize(cv2.imread(os.path.join(path,folder,file)),(224,224))/255,dtype = np.float32)
        y.append(classes[class_count])
        if count ==0:
            x=np.expand_dims(img,axis = 0)
        else:
            x = np.concatenate((x,np.expand_dims(img,axis = 0)),axis = 0)
        count += 1
        print(count)
    class_count += 1
np.savez(save_path+'/test.npz', x, y,file_name)