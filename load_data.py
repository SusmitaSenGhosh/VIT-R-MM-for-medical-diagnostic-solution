import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_data(data_name):

    if data_name == 'CBIS_DDSM':
        images=[]
        labels=[]
        feature_dictionary = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'label_normal': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string)
            }

        def _parse_function(example, feature_dictionary=feature_dictionary):
            parsed_example = tf.io.parse_example(example, feature_dictionary)
            return parsed_example

        def read_data(filename):
            full_dataset = tf.data.TFRecordDataset(filename,num_parallel_reads=tf.data.experimental.AUTOTUNE)
            # full_dataset = full_dataset.shuffle(buffer_size=31000)
            full_dataset = full_dataset.cache()
            print("Size of Training Dataset: ", len(list(full_dataset)))
            
            feature_dictionary = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'label_normal': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string)
            }   

            full_dataset = full_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            for image_features in full_dataset:
                image = image_features['image'].numpy()
                image = tf.io.decode_raw(image_features['image'], tf.uint8)
                image = tf.reshape(image, [299, 299])        
                image=image.numpy()
                image=cv2.resize(image,(224,224))
                image=cv2.merge([image,image,image])        
                #plt.imshow(image)
                images.append(image)
                labels.append(image_features['label'].numpy())

        filenames=['D:/Simpi/data/IDC/training10_0/training10_0.tfrecords',
                  # 'D:/Simpi/data/IDC/training10_1/training10_1.tfrecords',
                  # 'D:/Simpi/data/IDC/training10_2/training10_2.tfrecords',
                  # 'D:/Simpi/data/IDC/training10_3/training10_3.tfrecords',
                  # 'D:/Simpi/data/IDC/training10_4/training10_4.tfrecords']
                  ]
            
        for file in filenames:
            read_data(file)

        
        data = np.stack(images, axis=0 )
        label = np.asarray(labels)
        del images, labels

        trainS, testS, labelTr, labelTs = train_test_split(data, label, test_size=0.2, random_state=0,shuffle=True)
        del data, label

        
    elif data_name == 'Colorectal Histology':
        path = './data/Colorectal Histology/'
        temp = np.load(path+'/sample.npz')
        data = temp['arr_0']
        label = temp['arr_1'] 
        del temp

        trainS, testS, labelTr, labelTs = train_test_split(data, label, test_size=0.2, random_state=0,shuffle=True)
        del data, label
        
    elif data_name == 'ISIC18':
        path = './data/ISIC18/'
        temp = np.load(path+'/sample.npz')
        data = temp['arr_0']
        label = temp['arr_1']
        del temp

        trainS, testS, labelTr, labelTs = train_test_split(data, label, test_size=0.2, random_state=0,shuffle=True)
        del data, label
    


    elif data_name == 'Chestxray':
        path = './data/Chestxray/'
        temp = np.load(path+'/train.npz')
        trainS = temp['arr_0']
        labelTr = np.asarray(temp['arr_1'],dtype = np.uint8)
        del temp

        temp = np.load(path+'/test.npz')
        testS = temp['arr_0']
        labelTs = np.asarray(temp['arr_1'],dtype = np.uint8)
        del temp
        
    elif data_name == 'PBC':
        path = './data/PBC/'
        temp = np.load(path+'/sample.npz')
        data = temp['arr_0']
        data1 = np.zeros((data.shape[0],data.shape[1],data.shape[2],data.shape[3]),dtype = np.float32)
        for i in range(data.shape[0]):
            data1[i] = np.asarray(data[i]/255,dtype = np.float32)
        del data
        label = temp['arr_1'] 
        del temp

        trainS, testS, labelTr, labelTs = train_test_split(data1, label, test_size=0.2, random_state=0,shuffle=True)
        del data1, label

        
    elif data_name == 'Fundus':
        path = './data/Fundus/'
        temp = np.load(path+'/train.npz')
        trainS = np.asarray(temp['arr_0']/255,dtype = np.float32)
        #label = np.asarray(temp['arr_1'])
        labelTr = temp['arr_1'] 
        del temp
        
        temp = np.load(path+'/test.npz')
        testS = np.asarray(temp['arr_0']/255,dtype = np.float32)
        #label = np.asarray(temp['arr_1'])
        labelTs = temp['arr_1'] 
        del temp

    return trainS, labelTr, testS,labelTs