

from utils import get_meta
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import cv2
# get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm
from sklearn.cross_validation import train_test_split


from tflearn.data_utils import shuffle, to_categorical
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy

import tensorflow as tf





db='wiki'
mat_path='/data/wiki_crop/wiki.mat'
# mat_path='/home/geekysethi/Desktop/age-and-gender/age-and-gender/data/wiki_crop/wiki.mat'
full_path, dob, gender, photo_taken, face_score, second_face_score, age= get_meta(mat_path, db)

temp_dataframe={"full_path":full_path,'gender':gender,'face_score':face_score,'second_face_score':second_face_score,'age':age}
df=pd.DataFrame(temp_dataframe)
print(df.head())
train_dir='/data/wiki_crop/'
# train_dir='/home/geekysethi/Desktop/age-and-gender/age-and-gender/data/wiki_crop'
img_size=28
LR=1e-3
model_name='ageandgender-{}-{}.model'.format(LR,'2conv_basicnew')




path=[]
type(df.full_path)
for i in df.full_path:
    i=str(i)[2:-2]
    path.append(os.path.join(train_dir,str(i)))
# print(path[:10])    


df.full_path=path
# df.head()


# In[6]:




print(df.head())
print(df.isnull().values.any())
print(df.gender.isnull().sum())
df.drop(df[df.gender.isnull()==True].index,inplace=True)
print(df.gender.isnull().sum())
print(df.head())
df[df.face_score<0]=np.nan
df.drop(df[df.face_score.isnull()==True].index,inplace=True)
print(df.head())
df = df.sample(frac=1).reset_index(drop=True)

# In[8]:


img_paths=df.full_path.values
gender=df.gender.values


# In[9]:


print(df.gender.value_counts())


for i in img_paths[:10]:
    print(i)



# img_paths,gender=shuffle(img_paths,gender)
# n_files=len(df)
n_files=20000
gender=gender[:n_files]
unique, counts = np.unique(gender, return_counts=True)
print(counts)

def train_images(n_files,img_size,img_path):
    total_images = np.zeros((n_files, img_size, img_size, 3))
    count=0
    for img_path in tqdm(img_paths[:n_files]):
        img=cv2.imread(img_path,1)
        img=cv2.resize(img,(img_size,img_size))
        total_images[count]=np.array(img.reshape(-1,img_size,img_size,3))
        count+=1
    np.save('training_data_age-and-gender.npy',total_images)
    print('training data saved')
    return total_images



total_images=train_images(n_files,img_size,img_paths)






X, X_test, Y, Y_test = train_test_split(total_images, gender, test_size=0.2, random_state=42)
X, Y = shuffle(X, Y) 
Y = to_categorical(Y, 2)
Y_test = to_categorical(Y_test, 2)




print(np.shape(X))
print(np.shape(Y))

print(Y[:12])




###################################################


print("Training about to start")


tf.reset_default_graph()

# Convolutional network building


network = input_data(shape=[None, img_size, img_size, 3],   name='input')


# 1: Convolution layer with 32 filters, each 3x3x3
conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)

# 3: Convolution layer with 64 filters
conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')

# 4: Convolution layer with 64 filters
conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')

# 5: Max pooling layer
network = max_pool_2d(conv_3, 2)

# 6: Fully-connected 512 node layer
network = fully_connected(network, 512, activation='relu')

# 7: Dropout layer to combat overfitting
network = dropout(network, 0.5)

# 8: Fully-connected layer with two outputs
network = fully_connected(network, 2, activation='softmax')

# Configure how the network will be trained
acc = Accuracy(name="Accuracy")
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005, metric=acc)

# Wrap the network in a model object
model = tflearn.DNN(network, checkpoint_path='model_cat_dog_78.tflearn', max_checkpoints = 3,
                    tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')

###################################
# Train model for 100 epochs
###################################
model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=500,
      n_epoch=10, run_id='model_cat_dog_6', show_metric=True)

model.save('model_cat_dog_6_final.tflearn')






















