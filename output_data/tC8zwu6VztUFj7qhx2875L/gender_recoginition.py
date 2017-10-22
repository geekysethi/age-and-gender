


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

import pickle


db='wiki'
mat_path='/home/geekysethi/Desktop/age-and-gender/age-and-gender/data/wiki_crop/wiki.mat'
# mat_path='/data/wiki_crop/wiki.mat'
full_path, dob, gender, photo_taken, face_score, second_face_score, age= get_meta(mat_path, db)

temp_dataframe={"full_path":full_path,'gender':gender,'face_score':face_score,'second_face_score':second_face_score,'age':age}
df=pd.DataFrame(temp_dataframe)
print(df.head())
train_dir='/home/geekysethi/Desktop/age-and-gender/age-and-gender/data/wiki_crop/'
# train_dir='/data/wiki_crop/'

img_size=50
LR=1e-3
model_name='ageandgender-{}-{}.model'.format(LR,'2conv_basic')


# In[3]:


path=[]
type(df.full_path)
for i in df.full_path:
    i=str(i)[2:-2]
    path.append(os.path.join(train_dir,str(i)))
print(path[:10])    


# In[4]:


df.full_path=path
df.head()


# In[5]:


# def show_images(path):
#     fig=plt.figure()
#     for num, img_no in enumerate(path[:12]):
#         y=fig.add_subplot(3,4,num+1)
#         img=cv2.imread(img_no,3)
#         y.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         y.axes.get_xaxis().set_visible(False)
#         y.axes.get_yaxis().set_visible(False)

    
# show_images(path) 


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


# In[7]:


# df.hist()
# plt.plot


# In[8]:


data_0=(df.full_path[df.gender==0])[:1000]
data_1=shuffle(df.full_path[df.gender==1])
data_1=data_1[0][:1000]


# In[9]:


print(len(data_0))
print(len(data_1))

n_files=len(data_0)+len(data_1)
print(n_files)

allX = np.zeros((n_files, img_size, img_size, 3), dtype='float64')
ally = np.zeros(n_files)
count=0

for f in tqdm(data_0):
    img=cv2.imread(f,1)
    img=cv2.resize(img,(img_size,img_size))
    allX[count]=np.array(img)
    ally[count]=0
    count+=1
    
for f in tqdm(data_1):
    img=cv2.imread(f,1)
    img=cv2.resize(img,(img_size,img_size))
    allX[count]=np.array(img)
    ally[count]=1
    count+=1

# output=open('training_data_gender.pkl','wb')
# pickle.dump(allX,ally,output)
# output.close()
# print('Loading data...')
# pkl_file = open('training_data_gender.pkl', 'rb')
# allX,ally = pickle.load(pkl_file)
# pkl_file.close()
# print('Data loaded')



allX,ally=shuffle(allX,ally)
X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.2, random_state=42)
X, Y = shuffle(X, Y) 
Y = to_categorical(Y, 2)
Y_test = to_categorical(Y_test, 2)


# In[ ]:


img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping & rotating images
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)


# In[ ]:


###################################
# Define network architecture
###################################

# Input is a 32x32 image with 3 color channels (red, green and blue)
# network = input_data(shape=[None, img_size, img_size, 3],data_preprocessing=img_prep,data_augmentation=img_aug,name='input')
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
model = tflearn.DNN(network, checkpoint_p,tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')

###################################
# Train model for 100 epochs
###################################
model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=500,
      n_epoch=10, run_id=model_name, show_metric=True)

model.save(model_name)

