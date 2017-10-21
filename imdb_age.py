

from utils import get_meta
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import cv2
# get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm


db='imdb'
# mat_path='/data/wiki_crop/wiki.mat'
mat_path='/home/geekysethi/Desktop/age-and-gender/imdb/imdb.mat'
full_path, dob, gender, photo_taken, face_score, second_face_score, age= get_meta(mat_path, db)

temp_dataframe={"full_path":full_path,'gender':gender,'face_score':face_score,'second_face_score':second_face_score,'age':age}
df=pd.DataFrame(temp_dataframe)
print(df.head())
# train_dir='/data/wiki_crop/'


print(df.head())
print(df.isnull().values.any())
print(df.gender.isnull().sum())
df.drop(df[df.gender.isnull()==True].index,inplace=True)
print(df.gender.isnull().sum())
print(df.head())
df[df.face_score<0]=np.nan
df.drop(df[df.face_score.isnull()==True].index,inplace=True)
print(df.head())

print(dg.gender.values_count())
























