import glob
import cv2
import numpy as np
from random import shuffle


train_data_mask = []

with_mask = [1, 0]
without_mask = [0, 1]

for img in glob.glob("C:/Users/Alzaib/Desktop/Projects/MaskDetection/dataset/with_mask/*.png"):
    n= cv2.imread(img)
    n = cv2.resize(n, (100,100))
    n = cv2.cvtColor(n, cv2.COLOR_RGB2GRAY)
    train_data_mask.append([n, with_mask])
train_data_mask = np.array(train_data_mask)
file_name = 'training_data_withmask-{}.npy'.format(1)
np.save(file_name,train_data_mask)



train_data_withoutmask = []

for img in glob.glob("C:/Users/Alzaib/Desktop/Projects/MaskDetection/dataset/without_mask/*.png"):
    n= cv2.imread(img)
    n = cv2.resize(n, (100,100))
    n = cv2.cvtColor(n, cv2.COLOR_RGB2GRAY)
    train_data_withoutmask.append([n, without_mask])

train_data_withoutmask = np.array(train_data_withoutmask)

#shuffle(train_data)

file_name = 'training_data_withoutmask-{}.npy'.format(1)
np.save(file_name,train_data_withoutmask)



print (train_data_mask.shape)
print (train_data_withoutmask.shape)
