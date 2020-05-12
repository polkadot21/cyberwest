import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 33)


#make a random array, x[0] = 1, x[i] = x[i-1] + random.uniform(-1, 1)
def random_array(M):
    #random.seed(42)
    N=1024
    x = range(N)
    y = np.zeros(N*M)
    y[0] = 1
    for i in range(1, N*M):
        y[i] = y[i-1] + random.uniform(-1, 1)
    
    return y
    
#local maxima/minima of random_array M*1024 
#with distance between extrema > 3
#changing MAXs and MINs
#with T*std so that there are approx. 10 extrema in N=1024

def extrema(arr, M=3000):
    #arr = random_array(M)
    #flag variable
    found_min = False
    found = False
    #1/4 of std 
    x = 3
    T = 0.05
    Y_last = 0
    #distance between extrema >3
    dist = x+1
    Ymax = np.zeros(len(arr))
    Ymin = np.zeros(len(arr))
    #T*std
    D = T*np.std(arr[:-(len(arr)-1)] - arr[0:])
    for i in range(1, len(arr)-1):
        #print(D)
        #dist > 3 
        if dist > x:
            #decides to start with MAX or MIN at the beginning of the search
            if found_min or not found:
                if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                    #whether or not the difference between the last extremum and a new one is more than T.std
                    if np.abs(arr[i] - Y_last) >=D:
                        dist = 1
                        Ymax[i] = 1
                        found = True
                        found_min = False
                        Y_last = arr[i]
                    else:
                        i+=1
                    
            if not found_min or not found:
                if arr[i] < arr[i-1] and arr[i] < arr[i+1]:
                    if np.abs(arr[i] - Y_last) >= D:
                        dist = 1
                        Ymin[i] = 2
                        found = True
                        found_min = True
                        Y_last = arr[i]
                    else:
                        i+=1
        else:
            dist+=1
    #Y_mean = the number of extrema in N=1024       
    Y_mean = np.sum(np.r_[Ymax, Ymin])/M
    Y = Ymax+Ymin
    return Ymax, Ymin, Y_mean, Y

#M = 100
x = random_array(1)
X = x.reshape(-1, 1)
ymax, ymin, ymean, y = extrema(x, 1)


# Train - Test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)

# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)


#Scale
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#-----SMOTE-------
X_resampled, y_resampled = sm.fit_sample(X_train, y_train.ravel())


def get_class_distribution(obj):
    count_dict = {
        "no": 0,
        "max": 0,
        "min": 0
    }
    
    for i in obj:
        if i == 0: 
            count_dict['no'] += 1
        elif i == 1: 
            count_dict['max'] += 1
        elif i == 2: 
            count_dict['min'] += 1
                   
        else:
            print("Check classes.")
            
    return count_dict
    
    
print("Train:", get_class_distribution(y_resampled))
print("Test :", get_class_distribution(y_test))
print("Val  :", get_class_distribution(y_val))



fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,7))
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_resampled)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[0]).set_title('Class Distribution in Train Set')
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_val)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[1]).set_title('Class Distribution in Val Set')
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_test)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[2]).set_title('Class Distribution in Test Set')


class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def create_datasets():
    train_dataset = ClassifierDataset(
        torch.from_numpy(X_resampled).float(),
        torch.from_numpy(y_resampled).long()
    )
    val_dataset = ClassifierDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long()
    )
    test_dataset = ClassifierDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )
    return train_dataset, val_dataset, test_dataset
    
train_dataset, val_dataset, test_dataset = create_datasets()
