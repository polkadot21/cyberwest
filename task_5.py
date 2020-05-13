import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from google.colab import drive
drive.mount('/content/gdrive')


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

def extrema(arr):
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
          
    
    Y = Ymax + Ymin
    return Ymax, Ymin, Y

#M = 300
x = random_array(60)
X = x.reshape(-1, 1)
ymax, ymin, y, num = extrema(x)

# -------- Train - Test-------
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)

# ---------Split train into train-val--------
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)


#-----Scale-----
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


#-----SMOTE-------
X_resampled, y_resampled = sm.fit_sample(X_train, y_train.ravel())


#class_dist for each class
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






class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


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



class_count = [i for i in get_class_distribution(y_resampled).values()]
class_weights = 1./torch.tensor(class_count, dtype=torch.float) 

class_weights_all = class_weights[target_list]

weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)

#params

EPOCHS = 75
BATCH_SIZE = 16
LEARNING_RATE = 0.0007

NUM_FEATURES = 1
NUM_CLASSES = 3


class Min_Max_Classification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(Min_Max_Classification, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, 16)
        #self.layer_2 = nn.Linear(64, 32)
        #self.layer_3 = nn.Linear(32, 16)
        self.layer_out = nn.Linear(16, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(16)
        #self.batchnorm2 = nn.BatchNorm1d(32)
        #self.batchnorm3 = nn.BatchNorm1d(16)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        #x = self.layer_2(x)
        #x = self.batchnorm2(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        
        #x = self.layer_3(x)
        #x = self.batchnorm3(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x


#loader

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)

class Min_Max_Classification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(Min_Max_Classification, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, 16)
        #self.layer_2 = nn.Linear(64, 32)
        #self.layer_3 = nn.Linear(32, 16)
        self.layer_out = nn.Linear(16, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(16)
        #self.batchnorm2 = nn.BatchNorm1d(32)
        #self.batchnorm3 = nn.BatchNorm1d(16)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        #x = self.layer_2(x)
        #x = self.batchnorm2(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        
        #x = self.layer_3(x)
        #x = self.batchnorm3(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x

#cpu and gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#set model, 
#gpu/cpu
model = Min_Max_Classification(
    num_feature = NUM_FEATURES,
    num_class=NUM_CLASSES)
model.to(device)

print(model)

#multiclass critretion
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


model_save_name = 'classifier.pt'
path = F"/content/gdrive/My Drive/{model_save_name}"
model.load_state_dict(torch.load(path))


#make a random sample of length M*N
#print confusion matrix
def out_of_sample(M):
    x_sample = random_array(M)
    ymax_sample, ymin_sample, y_sample , num = extrema(x_sample)
    
    X_sample = x_sample.reshape(-1, 1)
    X_sample = scaler.transform(X_sample)
    
    sample_dataset = ClassifierDataset(torch.from_numpy(X_sample).float(),
                                    torch.from_numpy(y_sample).long())
    
    sample_loader = DataLoader(dataset=sample_dataset, batch_size=1)
    
    y_pred_list_sample = []

    with torch.no_grad():
        new_model.eval()
        for X_batch, _ in sample_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
            y_pred_list_sample.append(y_pred_tags.cpu().numpy())
            
    print(confusion_matrix(y_sample, y_pred_list_sample))
    return np.asarray(x_sample), np.asarray(y_pred_list_sample), np.asarray(y_sample)

out_of_sample(10)

#plot sample
#plot real extrema (blue)
#Plot predicted extrema (red)

def plot_out_of_sample(M, n=1024):
    #create a sample
    x_sample, y_pred, y_real = out_of_sample(M)
    #reshape predicted
    y_pred = np.reshape(y_pred, len(y_pred))
    t = np.linspace(0, n)
    y1 = x_sample
    y2 = y_real
    y3 = y_pred
    
    
    fig = go.Figure()

    # Add traces
    
                  
    fig.add_trace(go.Scatter(x=t, y=y1,
                    mode='lines+markers',
                    name='array_sample',
                    marker_color='rgb(154, 221, 43)'))
                  
    fig.add_trace(go.Scatter(x=t, y=y2,
                        mode='markers',
                        name='Y_real',
                        marker_color='rgb(0, 191, 255)'))
    
    fig.add_trace(go.Scatter(x=t, y=y3,
                    mode='markers',
                    name='Y_pred',
                    marker_color='rgb(253, 68, 43)'))


    return fig.show()

plot_out_of_sample(10)
