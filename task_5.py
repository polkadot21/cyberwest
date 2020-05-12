import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm

from matplotlib import pyplot as plt

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 33)

import plotly.express as px
import plotly.graph_objects as go




#random array, x[0] = 1, 
#x[i] = x[i-1] + random.uniform(-1, 1)
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
    T = 0.16
    Y_last = 0
    #distance between extrema >3
    dist = x+1
    Ymax = np.zeros(len(arr))
    Ymin = np.zeros(len(arr))
    #T*std
    #D = T*np.std(arr[:-(len(arr)-1)] - arr[0:])
    D = 0
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
                        D = T*np.std(arr[:-(len(arr)-1)] - arr[0:])
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
                        D = T*np.std(arr[:-(len(arr)-1)] - arr[0:])
                    else:
                        i+=1
        else:
            dist+=1
    #Y_mean = the number of extrema in N=1024       
    Y = Ymax+Ymin
    return Ymax, Ymin, Y
    
    
#make a random sample of length M*N
#print confusion matrix
def out_of_sample(M):
    x_sample = random_array(M)
    ymax_sample, ymin_sample, ymean_sample , y_sample = extrema(x_sample, M)
    
    X_sample = x_sample.reshape(-1, 1)
    X_sample = scaler.transform(X_sample)
    
    sample_dataset = ClassifierDataset(torch.from_numpy(X_sample).float(),
                                    torch.from_numpy(y_sample).long())
    
    sample_loader = DataLoader(dataset=sample_dataset, batch_size=1)
    
    y_pred_list_sample = []

    with torch.no_grad():
        model.eval()
        for X_batch, _ in sample_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
            y_pred_list_sample.append(y_pred_tags.cpu().numpy())
            
    print(confusion_matrix(y_sample, y_pred_list_sample))
    return np.asarray(x_sample), np.asarray(y_pred_list_sample), np.asarray(y_sample)
    
    
    
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