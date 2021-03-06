import numpy as np
from matplotlib import pyplot as plt
import random
import plotly.express as px
import plotly.graph_objects as go


#make a random array, x[0] = 1, x[i] = x[i-1] + random.uniform(-1, 1)
def random_array(M=1000):
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
#returns Ymax, Ymin, number of extrema per N and Y (all extrema)

def extrema(arr):
    #arr = random_array(M)
    #flag variable
    found_min = False
    found = False
    #1/4 of std 
    x = 3
    T = 0.014
    Y_last = 0
    #distance between extrema >3
    dist = x+1
    Ymax = np.zeros(len(arr))
    Ymin = np.zeros(len(arr))
    #T*std
    D = T*np.std(arr[:-(len(arr)-1)] - arr[0:])
    for i in range(1, len(arr)-1):
        #dist > 3 
        if dist > x:
            #decides to start with MAX or MIN at the beginning of the search
            if found_min or not found:
                if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                    #whether or not the difference between the last extremum and a new one is more than T.std
                    if np.abs(arr[i] - Y_last) >= D:
                        dist = 1
                        Ymax[i] = arr[i]
                        found = True
                        found_min = False
                        Y_last = arr[i]
                        


                    else:
                        i+=1
                    
            if not found_min or not found:
                if arr[i] < arr[i-1] and arr[i] < arr[i+1]:
                    if np.abs(arr[i] - Y_last) >= D:
                        dist = 1
                        Ymin[i] = arr[i]
                        found = True
                        found_min = True
                        Y_last = arr[i]
                        
                    else:
                        i+=1
        else:
            dist+=1
    #Y = the number of extrema in N=1024       
    #Y = (Ymax - Ymin)/1000
    return Ymax, Ymin, #Y
    
x = random_array()
ymax, ymin = extrema(x)

print('len of x:' ,len(x))
#print('num of max:', np.sum(ymax))
#print('num of min:' , np.sum(-ymin))
#print('num of extrema per N=1024:', np.sum(y))





#create random sample array from initial array 
def create_random_sample(arr):
    
    k = random.randint(10, 100)
    p = random.randint(101, 1000)
    for i in range(k, p):
        arr_sample = arr[k:p]
    return arr_sample
    
    
#the functition plots a random sample array and shows local maxima(red) and local minima(blue)

def plot_random_sample(arr, n=1024):
    #create a random sample
    arr_sample = create_random_sample(arr)
    #create Ymax and Ymin for random_sample
    Ymax_sample, Ymin_sample = extrema(arr_sample)
    #x-line for lenght = n
    t = np.linspace(0, n)
    
    for i in range(1, len(Ymax_sample)):
        if Ymax_sample[i] == 0:
            Ymax_sample[i] = np.nan
    for j in range(1, len(Ymin_sample)):
        if Ymin_sample[j] == 0:
            Ymin_sample[j] = np.nan

    y1 = Ymax_sample
    y2 = arr_sample
    y3 = Ymin_sample
    fig = go.Figure()

    # Add traces
   
    fig.add_trace(go.Scatter(x=t, y=y2,
                    mode='lines+markers',
                    name='array_sample',
                    marker_color='rgb(154, 221, 43)'))
    
    fig.add_trace(go.Scatter(x=t, y=y1,
                    mode='markers',
                    name='Ymax',
                    marker_color='rgb(253, 68, 43)'))
                  
    fig.add_trace(go.Scatter(x=t, y=y3,
                        mode='markers',
                        name='Ymin',
                        marker_color='rgb(0, 191, 255)'))


    fig.show()
    
    
plot_random_sample(x)
plot_random_sample(x)
plot_random_sample(x)
plot_random_sample(x)
