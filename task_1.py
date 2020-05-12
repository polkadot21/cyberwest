import numpy as np
from matplotlib import pyplot as plt
import random
import plotly.express as px
import plotly.graph_objects as go


#make a random array, x[0] = 1, x[i] = x[i-1] + random.uniform(-1, 1)
def random_array(M=3000):
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

def extrema(arr, M=3000):
    #arr = random_array(M)
    #flag variable
    found_min = False
    found = False
    #1/4 of std 
    x = 3
    T = 0.015
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
    #Y = the number of extrema in N=1024       
    Y = (Ymax+Ymin/2)/3000
    return Ymax, Ymin, Y
    
x = random_array()
ymax, ymin, y= extrema(x, 3000)

print('len of x:' ,len(x))
print('num of max:', np.sum(ymax))
print('num of min:' , np.sum(ymin)/2)
print('num of extrema per N=1024:', np.sum(y))