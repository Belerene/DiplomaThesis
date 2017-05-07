# -*- coding: utf-8 -*-
"""
Created on Wed Mar 1 16:38:23 2017

@author: Lukas

-- Use "1" signalig available neuron? Or "0" to signal neuron, that is aleardy in use?
-- How to detect the same episodes then? Compute SOM activity twice - once with once without the bias
-- and if the activity without (mapAvailable) bias is above threshold the episode is present in hippocampus?
-- Compute biases multiplicatively or additively?
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from matplotlib import animation as anim 
plt.ion()
class som:
    def __init__(self,width=20,height=15,numOfDataInstances=300,dimension=15,sensitivity=0.5,paramThreshold=4,paramRecency=1,hippocampus=True):
        self.forgotten = 0        
        self.sensitivity = sensitivity
        self.data_num = numOfDataInstances #Number of data instances
        self.dim = dimension
        self.w = width
        self.h = height    
        self.paramThreshold = paramThreshold
        #self.W = np.random.random_sample((dimension,self.w,self.h))
        self.W = np.random.rand(dimension,self.w,self.h)
        self.paramRecency = paramRecency
        self.mapAvailable = np.ones((self.w,self.h))
        self.mapRecency = np.ones((self.w,self.h))
        if True == hippocampus:        
            self.imRecency = plt.imshow(self.mapRecency,interpolation='nearest',origin='bottom',aspect='auto',vmin=0,vmax=1,cmap='gist_gray_r')       
            plt.pause(0.001)
        #self.mapAvailable[self.w/2,self.h/2] = 1
    
    def setDataInstances(self,dataInst):
        self.data_num = dataInst
        
    def get(self):
        return self.W
        
    def getEmotionMap(self,mixedParam,soloParam):
        eData = np.zeros((self.w,self.h))
        for width in range(self.w):
            for height in range(self.h):
                a = self.mapAvailable[width,height]
                if self.mapAvailable[width,height]!=True:
                    b=self.W[self.dim-1,width,height]
                    c=self.W[self.dim-2,width,height]
                    if self.W[self.dim-1,width,height] and self.W[self.dim-2,width,height]:
                        eData[width,height] = mixedParam
                    else:
                        if self.W[self.dim-1,width,height] or self.W[self.dim-2,width,height]:
                            eData[width,height] = soloParam
        return eData
    
    def getData2(self,prf,getPrf,emotions):
        empty = True
        data = np.zeros((self.dim,1,1))
        for width in range(self.w):
            for height in range(self.h):
                if self.mapAvailable[width,height]!=True:
                    """
                    Get a random number <0,1>, 0 with probability 50%-emotionWeight (for instance 20%), 1 with probability 50%+emotionWeight
                    emotionWeight is greater than zero is the memory has any emotion associated with it (higher if it has mixed emotion, typically)
                    """
                    if np.random.choice(2,1,p=[0.5-emotions[width,height],0.5+emotions[width,height]])[0]:
                        if empty:
                            data = self.W[:,width,height]
                            if getPrf:
                                activities = prf.findMostActiveNeuron(self.W[:,width,height],self.W,False)
                                data = np.vstack((data,prf.W[:,activities[0][0],activities[0][1]]))
                            empty = False
                        else:
                            data = np.vstack((data,self.W[:,width,height]))
                            if getPrf:
                                activities = prf.findMostActiveNeuron(self.W[:,width,height],self.W,False)
                                data = np.vstack((data,prf.W[:,activities[0][0],activities[0][1]]))
        return data
    def getData(self):
        mask = np.random.random_integers(0,1,(self.w,self.h))
        empty = True
        for width in range(self.w):
            for height in range(self.h):
                if mask[width,height]:
                    if empty:
                        data = self.W[:,width,height]               
                        empty = False
                    else:
                        data = np.vstack((data,self.W[:,width,height]))
        return data
                        
        

    def trainDay(self,alpha,lamb,epochs,data,paramForget,hippocampus):
        """
        Trains the som - alpha:learning rate, lamb:gaussian width of neigh.func., epochs:number of epochs, paramForget: forgetting parameter for recency (1=no decay)
        """  
        self.paramForget = paramForget
        self.bias = np.random.random_sample((self.w,self.h))
        eps = epochs
        lambda_s = lamb
        for ep in range(eps):
            QE = 0.0
            #lamb = lambda_s*np.exp(-float(ep)/float(eps))         
            #alpha = alpha_s*(alpha_f/alpha_s)**(float(ep)/float(eps)) Tato alpha sa pouzivala
            
            for i in range(self.data_num):
                x = data[i,:]
                a = self.mapAvailable
                b = self.W
                #bestCoords = self.findClosestNeuron(x,self.W)  
                """
                Find out if the episode is already present in hippocampus by exceeding threshold T
                T = (1/(w*h))*paramThreshold
                """
                activities = self.findMostActiveNeuron(x,self.W,useBias=False)  
                d = activities[2].mean()
                e = activities[2].mean()*self.paramThreshold
                if activities[1] < activities[2].mean()*self.paramThreshold or (self.w*self.h)==np.count_nonzero(self.mapAvailable):
                    activities = self.findMostActiveNeuron(x,self.W,useBias=True)
                    self.mapAvailable[activities[0][0],activities[0][1]] = 0
                act = activities[1]                
                bestCoords = activities[0]
                mapActivity = activities[2]
                self.updateRecency(mapActivity)
                self.mapRecency[bestCoords[0],bestCoords[1]] = 1
                self.forgetRecency
                QE = QE+act
                for width in range(self.w):
                    for height in range(self.h):
                        
                        ww = self.W
                        www = self.W[:,width,height]
                        """
                        Weight updating - formula (8)
                        """
                        self.W[:,width,height] = self.W[:,width,height] + alpha * self.neighFunc(width,height,bestCoords[0],bestCoords[1],lamb) * (x-self.W[:,width,height]) 
                        www = self.W[:,width,height]
                if True == hippocampus:
                    self.imRecency.set_array(self.mapRecency)
                    plt.draw()
                    plt.pause(0.000001)
            QE = QE/float(self.data_num)
            activities = self.findMostActiveNeuron(x,self.W,useBias=False)  
            mapActivity = activities[2]    
            
            
    def averageAdjust(self,wOld, wNew):
        dist = 0
        for w in range(self.w):
            for h in range(self.h):
                dist += np.linalg.norm(wNew[:,w,h] - wOld[:,w,h],2)
        return dist/float(self.w*self.h)
                
    """
    Updates the recency map - decay is determined by paramRecency (1 = no decay)
    """
    def updateRecency(self,mapActivity):
        for w in range(self.w):
            for h in range(self.h):
                self.mapRecency[w,h] = self.mapRecency[w,h]*self.paramRecency
      

    """
    Updates the map of available neurons and the SOM based on recency and forgetting threshold
    """          
    def forgetRecency(self):
        for w in range(self.w):
            for h in range(self.h):
                if self.mapRecency[w,h] < self.paramForget:
                    self.mapAvailable[w,h] = 1
                    self.W[:,w,h] = np.random.rand(self.dim)
                    self.forgotten = self.forgotten+1
    
    
    """
    #Returns activity of a single neuron - formula (4)
    """        
    def activity(self,x,w,bias):
        return bias*np.exp(-self.sensitivity*self.euclidDist(x,w)**2)
     
    """
    Returns activies for every neuron - formula (5)
    """ 
    def somActivity(self, x, useBias):
        Aj = 0
        for width in range(self.w):
            for height in range(self.h):
                #tmp = self.activity(x,self.W[:,width,height],self.biases(width,height),useBias)
                if useBias:
                    Aj = Aj+self.activity(x,self.W[:,width,height],self.biases(width,height))
                else:
                    Aj = Aj+self.activity(x,self.W[:,width,height],1)
        Ai = np.random.random_sample((self.w,self.h))
        for width in range(self.w):
            for height in range(self.h):
                #a = self.activity(x,self.W[:,width,height],self.biases(width,height),useBias)
                #b = np.sum(self.activity(x,self.W[:,width,height],self.biases(width,height),useBias))
                if useBias:
                    Ai[width,height] = self.activity(x,self.W[:,width,height],self.biases(width,height))/Aj
                else:
                    Ai[width,height] = self.activity(x,self.W[:,width,height],1)/Aj
        return Ai
     
    """
    Returns bias for specific neuron
    """
    def biases(self,w,h):
        a= (self.mapAvailable[w,h]*self.bias[w,h])/1
        return a
     
    
    def reconstructInput2(self,x):
        activities = self.findMostActiveNeuron(x,self.W,useBias=False)
        bestCoords = activities[0]
        y = self.W[:,bestCoords[0],bestCoords[1]]
        return y
    
    """
    Returns block-wise reconstructed input - formula (7)
    """
    def reconstructInput(self,t,mixing):
        Ai = self.somActivity(t,True)
        h = len(t)
        y = np.zeros((h))
        for k in range(h):
            summed = 0
            for width in range(self.w):
                for height in range(self.h):
                    summed = Ai[width,height] * self.W[:,width,height]  
            o = mixing*summed
            y[k] = mixing * summed[k]
        return y
        
        
    def dist(self,x,w,b):
        return np.sum(b * self.euclidDist(x,w))
        
        
    def neighFunc(self, m1,n1,m2,n2, lamb):
        w1 = [m1,n1]
        w2 = [m2,n2]
        d = np.linalg.norm(np.subtract(w1,w2),2)
        res = np.exp(-d**2/float(lamb)**2)
        return res
        
    def euclidDist(self,x,y):
        d = np.linalg.norm(x-y,2)
        return d
    
    def findMostActiveNeuron(self,x,w,useBias):
        activity = -9999999
        coords = [-1,-1]
        somAct = self.somActivity(x,useBias)
        for nx in range(self.w):
            for ny in range(self.h):
                a = somAct[nx,ny]
                if activity<somAct[nx,ny]:
                    activity = somAct[nx,ny]
                    coords = [nx,ny]
        return [coords,activity,somAct]
    
    def findClosestNeuron(self,x,w):
        distance = 9999999
        coords = [-1,-1]
        for nx in range(self.w):
            for ny in range(self.h):
                d = np.linalg.norm(x-w[:,nx,ny])
                if d<distance:
                    distance = d
                    coords = [nx,ny]
        return [coords,distance]
        
    def drawRecency(self):
        plt.ion()
        self.figRecency, self.axRecency = plt.subplots()
        self.imRecency = self.axRecency.imshow(self.mapRecency,
                                                    interpolation='nearest',
                                                    origin='bottom',
                                                    aspect='auto',
                                                    vmin=0,
                                                    vmax=1,
                                                    cmap='jet')
        self.cbRecency = plt.colorbar(self.imRecency)
        plt.show()
        #plt.draw()
        #self.pcRecency = self.axRecency[1].pcolor(self.mapRecency)
        #self.cb2Recency = plt.colorbar(self.pcRecency)
       
    
    def redrawRecency(self):
        self.imRecency.set_data(self.mapRecency)
        plt.show()
        plt.draw()


som()
