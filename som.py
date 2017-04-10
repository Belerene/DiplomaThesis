# -*- coding: utf-8 -*-
"""
Created on Wed Mar 1 16:38:23 2017

@author: Lukas
"""

import numpy as np
import matplotlib.pyplot as plt

class som:
    def __init__(self,width=20,height=15,numOfDataInstances=300,dimension=18,sensitivity=0.5):
        #data = self.loadTrain()
        self.sensitivity = sensitivity
        self.data_num = numOfDataInstances #Number of data instances
        self.dim = dimension
        self.w = width
        self.h = height        
        self.W = np.random.random_sample((dimension,self.w,self.h))
        #self.Ai = np.random.random_sample((self.w,self.h))
        self.bias = np.random.random_sample((self.w,self.h))
        self.mapAssigned = np.zeros((self.w,self.h))
        self.mapAssigned[self.w/2,self.h/2] = 1
     
    def get(self):
        return self.W
        

    def trainDay(self,alpha,lamb,epochs,data):
        """
        Trains the som - alpha:learning rate, lamb:gaussian width of neigh.func., epochs:number of epochs
        """  
        eps = epochs
        lambda_s = lamb
        for ep in range(eps):
            print(str(ep))
            QE = 0.0
            lamb = lambda_s*np.exp(-float(ep)/float(eps))         
            #alpha = alpha_s*(alpha_f/alpha_s)**(float(ep)/float(eps)) Tato alpha sa pouzivala

            #data = np.random.permutation(data.T).T
            
            for i in range(self.data_num):
                x = data[i,:]
                a = self.mapAssigned
                bestCoords = self.findClosestNeuron(x,self.W)
                bestCoords = self.findMostActiveNeuron(x,self.W)
                dist = bestCoords[1]                
                bestCoords = bestCoords[0]
                self.mapAssigned[bestCoords[0],bestCoords[1]] = 1
                QE = QE+dist
                for width in range(self.w):
                    for height in range(self.h):
                        ww = self.W
                        www = self.W[:,width,height]
                        self.W[:,width,height] = self.W[:,width,height] + alpha * self.neighFunc(width,height,bestCoords[0],bestCoords[1],lamb) * (x-self.W[:,width,height]) 
                        www = self.W[:,width,height]
            QE = QE/float(self.data_num)
            
    def averageAdjust(self,wOld, wNew):
        dist = 0
        for w in range(self.w):
            for h in range(self.h):
                dist += np.linalg.norm(wNew[:,w,h] - wOld[:,w,h],2)
        return dist/float(self.w*self.h)
                

    """
    Returns activity of a single neuron - formula (4)
    """        
    def activity(self,x,w,bias):
        return bias*np.exp(-self.sensitivity*self.euclidDist(x,w)**2)
     
     
    """
    Returns activies for every neuron - formula (5)
    """ 
    def somActivity(self, x):
        aj = 0
        for width in range(self.w):
            for height in range(self.h):
                aj = aj+np.sum(self.activity(x,self.W[:,width,height],self.biases(width,height)))    
        Ai = np.random.random_sample((self.w,self.h))
        for width in range(self.w):
            for height in range(self.h):
                b = np.sum(self.activity(x,self.W[:,width,height],self.biases(width,height)))
                Ai[width,height] = np.sum(self.activity(x,self.W[:,width,height],self.biases(width,height)))/aj
        return Ai
     
    
    def biases(self,w,h):
        a= (self.mapAssigned[w,h]+self.bias[w,h])/2
        return a
     
    """
    Returns block-wise reconstructed input - formula (7) 
    
    It should be implemented as Wjk (weight corresponding to a block), but is implemented as Wj
    """
    def reconstructInput(self,Ai,W,t):
        y = np.random.random_sample((len(t)))
        for k in range(t):
            summed = 0
            for width in range(self.w):
                for height in range(self.h):
                    summed = Ai[width,height] * W[width,height]  
            y[k] = t[k] * summed
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
    
    def findMostActiveNeuron(self,x,w):
        activity = -9999999
        coords = [-1,-1]
        somAct = self.somActivity(x)
        for nx in range(self.w):
            for ny in range(self.h):
                a = somAct[nx,ny]
                if activity<somAct[nx,ny]:
                    activity = somAct[nx,ny]
                    coords = [nx,ny]
        return [coords,activity]
    
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
        
    
    


som()
