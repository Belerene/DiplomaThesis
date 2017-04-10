# -*- coding: utf-8 -*-
"""
Created on Wed Mar 1 16:38:23 2017

@author: Lukas
"""

import numpy as np
import matplotlib.pyplot as plt

class som:
    def __init__(self,width=20,height=15,numOfDataInstances=210):
        #data = self.loadTrain()
        self.data_num = numOfDataInstances #Number of data instances
        self.w = width
        self.h = height        
        self.W = np.random.random_sample((self.w,self.h))
        self.Ai = np.random.random_sample((self.w,self.h))
        self.bias = np.random.random_sample((self.w,self.h))
        
     
    def get(self):
        return self.w
        

    def train(self,alpha,lamb,epochs,data):
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

            data = np.random.permutation(data.T).T
            for i in range(self.data_num):
                x = data[0:7,i]
                bestCoords = self.findClosestNeuron(x,self.W)
                dist = bestCoords[1]                
                bestCoords = bestCoords[0]
                QE = QE+dist
                for width in range(self.w):
                    for height in range(self.h):
                        self.W[width,height] = self.W[width,height] + alpha * self.neighFunc(width,height,bestCoords[0],bestCoords[1],lamb) * (x-self.W[width,height])                        
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
    def activity(self,x,w,sensitivity,bias):
        return bias*np.exp(-sensitivity*self.euclidDist(x,w)**2)
     
     
    """
    Returns activies for every neuron - formula (5)
    """ 
    def somActivity(self, x, W, sensitivity, bias):
        aj = 0
        for width in range(self.w):
            for height in range(self.h):
                aj = aj+self.activity(x,W[width,height],sensitivity,bias[width,height])    
        Ai = np.random.random_sample((self.w,self.h))
        for width in range(self.w):
            for height in range(self.h):
                Ai[width,height] = self.activity(x,W[width,height],sensitivity,bias[width,height])/aj
        return Ai
     
     
     
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
