# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:38:23 2016

@author: Lukas
"""

import numpy as np
import matplotlib.pyplot as plt

class som:
    def __init__(self):
        data = self.loadTrain()
        self.data_num = 210
        self.w = 20
        self.h = 15        
        W = np.random.random_sample((self.w,self.h))
        Ai = np.random.random_sample((self.w,self.h))
        bias = np.random.random_sample((self.w,self.h))
        eps = 100
        alpha_s = 1
        alpha_f = 0.5
        lambda_s = 5.0
        lambda_f = 0.01
        lambdaOld = lambda_s
        alphaOld = alpha_s
        estQEs = np.ones([1,1])
        alphas = np.ones([1,1])
        lambdas = np.ones([1,1])
        alpha = alpha_s
        for ep in range(eps):
            print(str(ep))
            QE = 0.0
            lamb = lambda_s*np.exp(-float(ep)/float(eps))
            #alpha = alpha_s*np.exp(-float(ep)/float(eps))
            #lamb = lambda_s*(lambda_f/lambda_s)**(float(ep)/float(eps))
            alpha = alpha_s*(alpha_f/alpha_s)**(float(ep)/float(eps))
            alphas = np.append(alphas,alphaOld-alpha)##
            alphaOld = alpha
            lambdas = np.append(lambdas,lambdaOld-lamb)##
            lambdaOld = lamb
            data = np.random.permutation(data.T).T
            wOldCoords = np.copy(W)
            for i in range(self.data_num):
                x = data[0:7,i]
                bestCoords = self.findClosestNeuron(x,W)
                dist = bestCoords[1]                
                bestCoords = bestCoords[0]
                QE = QE+dist
                for width in range(self.w):
                    for height in range(self.h):
                        W[width,height] = W[width,height] + alpha * self.neighFunc(width,height,bestCoords[0],bestCoords[1],lamb) * (x-W[width,height])                        
            QE = QE/float(self.data_num)
            estQEs = np.append(estQEs, QE)
        estQEs = np.delete(estQEs,0,0)
        alphas = np.delete(alphas,0,0)
        lambdas = np.delete(lambdas,0,0)
        
            
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
