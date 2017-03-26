# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:38:23 2016

@author: Lukas
"""

import numpy as np
import matplotlib.pyplot as plt
#from mlp_toolkits.mplot3d import Axes3D

class som:
    def __init__(self):
        data = self.loadTrain()
        dim = 7
        self.data_num = 210
      #  for i in range(dim-1):
      #      data[:,i] = (data[:,i]/ (np.max(data[:,i])-np.min(data[:,i])))
        self.w = 20
        self.h = 15        
        W = np.random.random_sample((dim,self.w,self.h))
        eps = 100
        alpha_s = 1
        alpha_f = 0.5
        lambda_s = 5.0
        lambda_f = 0.01
        lambdaOld = lambda_s
        alphaOld = alpha_s
        estQEs = np.ones([1,1])
        adjustment = np.ones([1,1])
        alphas = np.ones([1,1])
        lambdas = np.ones([1,1])
        alpha = alpha_s
        for ep in range(eps):
            print(str(ep))
            adjust = 0.0
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
                        W[:,width,height] = W[:,width,height] + alpha * self.neighFunc(width,height,bestCoords[0],bestCoords[1],lamb) * (x-W[:,width,height])                        
            adjust = self.averageAdjust(wOldCoords,W)
            QE = QE/float(self.data_num)
            adjustment = np.append(adjustment,adjust)
            estQEs = np.append(estQEs, QE)
        estQEs = np.delete(estQEs,0,0)
        adjustment = np.delete(adjustment,0,0)
        alphas = np.delete(alphas,0,0)
        lambdas = np.delete(lambdas,0,0)
        self.plotclass(data[0:7,:],data[7,:],W)
        print(adjustment) 
        plt.style.use('ggplot')
        plt.figure()
        plt.hold(True)
        plt.plot(adjustment)
        plt.figure()
        plt.plot(estQEs)
        print(str(estQEs))
        plt.figure()
        plt.plot(alphas)
        plt.figure()
        plt.plot(lambdas)
        plt.hold(False)
        self.plotHeatMapsForCoord(W)
        self.plotUMap(W)
        
            
    def averageAdjust(self,wOld, wNew):
        dist = 0
        for w in range(self.w):
            for h in range(self.h):
                dist += np.linalg.norm(wNew[:,w,h] - wOld[:,w,h],2)
        return dist/float(self.w*self.h)
                
        
    def plotclass(self,data,cls,w):
        plt.figure()
        plt.hold(True)
        plt.axis([-0.5,self.w + 0.1,-0.5,self.h+0.1])
        for i in range(self.data_num):
            x = data[:,i]
            coorCN = self.findClosestNeuron(x, w)
            coorCN = coorCN[0]
            if cls[i] == 1:
                plt.plot(coorCN[0], coorCN[1], marker ='o', mec = 'lime', mew='2', mfc = 'None', ms = 12.5, label='Trieda 1')
            elif cls[i] == 2:
                plt.plot(coorCN[0], coorCN[1], marker ='x', mec = 'orange', mew='2', ms = 8)
            else:
                plt.plot(coorCN[0], coorCN[1], marker ='v', mec = 'w', mew='2', mfc = 'navy', ms = 12.5)
        plt.grid(True)
        plt.hold(False)
        plt.show()
        
    
    def plotDistanceHeatMap(self, w):
        distMapVert = np.zeros((self.w,self.h))
        for i in range(self.w):
            for j in range(self.h):                
                if j+1< self.h:
                    distMapVert[i,j] = self.distance(w[:,i,j],w[:,i,j+1])
                    if j > 0:
                        distMapVert[i,j] = (distMapVert[i,j]+self.distance(w[:,i,j],w[:,i,j-1]))/float(1)
                else:
                    distMapVert[i,j] = self.distance(w[:,i,j],w[:,i,j-1])
        distMapHor = np.zeros((self.w,self.h))
        for i in range(self.w):
            for j in range(self.h):
                if i+1< self.w:
                    distMapHor[i,j] = self.distance(w[:,i,j],w[:,i+1,j])
                    if i > 0:
                        distMapHor[i,j] = (distMapHor[i,j]+self.distance(w[:,i,j],w[:,i-1,j]))/float(1)
                else:
                    distMapHor[i,j] = self.distance(w[:,i,j],w[:,i-1,j])
        plt.figure()
        plt.style.use('grayscale')        
        plt.imshow(distMapVert.T)        
        plt.figure()
        plt.imshow(distMapHor.T)
        
    def plotUMap(self, w):
        distUMap = np.zeros((self.w,self.h))
        for i in range(self.w):
            for j in range(self.h):                
                if j+1< self.h:
                    distUMap[i,j] = self.distance(w[:,i,j],w[:,i,j+1])
                    if j > 0:
                        distUMap[i,j] = (distUMap[i,j]+self.distance(w[:,i,j],w[:,i,j-1]))/float(1)
                else:
                   distUMap[i,j] = self.distance(w[:,i,j],w[:,i,j-1])
                if i+1< self.w:
                    distUMap[i,j] = distUMap[i,j]+self.distance(w[:,i,j],w[:,i+1,j])
                    if i > 0:
                        distUMap[i,j] = (distUMap[i,j]+self.distance(w[:,i,j],w[:,i-1,j]))/float(1)
                else:
                    distUMap[i,j] = distUMap[i,j]+self.distance(w[:,i,j],w[:,i-1,j])
                
        plt.figure()
        plt.style.use('grayscale')        
        plt.imshow(distUMap.T,interpolation='none')
        
    def plotHeatMapsForCoord(self,w):
        for k in range(7):
            plt.figure()
            plt.imshow(w[k,:,:].T)
            
                
        
            
    def activity(self,x,w,sensitivity,bias):
        return bias*np.exp(-sensitivity*self.euclidDist(x,w)**2)
        
    def somActivity(self, ai, aj, bias, sensitivity):
        return self.activity(ai)/np.sum(self.activity(aj))
     
     
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
        
    
    
    def loadTrain(self):
        trainn = np.loadtxt('seeds.txt', dtype={'names': ('1', '2', '3', '4', '5', '6', '7', '8'),'formats': ('float64', 'float64', 'float64','float64','float64','float64','float64','float64')})
        train = np.zeros([210,8])
        for i in range(0,210):
            for j in range(0,8):
                train[i][j] = trainn[i][j]
        return train.T

som()