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
    def __init__(self,width=20,height=15,numOfDataInstances=300,dimension=15,sensitivity=0.5,paramThreshold=4,paramRecency=1,paramForgetThr=0.4,hippocampus=True):
        self.forgotten = 0 
        self.consolidated = 0
        self.forgunconsolidated = 0
        self.forgunconsolidatedemotions = 0
        self.mapConsolidated = np.zeros((width,height))
        self.mapConsolidatedEmotions = np.zeros((width,height))
        self.remembered = 0
        self.paramForgetThr = paramForgetThr
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
        self.mapRecency = np.zeros((self.w,self.h))
        self.mapActivities = np.ones((self.w,self.h,1))
        self.fig, self.axes = plt.subplots(1,2)
        if False == hippocampus:        
            #self.imRecency = plt.imshow(self.W[0,:,:],interpolation='nearest',origin='bottom',aspect='auto',vmin=0,vmax=1,cmap='gist_gray_r')       
            self.c = self.axes[0].pcolor(np.array(np.zeros((self.w,self.h))+0.3),edgecolors='k',linewidths=2,cmap='Blues', vmin=0,vmax=1)
            plt.pause(0.001)
            self.d = self.axes[1].pcolor(np.array(np.zeros((self.w,self.h))),edgecolors='k',linewidths=2,vmin=0,vmax=1,cmap='gist_gray_r') 
                
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
                            self.consolidated = self.consolidated+1
                            self.mapConsolidated[width,height] = 1
                            if self.W[17,width,height] or self.W[16,width,height]:
                                self.mapConsolidatedEmotions[width,height] = 1
                            if getPrf:
                                activities = prf.findMostActiveNeuron(self.W[:,width,height],self.W,False)
                                data = np.vstack((data,prf.W[:,activities[0][0],activities[0][1]]))
                            empty = False
                        else:
                            self.consolidated = self.consolidated+1
                            self.mapConsolidated[width,height] = 1
                            if self.W[17,width,height] or self.W[16,width,height]:
                                self.mapConsolidatedEmotions[width,height] = 1
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
            activitiesMap = np.zeros((self.w,self.h))
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
                if hippocampus:
                    #if activities[1] < activities[2].mean()*self.paramThreshold or (self.w*self.h)==np.count_nonzero(self.mapAvailable):
                    aaact = self.mapAvailable
                    if activities[1] != 1 or (self.w*self.h)==np.count_nonzero(self.mapAvailable):
                        activities = self.findMostActiveNeuron(x,self.W,useBias=True)
                        self.mapAvailable[activities[0][0],activities[0][1]] = 0
                        self.remembered = self.remembered+1
                act = activities[1]                
                bestCoords = activities[0]
                mapActivity = activities[2]
                activitiesMap = activitiesMap+mapActivity
                self.updateRecency(mapActivity)
                self.mapRecency[bestCoords[0],bestCoords[1]] = 1
                self.forgetRecency()
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
                #print("...training data: " + str(i) + "...")
            if self.mapActivities.shape[2]==1:
                c = self.data_num
                a = self.mapActivities
                self.mapActivities = np.append(self.mapActivities,np.atleast_3d(activitiesMap/self.data_num),axis=2)
            else:
                a = self.mapActivities
                b = activitiesMap/self.data_num
                c = self.data_num
                self.mapActivities = np.append(self.mapActivities,np.atleast_3d(activitiesMap/self.data_num),axis=2)
            if False == hippocampus:
                #c = plt.pcolor(self.mapRecency,edgecolors='k',linewidths=2,cmap='RdBu', vmin=0,vmax=1) 
                #self.imRecency.set_array(self.W[0,:,:])
                #self.c.set_array()
                ax = self.axes[0]
                self.axes[0].clear()
                ax.clear()
                self.c.set_array((activitiesMap/self.data_num).flatten())
                self.showValues(self.c)
                self.d.set_array((activitiesMap/self.data_num).flatten())
                #c = plt.pcolor(self.W[0,:,:],edgecolors='k',linewidths=2,cmap='RdBu', vmin=0,vmax=1)
                
                #plt.colorbar(c)
                #plt.draw()
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
    Updates the recency map - decay is determined by paramRecency (0 = no decay)
    """
    def updateRecency(self,mapActivity):
        for w in range(self.w):
            for h in range(self.h):
                self.mapRecency[w,h] = self.mapRecency[w,h]-self.paramRecency
      

    """
    Updates the map of available neurons and the SOM based on recency and forgetting threshold
    """          
    def forgetRecency(self):
        for w in range(self.w):
            for h in range(self.h):
                a = self.mapRecency
                if self.mapRecency[w,h] < self.paramForgetThr and self.mapAvailable[w,h]==0:
                    self.mapAvailable[w,h] = 1
                    self.W[:,w,h] = np.random.rand(self.dim)
                    self.forgotten = self.forgotten+1
                    self.remembered = self.remembered-1
                    if self.mapConsolidated[w,h]:
                        self.forgunconsolidated = self.forgunconsolidated+1
                        self.mapConsolidated[w,h] = 0
                    if self.mapConsolidatedEmotions[w,h]:
                        self.forgunconsolidatedemotions = self.forgunconsolidatedemotions+1
                        self.mapConsolidatedEmotions[w,h] = 0
    
    
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
                    #Ai[width,height] = self.activity(x,self.W[:,width,height],self.biases(width,height))/Aj
                    Ai[width,height] = self.activity(x,self.W[:,width,height],self.biases(width,height))
                else:
                    #Ai[width,height] = self.activity(x,self.W[:,width,height],1)/Aj
                    Ai[width,height] = self.activity(x,self.W[:,width,height],1)
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
                if activity<somAct[nx,ny]:
                    activity = somAct[nx,ny]
                    coords = [nx,ny]
        return [coords,activity,somAct]
    
        
    def showValues(self,pc, fmt="%.2f", **kw):
        from itertools import izip
        pc.update_scalarmappable()
        emotions, labels, distances = self.getLabels()
        ax = self.axes[0]
        for p, color, val1, val2, txt in izip(pc.get_paths(), pc.get_facecolors(), emotions[0,:,:].flatten(), emotions[1,:,:].flatten(), labels.flatten()):
            x,y = p.vertices[:-2,:].mean(0)
            if val1>0.5 and val2>0.5:
                color=(1*val2,1*val1,0)
            else:
                if val1>0.5:
                    color = (0,1*val1,0)
                if val2>0.5:
                    color = (1*val2,0,0)
                if val1<=0.5 and val2 <=0.5:
                    color = (1*(val1*val2)/2,1*(val1*val2)/2,1*(val1*val2)/2)
            ax.text(x,y, txt, ha="center",va="center", color=color, **kw)
    def getLabels(self):
        eps, appMap = self.getApproxMap()
        result = np.chararray((self.w,self.h),itemsize=3)
        emotions = eps[16:,:,:]
        for w in range(self.w):
            for h in range(self.h):
                v = eps[:,w,h]
                agent = str(v[0:4].argmax())
                action = str(v[4:8].argmax())
                patient = str(v[8:16].argmax())
                result[w,h] = agent+action+patient
        return (emotions, result, appMap)
        
    def getApproxMap(self):
        result = np.zeros((self.w,self.h))
        episodes = np.zeros((self.dim,self.w,self.h))
        for w in range(self.w):
            for h in range(self.h):
                vect = self.W[:,w,h]
                episode = self.getEpisode(vect)
                result[w,h] = self.difference(vect,episode)
                episodes[:,w,h] = episode
        return (episodes,result)
                
    
    def getEpisode(self, v):
        agent = np.zeros(4)
        action = np.zeros(4)
        patient = np.zeros(8)
        agent[v[0:4].argmax()] = 1
        action[v[4:8].argmax()] = 1
        patient[v[8:16].argmax()] = 1
        result = np.hstack((np.hstack((np.hstack((agent,action)),patient)),v[16:]))
        return result
        
                
    def difference(self, im1, im2):
        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
        diff = np.sum(abs(im1-im2))/im1.shape[0]
        return diff


som()
