# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 21:56:36 2017

@author: Lukas
"""

import som as s
import datasetGenerator as dg
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython import display

class main:
    def __init__(self):
        days = 40
        mixedEmotionsParam =0.2
        singleEmotionParam = 0.2
        dat = dg.datasetGenerator('epNumEmotions.txt')
        hippocampus = s.som(30,30,40,18,1.0,150,0.02,0.2,True)
        prefrontal =  s.som(15,15,20,18,0.4,100,0.0,1,False)
        #fig = plt.figure('matrix test')
        #ax = fig.add_subplot(111)
        #im = ax.imshow(hippocampus.mapRecency,interpolation='nearest',origin='bottom',aspect='auto',vmin=0,vmax=1,cmap='gist_gray_r')
        #plt.show(block=False)       
        ##hippocampus.drawRecency()
        
        for j in range(days):
            print("...training day: " + str(j) + "...")
            data = dat.getEpisodes(40)
            if j==0:
                usedData = data
            else:
                usedData = np.vstack((usedData,data))
            #data = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[0,0,0],[1,1,1],[1,0,1],[1,1,0],[0,0,1],[1,1,1],[1,0,0]])
            #data = np.vstack((data,data))
            #data = np.vstack((data,data))            
            hippocampus.trainDay(alpha=1,lamb=0.01,epochs=1,data=data,paramForget=0.05,hippocampus=True)
            aaa = hippocampus.mapAvailable
            #d = (15*15)-np.count_nonzero(aaa)
            emotions = hippocampus.getEmotionMap(mixedEmotionsParam,singleEmotionParam)
            if j>0:
                getPrefrontal = True
                testData = hippocampus.getData2(prefrontal,getPrefrontal,emotions)
            else:
                getPrefrontal = False
                testData = hippocampus.getData2(prefrontal,getPrefrontal,emotions)
            prefrontal.setDataInstances(testData.shape[0])
            prefrontal.trainDay(alpha=0.2,lamb=0.7,epochs=5,data=testData,paramForget=1,hippocampus=False)
            a1 = prefrontal.reconstructInput2(np.array([1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]))
            a2 = hippocampus.reconstructInput2(np.array([1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]))
            d = 5
            aaa = hippocampus.mapAvailable
            dd = hippocampus.forgotten
            ddd = hippocampus.remembered
            c, usedData = self.countInstances(usedData)
            d = (30*30)-np.count_nonzero(aaa)
            #print("...accuracy: " + str(float(c)/float(d)) + "...")
            b = 8
            print("...forgotten: " + str(dd) + "...")
            d = hippocampus.consolidated
            print("...consolidated: " + str(d) + "...")
            ddd=hippocampus.forgunconsolidated
            dddd = hippocampus.forgunconsolidatedemotions
            print("...forgotten and consolidated: " + str(ddd) + "...")
            print("...forgotten and consolidated emotional: " + str(dddd) + "...")
            if j==0:
                forg = np.array((dd))
                cons = np.array((d))
                forgcons = np.array((ddd))
                forgconsemo = np.array((dddd))
                hippocampalAccuracy = np.array(())
            else:
                forg = np.hstack((forg,dd))
                cons = np.hstack((cons,d))
                forgcons = np.hstack((forgcons,ddd))
                forgconsemo = np.hstack((dddd))
            ##hippocampus.redrawRecency()
        plt.plot(forg)
        plt.plot(cons)
        plt.plot(forgcons)
        plt.plot(forgconsemo)
        plt.legend(['f', 'c', 'f+c', 'f+c+e'], loc='upper left')
        plt.show()
        plt.draw()
        plt.show(block=True)
        plt.ioff()
        aa = hippocampus.get()
        pft = prefrontal.get()
        aaa = hippocampus.mapAvailable
        pftAssigned = prefrontal.mapAvailable
        print("...DONE...")
        #a0 = hippocampus.somActivity(data[0,:],True)
        #a1 = hippocampus.somActivity(data[1,:],True)
        #a2 = hippocampus.somActivity(data[2,:],True)
        #a3 = hippocampus.somActivity(data[3,:],True)
        #c = self.countInstances(usedData)
        #d = (15*15)-np.count_nonzero(aaa)
        a1 = prefrontal.reconstructInput2(np.array([1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0]))
        a2 = hippocampus.reconstructInput2(np.array([1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0]))
        a3 = prefrontal.reconstructInput2(np.array([1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1]))
        a4 = hippocampus.reconstructInput2(np.array([1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1]))
        a5 = prefrontal.reconstructInput2(np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]))
        a6 = hippocampus.reconstructInput2(np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]))
        d = 5
        b = 6
            
    def countInstances(self,data):
        data_unique = np.vstack({tuple(row) for row in data})
        h,w = data_unique.shape
        return h, data_unique
        
    def pltsin(self,fig,im,data):
        im.set_array(data)
        y = data
        fig.canvas.draw()
            
main()
