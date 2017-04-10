# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 20:50:21 2017

@author: Lukas
"""

import numpy as np

class datasetGenerator:
    numberOfInstances = [7,7,4]
    sentence = np.array([])
    
    def __init__(self,txtFile):
        """7patients, 4transitive, 3intransitive """
        self.sentence = np.array([])
        self.sentence = np.loadtxt(txtFile,dtype=str,delimiter=';',usecols=range(3))
        self.sentence = self.oneHotEncodeDataset()
        
    def get(self):
        return self.sentence
        

    def oneHotEncodeDataset(self):
        h,w = self.sentence.shape
        encodedSentence = np.array([]) 
        for i in range(h):
            encodedEp = np.array([])
            for j in range(w):
                if j==0:
                    encodedEp = self.oneHotEncodeRow(j,self.sentence[i][j])
                else:
                    encodedEp = np.hstack((encodedEp,self.oneHotEncodeRow(j,self.sentence[i][j])))
            if i==0:
                encodedSentence = encodedEp
            else:
                encodedSentence = np.vstack((encodedSentence,encodedEp))
        return encodedSentence

        
    def oneHotEncodeRow(self,wordType,pos):
        """ 
        wordType: 0 - patient; 1 - transitive; 2 - intransitive
        pos - label number
        """
        sentenceLen = self.numberOfInstances[wordType]
        b = np.zeros((1,sentenceLen))
        if int(pos)==0:
            return b    
        else:
            pos = str(int(pos)-1)
            b[np.arange(1), pos] = 1
        return b
        
