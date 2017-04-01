# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 20:50:21 2017

@author: Lukas
"""

import numpy as np

class datasetGenerator:
    def __init__(self,wordLen,sentenceLen):
        self.sentence = np.array([])
        for i in range(sentenceLen):
            self.sentence = np.append(self.sentence,self.generateWord(wordLen,sentenceLen,i))
    
    def get(self):
        return self.sentence
        
    def generateWord(self,wordLen,sentenceLen,pos):
        return self.oneHotEncode(wordLen,sentenceLen,pos)
        
    def oneHotEncode(self,wordLen,sentenceLen,pos):
        b = np.zeros((1,sentenceLen))
        pos = np.mod(pos*np.random.randint(550),wordLen)
        b[np.arange(1), pos] = 1
        return b
        
