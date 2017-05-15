# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 20:50:21 2017

@author: Lukas
"""

import numpy as np

class datasetGenerator:
    numberOfInstances = [4,4,8,1,1]
    sentence = np.array([])
    
    def __init__(self,txtFile):
        """4 agents, 7patients, 4transitive/3intransitive, 1positive emotion, 1negative emotion """
        self.sentence = np.array([])
        print("...loading the dataset...")
        self.sentence = np.loadtxt(txtFile,dtype=str,delimiter=';',usecols=range(6))
        self.sentence = self.oneHotEncodeDataset()
        
    def get(self):
        return self.sentence
        
    def getEpisodes(self,numOfEpisodes):
        data = self.sentence
        np.random.shuffle(data)
        return data[:numOfEpisodes,:]
        

    def oneHotEncodeDataset(self):
        h,w = self.sentence.shape
        w = 5
        encodedSentence = np.array([]) 
        for i in range(h):
            word = self.processEmotions(self.sentence[i])
            encodedEp = np.array([])
            for j in range(w):
                if j==0:
                    encodedEp = self.oneHotEncodeWord(j,word[j])
                else:
                    encodedEp = np.hstack((encodedEp,self.oneHotEncodeWord(j,word[j])))
            if i==0:
                encodedSentence = encodedEp
            else:
                encodedSentence = np.vstack((encodedSentence,encodedEp))
        return encodedSentence

    def processEmotions(self,sent):
        """
        1 - positive, 0 - neutral, -1 - negative. If there are more positive - 10, more negative - 01, allneutral - 00, same number - 11        
        """
        fst = sent[1]
        sec = sent[3]
        trd = sent[5]
        positive=0
        neutral=0
        negative=0
        unique, counts = np.unique(np.array((fst,sec,trd)), return_counts=True)
        if '-1' in unique:
            i, = np.where('-1'==unique)
            negative = counts[i[0]]
        if '0' in unique:
            i, = np.where('0'==unique)
            neutral = counts[i[0]]
        if '1' in unique:
            i, = np.where('1'==unique)
            positive = counts[i[0]]
        if(neutral==3):
            emotion = np.array(('0','0'))
        if(positive>negative):
            emotion = np.array(('1','0'))
        if(positive<negative):
            emotion = np.array(('0','1'))
        if(positive==1 and negative==1 and neutral==1):
            emotion = np.array(('1','1'))
        result = np.array((sent[0],sent[2],sent[4]))
        result = np.hstack((result,emotion))
        return result
     
    def oneHotEncodeWord(self,wordType,pos):
        """ 
        wordType: 0 - agent; 1 - patient/intransitive; 2 - transitive/padding
        pos - label number
        """
        sentenceLen = self.numberOfInstances[wordType]
        b = np.zeros((1,sentenceLen))
        if wordType>=2:
            pos = str(int(pos))
            if wordType>=3:
                return np.array([[int(pos)]]).T
        else:
            pos = str(int(pos)-1)
        b[np.arange(1), pos] = 1
        return b
       
