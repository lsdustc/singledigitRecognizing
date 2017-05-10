# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 19:53:01 2017

@author: Shidong Li
"""

import numpy as np
import librosa
from math import ceil
import matplotlib.pyplot as plt
from librosa.display import specshow
import os
import tensorlayer as tl

path = 'Record/'
N_frames = 20 
wavlist = os.listdir(path)
framelen = 400
nummfcc = 45
def rawdata():
    audio = []
    labels = []
    for n in range(len(wavlist)):
        string = wavlist[n]
        string = string[0:string.index('.')]
        label  = int(string)%10
        wav,_  = librosa.load(path+wavlist[n],sr = 8000)
        data = np.square(wav)
#       data = np.append(data,np.zeros([1,framelen - len(data)%framelen],np.float32))
        framestack = []
        k = 0
        for i in range(int(len(data)/framelen)):
            frame =np.sum(data[i*framelen:i*framelen+framelen])
            if bool(frame > 0.7) is True :
                framestack.append(wav[i*framelen:i*framelen+framelen])
            elif bool(len(framestack) != 0) is True :
                nextFrame  = np.sum(data[(i+1)*framelen:(i+3)*framelen])
                if bool(nextFrame > 0.5) is True:
                    framestack.append(wav[i*framelen:i*framelen+framelen])
                else:
                    length = len(np.reshape(np.array(framestack),[-1,]))
                    if length < framelen*2.5:
                        framestack = []
                    else:
                        framestack= list(np.reshape(np.array(framestack),[-1,]))
                        audio.append(framestack)
                        framestack = []
                        k+=1
        labels.extend(list(label*np.ones([k,],dtype = int)))

    return audio,labels
def mfccset():
    wavdata,label = rawdata()
    mfccs = []
    for i in wavdata:
        if len(i)%(N_frames-1) != 0 :
             length = int(ceil(len(i)/(N_frames-1)))
             sli =  np.append(np.array(i)
                             ,np.zeros(length-len(i)%length))
        else:
             sli = np.array(i)
             length=int(len(i)/(N_frames-1))
        mfccs.append(librosa.feature.mfcc(sli,sr=8000,n_mfcc = nummfcc,
                                              n_fft = 512,hop_length = length))
    lib = np.array(mfccs,dtype = np.float32)
    label = np.array(label,dtype = np.float32)
    TestX = lib[2201:]
    TestY = label[2201:]
    lib = lib[:2200]
    label =label[:2200]
    indices = np.arange(lib.shape[0])
    np.random.shuffle(indices)
    X = lib[indices]
    Y = label[indices]

    return X,Y,TestX,TestY
def plotMfcc(mfccs):
    plt.figure(figsize=(10, 4))
    specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout() 

def savedata():
    X,Y,TestX,TestY =mfccset()
    tl.files.save_any_to_npy({'TestX':TestX,'TestY':TestY,'X':X,'Y':Y,'nummfcc':nummfcc,'N_frames':N_frames},'data.npy')

if __name__ == "__main__":
    savedata()
