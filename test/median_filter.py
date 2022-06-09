#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class MedianFilter():
    
    def __init__(self, data_size, windows_size):
        self.data_size = data_size
        self.windows_size = windows_size
        self.data_storage = np.matrix(np.zeros((data_size, windows_size)))
        self.sort_storage = np.copy(self.data_storage)
        self.counter = 0
        self.flag = False
        self.middle = windows_size // 2

    def apply_median(self, data):
        
        if self.counter+1 == self.windows_size:
            self.flag = True
            self.counter = 0
        self.sort_storage = self.data_storage
        
        if self.flag:
           #for i in range (len(self.sort_storage)):
           self.sort_storage = np.sort(self.sort_storage)
                #self.sort_storage = self.sort_storage[:,self.sort_storage[i].argsort()]
                
    

        self.data_storage[:,self.counter] = np.matrix(data).T
        self.counter +=1
        return self.sort_storage[:,self.middle]

#MedianFilter = MedianFilter(6, 15)
#MedianFilter.test()
