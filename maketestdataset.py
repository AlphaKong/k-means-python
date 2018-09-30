# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def dataset_test():
    area=[]
    for i in range(250):
        x=np.random.randint(0,100)
        y=np.random.randint(0,100)
        area.append([x,y])
        
#    for i in range(250):
#        x=np.random.randint(10,45)
#        y=np.random.randint(10,45)
#        area.append([x,y])
#    
#    
#    for i in range(50):
#        x=np.random.randint(-5,5)
#        y=np.random.randint(23,38)
#        area.append([x,y])
#    
#    for i in range(50):
#        x=np.random.randint(15,25)
#        y=np.random.randint(-5,10)
#        area.append([x,y])
    
    a=np.array(area)/10.0
    
    plt.scatter(a[:,0],a[:,1])
    plt.show()
    return a


dataset_test()