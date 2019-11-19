import numpy as np
import pandas as pd

data = pd.read_csv('data_som.csv')
learn_rate = 0.6
init_radius = 0
print(data)

def kohonenSOM(vectors,weight,learn_rate,init_radius):
    while(learn_rate >= 0.1):
        index_weight = 0
        for i in range(len(vectors)):
            print('vector ke',i)
            dst_min = 9999
            index_weight = 0
            for k in range(len(weight)):
                Dist = 0
                for j in range(len(vectors[1])):
                    Dist += np.power((weight[k][j] - vectors[i][j]),2)
                print('Dist ke',k,'=',round(Dist, 4))
                if(Dist < dst_min):
                    dst_min = Dist
                    index_weight = k
            print('distance min',round(dst_min, 4), '\n')
            for l in range(len(weight[1])):
                temp = weight[index_weight][l] + learn_rate*(vectors[i][l] - weight[index_weight][l])
                weight[index_weight][l] = temp
        learn_rate = 0.5 * learn_rate
        print(np.round(weight, 4))
        print('learning rate =',round(learn_rate, 4), '\n\n')
        init_radius = 0

vectors_dt = []
for i in range(2,6):
    vectors_dt.append(data.iloc[:,i].tolist())

weight_dt = []
for i in range(0,2):
    weight_dt.append(data.iloc[:,i].tolist())
    
#print(vectors_dt, weight_dt)
kohonenSOM(vectors_dt, weight_dt, learn_rate, init_radius)