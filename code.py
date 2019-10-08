import pandas as pd
import numpy as np
import random
from statistics import mean
import warnings
warnings.filterwarnings('ignore')

# Distances
def euclidian(p1, p2): 
    dist = 0
    for i in range(len(p1)):
        dist = dist + np.square(p1[i]-p2[i])
    dist = np.sqrt(dist)
    return dist;

def manhattan(p1, p2): 
    dist = 0
    for i in range(len(p1)):
        dist = dist + abs(p1[i]-p2[i])
    return dist;

def cosine(p1,p2):
    dist = 0
    for i in range(len(p1)):
        dist = dist + (p1[i]*p2[i]) // (np.sqrt(np.square(p1[i])) * np.sqrt(np.square(p2[i])))
    return dist;

# kNN Function
def kNN(X_train,y_train, X_test, k, dist='cosine',q=2):
    pred = []
    # Adjusting the data type
    if isinstance(X_test, np.ndarray):
        X_test=pd.DataFrame(X_test)
    if isinstance(X_train, np.ndarray):
        X_train=pd.DataFrame(X_train)
        
    for i in range(len(X_test)):    
        # Calculating distances for our test point
        newdist = np.zeros(len(y_train))

        if dist=='euclidian':
            for j in range(len(y_train)):
                newdist[j] = euclidian(X_train.iloc[j,:], X_test.iloc[i,:])
    
        if dist=='manhattan':
            for j in range(len(y_train)):
                newdist[j] = manhattan(X_train.iloc[j,:], X_test.iloc[i,:])
        
        if dist=='cosine':
            for j in range(len(y_train)):
                newdist[j] = cosine(X_train.iloc[j,:], X_test.iloc[i,:])

        # Merging actual labels with calculated distances
        y_train = y_train.flatten()
        newdist = np.array([newdist, y_train])

        ## Finding the closest k neighbors
        # Sorting index
        idx = np.argsort(newdist[0,:])

        # Sorting the all newdist
        newdist = newdist[:,idx]
        #print(newdist)

        # We should count neighbor labels and take the label which has max count
        # Define a dictionary for the counts
        c = {'0':0,'1':0,'2':0, '3':0,'4':0,'5':0,'6':0, '7':0,'8':0,'9':0 }
        # Update counts in the dictionary 
        for j in range(k):
            c[str(int(newdist[1,j]))] = c[str(int(newdist[1,j]))] + 1

        key_max = max(c.keys(), key=(lambda k: c[k]))
        pred.append(int(key_max))
        
    return pred

def cross_validation_split(dataset, folds):
        dataset_split = []
        df_copy = dataset
        fold_size = int(df_copy.shape[0] / folds)
        
        # for loop to save each fold
        for i in range(folds):
            fold = []
            # while loop to add elements to the folds
            while len(fold) < fold_size:
                # select a random element
                r = random.randrange(df_copy.shape[0])
                # determine the index of this element 
                index = df_copy.index[r]
                # save the randomly selected line 
                fold.append(df_copy.loc[index].values.tolist())
                # delete the randomly selected line from
                # dataframe not to select again
                df_copy = df_copy.drop(index)
            # save the fold     
            dataset_split.append(np.asarray(fold))
            
        return dataset_split 

def kfoldCV(dataset, f, k, model="knn"):
    data=cross_validation_split(dataset,f)
    result=[]
    # determine training and test sets 
    for i in range(f):
        r = list(range(f))
        r.pop(i)
        for j in r :
            if j == r[0]:
                cv = data[j]
            else:    
                cv=np.concatenate((cv,data[j]), axis=0)
        
        # apply the selected model
        # default is logistic regression
        if model == "logistic":
            # default: alpha=0.1, num_iter=30000
            # if you change alpha or num_iter, adjust the below line         
            test = logistic(cv[:,0:9],cv[:,8],data[i][:,0:9])     
        elif model == "knn":
            test = kNN(cv[:,0:8],cv[:,8],data[i][:,0:8],k)
            
        # calculate accuracy    
        acc=(test == data[i][:,8]).sum()
        result.append(acc/len(test))
        
    return result

ys = pd.read_csv('yeast.data', sep = '\s+' ,header = None)
ys.columns=['id','mcg','gvh','alm','mit','erl','pox','vac','nuc','class']
print(ys)

yt = ys.iloc[:,1:]
print(yt)

ys['class'].unique()
yt['Class']=yt['class']
yt['Class'] = yt['Class'].map({'MIT': 0, 'NUC': 1, 'CYT': 2, 'ME1': 3, 'EXC': 4, 'ME2': 5, 'ME3': 6, 'VAC': 7, 'POX': 8, 'ERL': 9})
yt["Class"].unique()
yt = yt.drop(['class'],axis=1)

#run
acc = kfoldCV(yt, f=10, k=10, model="knn")
print("Akurasi euclidian:", acc)
print("rata-rata akurasi:", round(mean(acc), 3))

acc = kfoldCV(yt, f=10, k=10, model="knn")
print("Akurasi Manhattan:", acc)
print("rata-rata akurasi:", round(mean(acc), 3))

acc = kfoldCV(yt, f=10, k=10, model="knn")
print("Akurasi Cosine:", acc)
print("rata-rata akurasi:", round(mean(acc), 3))
