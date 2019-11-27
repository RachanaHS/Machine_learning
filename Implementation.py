#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from numpy.random import RandomState
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from random import seed
from random import randrange
from csv import reader
from math import sqrt


# In[2]:


#initial dataframe
dataframe=pd.read_csv('cat1.csv')
pd.value_counts(dataframe['class']).plot.bar()#plotting bar graph
plt.title('stars and quessars')
plt.ylabel('frequency')
plt.xlabel('class')
dataframe['class'].value_counts()


# In[3]:


i=1
dataframe.drop(dataframe.columns[i], axis=1)


# In[4]:


#random upsampling using resample
from sklearn.utils import resample

class_star=dataframe[dataframe['class']==0]   #minority class--->stars
class_quessar=dataframe[dataframe['class']==1]    #majority class---->quessars
#print(class_star.head())
#print(len(class_star))
#print(len(class_quessar))

#upsampling minority class
#here we use sampling with replacement by matching it with quessar class.
#random_state is used to reproduce results
class_star_upsampled= resample(class_star,
                               replace=True,
                               n_samples=len(class_quessar),
                               random_state=123)

#combine the class_quessar(majority) with the class_star_upsampled(upsampled minority class)
upsampled =pd.concat([class_quessar,class_star_upsampled])
upsampled['class'].value_counts()
#upsampled.head()


# In[5]:


upsampled.head()


# In[6]:


upsampled=upsampled.drop(['pred','spectrometric_redshift'],axis=1)
upsampled.head()


# In[7]:


column=['galex_objid','sdss_objid','u','g','r','i','z','extinction_u','extinction_g','extinction_r','extinction_i','extinction_z','nuv_mag','fuv_mag','nuv-u','nuv-g','nuv-r','nuv-i','nuv-z','u-g','u-r','u-i','u-z','g-r','g-i','g-z','r-i','r-z','i-z','fuv-nuv','fuv-u','fuv-g','fuv-r','fuv-i','fuv-z','class']


# In[8]:


#reindexing is done so that the class column which is present in the middle of the dataset is shifted to the last for easy handling
upsampled=upsampled.reindex(columns=column)


# In[9]:


upsampled.head()


# In[10]:


#converting upsampled dataframe to list 
list_upsampled=upsampled.values.tolist()
print(len(list_upsampled))


# In[11]:


#we calculate the euclidean distance between two rows
#If the euclidean distance is small,similar is the record
#larger the euclidean distance dissimilar is the record


def get_Euclidean_Distance(data_row1, data_row2):
    d = 0.0
    for r in range(len(data_row1)-1):
        d += (data_row1[r] - data_row2[r])**2
    dist=sqrt(d)
    return dist 

#calculating the accuracy when the function accuracy_metric is called
#if the class label is equal to the one predicted by the algorithm,variable correct_values is being incremented
def metrix_accuracy(actual_values, predicted_values,actual_list,predicted_list):
    correct_values = 0
    for i in range(len(actual_values)):
        actual_list.append(actual_values[i])
        predicted_list.append(predicted_values[i])
        if actual_values[i] == predicted_values[i]:
            correct_values = correct_values + 1                       
    result = correct_values/ float(len(actual_values)) * 100.0         
    return result,actual_list,predicted_list




# splitting the entire upsampled dataset into k folds                        
def cross_validation_split(upsampled_data, n_folds):
    dataset_copy = list(upsampled_data)
    fold_size = int(len(upsampled_data) / n_folds)
    split_dataset = []                                     
    for _ in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            i = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(i))
        split_dataset.append(fold)
    return split_dataset



# Evaluate an algorithm using a cross validation split                              #changed
def evaluate_algorithm(upsampled, algorithm, n_fold, *args):
    folds = cross_validation_split(upsampled, n_fold)
    accuracy_list = []
    actual_list=[]
    predicted_list=[]
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for i in fold:
            row_copy = list(i)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted_label = algorithm(train_set, test_set, *args)
        actual_label = [row[-1] for row in fold]
        accuracy,actual_list,predicted_list = metrix_accuracy(actual_label, predicted_label,actual_list,predicted_list)
        accuracy_list.append(accuracy)
    return accuracy_list,actual_list,predicted_list




#this function is used to get the nearest neighbors based on the euclidean distance which is calculated                                  
def neighbors_get(train_set, test_row, n_neighbors):
    d = []
    for train_row in train_set:
        dist = get_Euclidean_Distance(test_row, train_row)
        d.append((train_row, dist))
    d.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(n_neighbors):
        neighbors.append(d[i][0])
    return neighbors

#classification is predicted based on its neighbors                                                 
def classification_predict(train_set, test_row, n_neighbors):
    neighbors = neighbors_get(train_set, test_row, n_neighbors)
    labels = [i[-1] for i in neighbors]    #the last column represents the class
    #returns the label which is maximum with its neighbors
    pred = max(set(labels), key=labels.count)
    return pred
 
def k_nearest_neighbors(train_set, test_set, n_neighbors):
    pred = []                                                                       
    for i in test_set:
        #for a given test_set checks with its k nearest neighbors 
        class_labels = classification_predict(train_set, i, n_neighbors)
        pred.append(class_labels)
    return(pred)



#KNN model is tested on the astronomical dataset
neighbors = 5
folds=5
seed(1)
accuracies,actual_list,predicted_list = evaluate_algorithm(list_upsampled, k_nearest_neighbors, folds, neighbors)
print('Individual accuracy every iteration : %s' % accuracies)
print('Mean Accuracy: %.3f%%' % (sum(accuracies)/float(len(accuracies))))
print(len(actual_list))
print(len(predicted_list))


# In[71]:


def class_accuracy(actual_list,predicted_list):
    len_class0=0
    len_class1=0
    class0_count=0
    class1_count=0
    for j in range(len(actual_list)):
        if(actual_list[j]==0):
            len_class0 +=1
        else:
            len_class1 +=1
    print((len_class0))
    print((len_class1))
    for i in range(len(actual_list)):
        if(predicted_list[i]==0 and actual_list[i]==0):
            class0_count +=1
        if(predicted_list[i]==1 and actual_list[i]==1):
            class1_count +=1
    class0_accuracy=(class0_count/len_class0)*100
    class1_accuracy=(class1_count/len_class1)*100
    return class0_accuracy,class1_accuracy


# In[72]:


class_accuracy(actual_list,predicted_list)


# In[73]:


def confusion_matrix(actual_list,predicted_list):
    tp_count=0
    fp_count=0
    tn_count=0
    fn_count=0
    for i in range(len(actual_list)):
        if (predicted_list[i]==1 and actual_list[i]==1):
            tp_count=tp_count+1
        if(predicted_list[i]==0 and actual_list[i]==0):
            tn_count+=1
        if(actual_list[i]==0 and predicted_list[i]==1):
            fp_count+=1
        if(actual_list[i]==1 and predicted_list[i]==0):
            fn_count+=1
            
    confusion_matrix=[]
    confusion_matrix.append(tp_count)
    confusion_matrix.append(fp_count)
    confusion_matrix.append(tn_count)
    confusion_matrix.append(fn_count)
    sum=tp_count+tn_count+fp_count+fn_count
    accuracy=((tp_count+tn_count)/(sum))*100
    precision=((tp_count)/(tp_count+fp_count))*100
    sensitivity=((tp_count)/(tp_count+fn_count))*100
    specificity=((tn_count)/(tn_count+fp_count))*100
    return confusion_matrix,accuracy,precision,sensitivity,specificity


# In[74]:


confusion_matrix,accuracy,precision,sensitivity,specificity=confusion_matrix(actual_list,predicted_list)
print(confusion_matrix)
print("accuracy :",accuracy)
print("precision :", precision)
print("sensitivity :",sensitivity)
print("specificity :", specificity)


# In[ ]:




