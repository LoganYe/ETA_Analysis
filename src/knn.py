#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import neighbors

def normalize(nums):
    m = np.mean(nums)
    v = np.var(nums)
    for i in range(len(nums)):
        nums[i] = (nums[i] - m) * 1.0 / pow(v, 0.5)
    return nums


#dataset = []
start_lng = []
start_lat = []
end_lng = []
end_lat = []
start_timestamp = []
duration = []

with open('../data/train.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    cnt = 0
    for row in spamreader:
        if 0 == cnt:
            cnt += 1
            continue
        #if cnt > 1000000:
        #    break
        cnt += 1
        #print(row)

        start_lng.append(float(row[1]))
        start_lat.append(float(row[2]))
        end_lng.append(float(row[3]))
        end_lat.append(float(row[4]))
        start_timestamp.append(int(row[5]))
        duration.append(int(row[6]))

start_lng = normalize(start_lng)
start_lat = normalize(start_lat)
end_lng = normalize(end_lng)
end_lat = normalize(end_lat)
start_timestamp = normalize(start_timestamp)

#print(start_lng,start_lat,end_lng,end_lat,start_timestamp,duration)

X = []
y = []
for i in range(cnt - 1):
    temp = []
    temp.append(start_lng[i])
    temp.append(start_lat[i])
    temp.append(end_lng[i])
    temp.append(end_lat[i])
    temp.append(start_timestamp[i])
    #temp.append(int(row[6]))
    #dataset.append(temp)
    X.append(temp)
    y.append(duration[i])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#print (X_train)
#print (X_test)
#print (y_train)
#print (y_test)

#n_neighbors = 3
n_neighbors = 1000

#for i, weights in enumerate(['uniform', 'distance']):
knn = neighbors.KNeighborsRegressor(n_neighbors, weights = 'distance')
knn.fit(X_train, y_train)

"""
SSE1 = SST1 = 0
y_bar = np.mean(y_test)
for i, test in enumerate(X_test):
    predict = knn.predict(test)
    #print (predict, y_test[i])
    SSE1 += (predict - y_test[i]) ** 2
    SST1 += (y_test[i] - y_bar) ** 2

SSE = SST = 0
y_bar = np.mean(y_train + y_test)
for i, test in enumerate(X_test):
    predict = knn.predict(test)
    #print (predict, y_test[i])
    SSE += (predict - y_test[i]) ** 2
    SST += (y_test[i] - y_bar) ** 2
for i, test in enumerate(X_train):
    predict = knn.predict(test)
    #print (predict, y_test[i])
    SSE += (predict - y_train[i]) ** 2
    SST += (y_train[i] - y_bar) ** 2

print("testing r square:")
print(1 - (SSE1 / SST1))
print("sum r square:")
print(1 - (SSE / SST))
"""

"""
MSE = 0
y_bar = np.mean(y_test)
for i, test in enumerate(X_test):
    predict = knn.predict(test)
    #print (predict, y_test[i])
    MSE += (predict - y_test[i]) ** 2

print("testing MSE:")
print(MSE/cnt)
"""


start_lng_t = []
start_lat_t = []
end_lng_t = []
end_lat_t = []
start_timestamp_t = []

with open('../data/test.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    cnt = 0
    for row in spamreader:
        if 0 == cnt:
            cnt += 1
            continue
        #if cnt > 10:
        #    break
        cnt += 1
        start_lng_t.append(float(row[1]))
        start_lat_t.append(float(row[2]))
        end_lng_t.append(float(row[3]))
        end_lat_t.append(float(row[4]))
        start_timestamp_t.append(int(row[5]))

start_lng_t = normalize(start_lng_t)
start_lat_t = normalize(start_lat_t)
end_lng_t = normalize(end_lng_t)
end_lat_t = normalize(end_lat_t)
start_timestamp_t = normalize(start_timestamp_t)

with open('../output/output_knn.csv', 'w') as csvfile:
    fieldnames = ['row_id', 'duration']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(cnt - 1):
        temp = []
        temp.append(start_lng_t[i])
        temp.append(start_lat_t[i])
        temp.append(end_lng_t[i])
        temp.append(end_lat_t[i])
        temp.append(start_timestamp_t[i])

        predict = knn.predict(temp)[0]
        writer.writerow({'row_id': i, 'duration': predict})






