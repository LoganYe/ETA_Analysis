#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# normalize functions
def normalize(nums):
    m = np.mean(nums)
    s = np.std(nums)
    nums = (nums - m) * 1.0 / s
    return nums

def normalize_duration(time):
    m = np.mean(time)
    time = time / m
    return time

def normalize_timestamp(time):
    mi = np.min(time)
    time = (time - mi)/1000
    return time

point_num = 100000
train = pd.read_csv("./train.csv", delimiter=',', nrows=point_num)

lng = train[["start_lng"]]
lng = np.append(lng,train[["end_lng"]])
lat = train[["start_lat"]]
lat = np.append(lat,train[["end_lat"]])

lng = normalize(lng)
lat = normalize(lat)

train["norm_start_lng"] = lng[:len(lng)/2]
train["norm_end_lng"] = lng[len(lng)/2:]
train["norm_start_lat"] = lat[:len(lat)/2]
train["norm_end_lat"] = lat[len(lat)/2:]

train["distance"] = abs(train["norm_start_lng"]+train["norm_start_lat"]-train["norm_end_lng"]-train["norm_end_lat"])
train["norm_duration"] = normalize_duration(train["duration"])
train["velocity"] = train["distance"] / train["norm_duration"]

train["norm_start_timestamp"] = normalize_timestamp(train["start_timestamp"])


plt.figure(1)
plt.plot(train[["norm_start_timestamp"]].values,train[["velocity"]].values, 'or')
plt.title('point num: ' + str(point_num))
plt.xlim(0, 500)
# plt.xlim(0, 10)
plt.show()