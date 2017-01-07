#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import datetime

# normalize functions
def normalize(nums):
    m = np.mean(nums)
    s = np.std(nums)
    nums = (nums - m) * 1.0 / s
    return nums,m,s

std_timestamp = 1420088400  # 2015/01/01 00:00:00 NewYork TimeZone Thursday
def normalize_timestamp(time):
    mi = np.min(time)
    time = time - std_timestamp
    return time

point_num = 100000
train = pd.read_csv("./train.csv", delimiter=',', nrows=point_num)

lng = train[["start_lng"]]
lng = np.append(lng,train[["end_lng"]])
lat = train[["start_lat"]]
lat = np.append(lat,train[["end_lat"]])

lng,lng_mean,lng_std = normalize(lng)
lat,lat_mean,lat_std = normalize(lat)

train["norm_start_lng"] = lng[:len(lng)/2]
train["norm_end_lng"] = lng[len(lng)/2:]
train["norm_start_lat"] = lat[:len(lat)/2]
train["norm_end_lat"] = lat[len(lat)/2:]

train["distance"] = abs(train["norm_start_lng"]+train["norm_start_lat"]-train["norm_end_lng"]-train["norm_end_lat"])
train["velocity"] = train["distance"] / train["duration"]

train["norm_start_timestamp"] = normalize_timestamp(train["start_timestamp"])

velocity_mean = np.mean(train["velocity"])

time_scale = 20     # each segment per 30 minutes 
train["simple_time"] = np.round(train["norm_start_timestamp"]/(60.0*time_scale))

new_timestamp = []
new_velocity = []
real_datetime = []
real_timestamp = []
base = int(24 * (60.0 / time_scale) * 4) # start from Monday (add 4 days) (24*60/time_scale hours)
offset = base

print(range(int((60.0/time_scale)*24*25)))

# seperate by half hour
for i in xrange(int((60.0/time_scale)*24*25)):
    new_velocity.append(np.mean(train.loc[(train["simple_time"] == offset), "velocity"]))
    new_timestamp.append(offset)
    real_timestamp.append(std_timestamp+offset*time_scale*60)
    real_timestamp_tmp = std_timestamp+offset*time_scale*60
    real_datetime.append(datetime.datetime.utcfromtimestamp(real_timestamp_tmp))

    offset += 1

stl_val = {"new_velocity" : pd.Series(new_velocity)*1000}


# scale timestamp divide size to reduce Nan value
df_stl = pd.DataFrame(stl_val,index=real_datetime)
df_stl["new_velocity"] = df_stl["new_velocity"].fillna(velocity_mean)


df_stl.index = pd.DatetimeIndex(real_datetime)

# plt.figure()
res = sm.tsa.seasonal_decompose(df_stl.values,freq=len(df_stl.values)/2)
fig = res.plot()
# fig.show()
plt.show()








def show_fig_new_month():
    plt.figure(figsize=(10,10),dpi=150)
    plt.plot(new_timestamp,new_velocity)
#     plt.title('point num: ' + str(point_num))
#     plt.ylim(0, 50)
#     base = 86400*day + 86400*4 # from Monday
#     offset = base + 86400
#     # plt.xlim(i_num*86400, i_num*86400+3600)
    plt.axis([0, (60/time_scale)*24*30, 0, 0.01])
    plt.show()

def show_fig_day(day):
    plt.figure(figsize=(10,10),dpi=150)
    plt.plot(train["norm_start_timestamp"].values,train["velocity"].values, 'or')
    plt.title('point num: ' + str(point_num))
    # plt.ylim(0, 50)
    base = 86400*day + 86400*4 # from Monday
    offset = base + 86400
    # plt.xlim(i_num*86400, i_num*86400+3600)
    plt.axis([offset, offset+86400, 0, 0.01])
    plt.show()

def show_fig_week(num_of_week):
    plt.figure(figsize=(10,10),dpi=150)

    base = 86400*7*num_of_week + 86400*4 # from Monday
    offset = base + 86400

    p1 = plt.subplot(241)
    p1.plot(train["norm_start_timestamp"].values,train["velocity"].values, '.')
    # p1.ylim(0, 50)
    p1.axis([offset, offset+86400, 0, 0.01])

    p2 = plt.subplot(242)
    p2.plot(train["norm_start_timestamp"].values,train["velocity"].values, '.')
    # p1.ylim(0, 50)
    offset += 86400
    p2.axis([offset, offset+86400, 0, 0.01])

    p3 = plt.subplot(243)
    p3.plot(train["norm_start_timestamp"].values,train["velocity"].values, '.')
    # p1.ylim(0, 50)
    offset += 86400
    p3.axis([offset, offset+86400, 0, 0.01])

    p4 = plt.subplot(244)
    p4.plot(train["norm_start_timestamp"].values,train["velocity"].values, '.')
    # p1.ylim(0, 50)
    offset += 86400
    p4.axis([offset, offset+86400, 0, 0.01])

    p5 = plt.subplot(245)
    p5.plot(train["norm_start_timestamp"].values,train["velocity"].values, '.')
    # p1.ylim(0, 50)
    offset += 86400
    p5.axis([offset, offset+86400, 0, 0.01])

    p6 = plt.subplot(246)
    p6.plot(train["norm_start_timestamp"].values,train["velocity"].values, '.')
    # p1.ylim(0, 50)
    offset += 86400
    p6.axis([offset, offset+86400, 0, 0.01])

    p7 = plt.subplot(247)
    p7.plot(train["norm_start_timestamp"].values,train["velocity"].values, '.')
    # p1.ylim(0, 50)
    offset += 86400
    p7.axis([offset, offset+86400, 0, 0.01])

    plt.title('point num: ' + str(point_num))
    plt.show()

# print(min(train["start_timestamp"]))
show_fig_day(0)
show_fig_week(2)
show_fig_new_month()