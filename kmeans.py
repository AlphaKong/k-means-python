# -*- coding: utf-8 -*-
from collections import defaultdict
from random import uniform
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np


def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    NB. points can have more dimensions than 2
    
    Returns a new point which is the center of all the points.
    """
    dimensions = len(points[0])
    print(dimensions)

    new_center = []

    for dimension in range(dimensions):
        dim_sum = 0  # dimension sum
        for p in points:
            dim_sum += p[dimension]

        # average of each dimension
        new_center.append(dim_sum / float(len(points)))

    return new_center

#在新的聚类类别里面选出新的中心点,即该类中所有样本的特征的均值
#注意defaultdict的用法
def update_centers(data_set, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers where `k` is the number of unique assignments.
    """
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)
    
    for points in new_means.items():
        print('------------')
        print(len(points[1]))
        centers.append(point_avg(points[1]))

    return centers

#以每个k点为中心，对数据集的每个数据进行k类标注
def assign_points(data_points, centers):
    """
    Given a data set and a list of points betweeen other points,
    assign each point to an index that corresponds to the index
    of the center point on it's proximity to that point. 
    Return a an array of indexes of centers that correspond to
    an index in the data set; that is, if there are N points
    in `data_set` the list we return will have N elements. Also
    If there are Y points in `centers` there will be Y unique
    possible values within the returned list.
    """
    assignments = []
    for point in data_points:
        shortest = float('inf')  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    """
    dimensions = len(a)
    
    _sum = 0
    for dimension in range(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return sqrt(_sum)


#根据数据集的每个特征的上限和下限随机生成k个点
def generate_k(data_set, k):
    """
    Given `data_set`, which is an array of arrays,
    find the minimum and maximum for each coordinate, a range.
    Generate `k` random points between the ranges.
    Return an array of the random points within the ranges.
    """
    centers = []
    dimensions = len(data_set[0])
    min_max = defaultdict(int)
    
    #每个特征最大最小的范围
    for point in data_set:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val
    
    #随机生成点
    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]
            
            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)

    return centers


def plt_draw(dataset,assignments):
    point_arr = defaultdict(list)
    for p,a in zip(dataset,assignments):
        point_arr[a].append(p)
    for points in point_arr.items():
        tmp_p=np.array(points[1][1:])
        plt.scatter(tmp_p[:,0],tmp_p[:,1])
    plt.show()


def k_means(dataset, k=2):
    #根据数据集，随机生成K个点
    k_points = generate_k(dataset, k)
    ##以每个k点为中心，对数据集的每个数据进行k类标注
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        print('-----------------------------------')
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
        plt_draw(dataset,assignments)
    return assignments


'''
step 1:
    输入数据样本dataset
st3p 2:
    根据数据样本随机生成k个中心点
step 3:
    对每个样本x分别与k个中心点求相似，并标记类别
step 4:
    在新类别中选出一个新的类别中心
step 5:
    反复重复3和4的步骤，知道某个条件满足

'''



if __name__=='__main__':
    from maketestdataset import dataset_test
    dataset=dataset_test()
    assignments=k_means(dataset,6)
    plt_draw(dataset,assignments)
  
 


