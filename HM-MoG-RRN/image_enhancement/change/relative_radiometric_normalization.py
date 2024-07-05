from image_enhancement.histogram.histogram import apply_mappings,solve_least_l1_parameters_from_weighted,histogram_matching_from_weighted_mpoints_to_mpoints,histogram_matching_from_weighted_mpoints_to_mpoints_with_mappings,least_l1_mapping_from_weighted_mpoints_to_mpoints_simple,obtain_least_l1_parameters_from_weighted_mpoints_to_mpoints_with_points_simple
from coordinate_processing.transformation import obtain_intersection_area,obtain_corresponding_point_and_location,generate_change_point_image
import numpy as np
from tif_shp_utils.read_write_tifs import get_geo,read_tif
import matplotlib.pyplot as plt
from mathmatica.distribution import gaussian_distribution,laplace_distribution
from numba import njit
from image_enhancement.stretching.stretching import linear_stretching_for_images,get_mins_maxs_for_images,recover_scale_for_images_for_1_255_linear_stretching
from image_enhancement.regression.least_square import least_square_mapping_from_weighted_mpoints_to_mpoints,obtain_least_square_parameters_from_weighted_mpoints_to_mpoints_with_points,least_square_mapping_from_weighted_mpoints_to_mpoints_simple,obtain_least_square_parameters_from_weighted_mpoints_to_mpoints_with_points_simple

def relative_radiometric_normalization_via_change_noise_model(images_s,images_t,geo_s,geo_t,disable_bias=False,number_of_noise=2,mapping="HM",hypothesis="MoG"):
    # 获取影像重叠区域
    images_s_o,geo_s_o,images_t_o,geo_t_o = obtain_intersection_area(images_s,geo_s,images_t,geo_t)
    print("获取影像重叠区域")
    # 获取影像对应点数组，位置数组
    point_s,point_t,location = obtain_corresponding_point_and_location(images_s_o,geo_s_o,images_t_o,geo_t_o)
    print("获取影像对应点数组，位置数组")
    # 获取映射

    if mapping=='HM' and hypothesis=='MoG' :
        mappings,ratio,probability = mixture_change_noise_model(point_s,point_t,disable_bias=disable_bias,number_of_noise=number_of_noise,hypothesis="MoG",mapping="HM")
        print("获取映射")
        mapped_images = apply_mappings(images_s,mappings)

    print("mapping",mapping,"hypothesis",hypothesis)
    # 输出变点图
    change_point_image = generate_change_point_image(location,probability,geo_s,images_s_o.shape[0],images_s_o.shape[1])
    return mapped_images,change_point_image

@njit
def mixture_change_noise_model(point_s,point_t,disable_bias = True,number_of_noise=2,hypothesis="hmg",mapping="HM"):
    # 获取颜色分辨率
    color_resolution_s = np.max(point_s)
    color_resolution_t = np.max(point_t)

    # 设置混合分布的个数
    g = number_of_noise

    # 设置不变点/变点正态分布标准差初始值
    center_no_change_std = np.zeros((point_t.shape[1],g))
    for i in range(point_t.shape[1]):
        for j in range(g):
            center_no_change_std[i,j] = color_resolution_t/255*(10+80*j)

    # 设置不变点/变化点正态分布偏移初始值
    mu = np.zeros((point_t.shape[1],g))
    for i in range(point_t.shape[1]):
        for j in range(g):
            mu[i,j] = color_resolution_t/255*(0+30*j)
    if disable_bias == True:
        mu = np.zeros((point_t.shape[1], g)) # no bias

    # 设置不变点变化点比例
    ratio = np.ones((g))/g

    # 初始化不变点比例函数
    gamma = np.ones((point_s.shape[0],g))

    # 根据总点数进行采样

    # 计算加权方法
    weight = np.ones((point_s.shape[0], point_s.shape[1]))

    difference =  point_t+0.0-point_s



    # 进行加权直方图匹配
    if mapping == "HM":
        mapped_point_s = histogram_matching_from_weighted_mpoints_to_mpoints(weight, point_s, point_t)
       # 计算差异
    difference = (point_t+0.0 - mapped_point_s)

    # 计算log 似然函数
    log_likelihood_new = 0
    for i in range(point_s.shape[0]):
        temp = np.ones(g)
        for k in range(g):
            temp[k] *= ratio[k]
            for j in range(point_s.shape[1]):
                if hypothesis=="MoG":
                    temp[k] = gaussian_distribution(difference[i, j]-mu[j,k], 0, center_no_change_std[j, k]) * temp[k]

        log_likelihood_new += np.log(np.sum(temp))
    print("mean log_likelihood",log_likelihood_new/point_s.shape[0] )
    # 进行循环
    count = 0
    while True:
        log_likelihood_old = log_likelihood_new
        # E 求期望
        # 计算差异
        difference = (point_t+0.0-mapped_point_s)

        for i in range(len(gamma)):
            temp = np.ones(g)
            for k in range(g):
                temp[k] *= ratio[k]
                for j in range(point_s.shape[1]):
                    if hypothesis=="MoG":
                        temp[k]= gaussian_distribution(difference[i,j]-mu[j,k],0,center_no_change_std[j,k])*temp[k]

            for k in range(g):
                if np.sum(temp)==0:
                    gamma[i, k]=1
                else:
                    gamma[i,k] = temp[k]/np.sum(temp)

        # M 最大化

        n = np.zeros(g)
        for k in range(g):
            n[k] = np.sum(gamma[:,k])

        for k in range(g):
            ratio[k] = n[k]/len(gamma)
        print("不变点变点比", ratio)
        print("不变点，变化点数量",n)
        for j in range(point_s.shape[1]):
            for k in range(g):
                if hypothesis == "MoG":
                    center_no_change_std[j,k] = np.sqrt(np.sum(gamma[:,k]*(difference[:,j]-mu[j,k])**2)/n[k])

        print("标准差",center_no_change_std)

        for j in range(point_s.shape[1]):
            for k in range(1,g):
                if hypothesis == "MoG":
                    mu[j,k] = np.sum(gamma[:,k]*difference[:,j])/n[k]

        if disable_bias == True:
            mu = np.zeros((point_t.shape[1], g))
        print("偏执", mu)
        # 进行加权直方图匹配
        # 计算加权方法
        weight = np.zeros((point_s.shape[0], point_s.shape[1]))
        for j in range(point_s.shape[1]):
            if hypothesis == "MoG":
                weight[:, j] = gamma[:, 0] / center_no_change_std[j, 0] ** 2





        if mapping == "HM":
            mapped_point_s,mappings = histogram_matching_from_weighted_mpoints_to_mpoints_with_mappings(weight, point_s, point_t)


        # 计算差异
        difference = (point_t-mapped_point_s)
        # 计算log 似然函数
        log_likelihood_new = 0
        for i in range(point_s.shape[0]):
            temp = np.ones(g)
            for k in range(g):
                temp[k] *= ratio[k]
                for j in range(point_s.shape[1]):
                    if hypothesis == "MoG":
                        temp[k] = gaussian_distribution(difference[i, j]-mu[j,k], 0, center_no_change_std[j, k]) * temp[k]

            log_likelihood_new += np.log(np.sum(temp))
        print("mean log_likelihood_old",log_likelihood_old/point_s.shape[0],"mean log_likelihood_new",log_likelihood_new/point_s.shape[0])
        if log_likelihood_old>log_likelihood_new:
            break

    return mappings,ratio,gamma
