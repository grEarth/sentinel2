import numpy as np
import matplotlib.pyplot as plt
from image_enhancement.mask.mask import obtain_mask
from tif_shp_utils.read_write_tifs import read_tif
from numba import njit

@njit
def count_weighted_points_histogram(weight,points,length=None):
    if length is None:
        length = np.max(points)
    histogram = np.zeros(int(length)+1)
    for i in range(points.shape[0]):
        histogram[int(points[i])] += weight[i]
    return histogram

@njit
def count_weighted_points_histogram_with_negative(weight,points,length=None):
    bound=int(np.min(points))
    if length is None:
        length = np.max(points)-bound
    histogram = np.zeros(int(length)+1)
    for i in range(points.shape[0]):
        histogram[int(points[i])-bound] += weight[i]
    return histogram,bound

@njit
def count_image_histogram(image,length=None):
    if length is None:
        length = np.max(image)
    histogram = np.zeros(length+1,np.uint64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j]!=0:
                histogram[image[i,j]] += 1
    return histogram

@njit
def count_images_histograms(images):
    number = images.shape[2]
    length = np.max(images)
    histograms = np.zeros((length + 1,number), np.uint128)
    for i in range(number):
        histograms[:,i] = count_image_histogram(images[:,:,i])
    return histograms

@njit
def calculate_density(histogram):
    density = histogram/np.sum(histogram)
    return density

@njit
def calculate_cumulative_distribution(histogram):
    density = calculate_density(histogram)
    cumulative = np.zeros_like(density)
    for i in range(1,len(density)):
        cumulative[i]=density[i] + cumulative[i-1]
    return cumulative

@njit
def obtain_histogram_mapping(histogram_s,histogram_t):
    cumulative_s = calculate_cumulative_distribution(histogram_s)
    cumulative_t = calculate_cumulative_distribution(histogram_t)
    index_i = 0
    index_j = 0
    mapping = np.zeros_like(histogram_s)
    while index_i!=len(histogram_s):
        if cumulative_s[index_i] <= cumulative_t[index_j]:
            mapping[index_i] = index_j
            index_i += 1
        if cumulative_s[index_i] > cumulative_t[index_j]:
            if index_j+1<len(histogram_s):
                if cumulative_s[index_i] > cumulative_t[index_j+1]:
                    index_j += 1
                else:
                    if np.abs(cumulative_s[index_i] -cumulative_t[index_j]) <= np.abs(cumulative_t[index_j + 1]-cumulative_s[index_i]):
                        mapping[index_i]=index_j
                        index_i +=1
                    else:
                        mapping[index_i]=index_j+1
                        index_i +=1
            else:
                mapping[index_i] = index_j
                index_i += 1
    return mapping

@njit
def apply_mapping(image,mapping):
    mapped_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            mapped_image[i,j] = mapping[image[i,j]]
    return mapped_image

@njit
def apply_mapping_for_points(points,mapping):
    mapped_points = np.zeros_like(points)
    for i in range(points.shape[0]):
        mapped_points[i] = mapping[int(points[i])]
    return mapped_points

@njit
def apply_mappings_for_mpoints(mpoints,mappings):
    mapped_mpoints = np.zeros_like(mpoints)
    for c in range(mpoints.shape[2]):
        mapped_mpoints[:,c] = apply_mapping_for_points(mpoints[:,c],mappings[:,c])
    return mapped_mpoints

@njit
def apply_mappings(images,mappings):
    mapped_images = np.zeros_like(images)
    for c in range(images.shape[2]):
        mapped_images[:,:,c] = apply_mapping(images[:,:,c],mappings[:,c])
    return mapped_images

@njit
def histogram_matching_from_image_to_image(image_s,image_t):
    histogram_s = count_image_histogram(image_s)
    histogram_t = count_image_histogram(image_t)
    mapping = obtain_histogram_mapping(histogram_s,histogram_t)
    mapped_image =  apply_mapping(image_s,mapping)
    return mapped_image

@njit
def histogram_matching_from_images_to_images(images_s,images_t):
    mapped_images = np.zeros_like(images_s)
    for c in range(images_s.shape[2]):
        mapped_images[:,:,c] = histogram_matching_from_image_to_image(images_s[:,:,c],images_t[:,:,c])
    return mapped_images

@njit
def histogram_matching_from_weighted_points_to_points(weight,points_s,points_t):
    histogram_s = count_weighted_points_histogram(weight,points_s)
    histogram_t = count_weighted_points_histogram(weight,points_t)
    mapping = obtain_histogram_mapping(histogram_s,histogram_t)
    mapped_points =  apply_mapping_for_points(points_s,mapping)
    return mapped_points

@njit
def histogram_matching_from_weighted_points_to_points_with_mapping(weight,points_s,points_t,length=None):
    histogram_s = count_weighted_points_histogram(weight,points_s,length)
    histogram_t = count_weighted_points_histogram(weight,points_t)
    mapping = obtain_histogram_mapping(histogram_s,histogram_t)
    mapped_points =  apply_mapping_for_points(points_s,mapping)
    return mapped_points,mapping

@njit
def histogram_matching_from_weighted_mpoints_to_mpoints(weight,mpoints_s,mpoints_t):
    mapped_mpoints = np.zeros_like(mpoints_s)
    for c in range(mpoints_s.shape[1]):
        mapped_mpoints[:,c] = histogram_matching_from_weighted_points_to_points(weight[:,c],mpoints_s[:,c],mpoints_t[:,c])
    return mapped_mpoints


@njit
def histogram_matching_from_weighted_mpoints_to_mpoints_with_mappings(weight, mpoints_s, mpoints_t):
    length = np.max(mpoints_s)
    mapped_mpoints = np.zeros_like(mpoints_s)
    agument_mpoints = mpoints_t.copy()
    mappings = np.zeros((int(length)+1,mpoints_s.shape[1]))
    # gamma mu 目标函数上，有未了解的部分，需要考虑之后修复
    # for i in range(mpoints_s.shape[0]):
    #     index_max = np.argmax(gamma[i])
    #     agument_mpoints[i,:] -= mu[:,index_max]
    for c in range(mpoints_s.shape[1]):
        mapped_mpoints[:,c],mappings[:,c] = histogram_matching_from_weighted_points_to_points_with_mapping(weight[:,c],mpoints_s[:,c],agument_mpoints[:,c],length)
    return mapped_mpoints,mappings


@njit
def least_l1_mapping_from_weighted_mpoints_to_mpoints_simple(weight,mpoints_s,mpoints_t):

    W = np.zeros((2, mpoints_s.shape[1]))

    for j in range(weight.shape[1]):
        b_old = 0.0
        a_old = 0.1
        while True:
            # solve a

            sythentic_weight = weight[:,j]*mpoints_s[:,j]/1000
            sythentic_t = (1000*mpoints_t[:,j]/mpoints_s[:,j]).astype(np.int32)
            sythentic_histogram_t = count_weighted_points_histogram(sythentic_weight,sythentic_t)

            sythentic_cumulative_t = calculate_cumulative_distribution(sythentic_histogram_t)
            index_i = 0
            while index_i != len(sythentic_histogram_t):
                if sythentic_cumulative_t[index_i] >= 0.5:
                  break
                index_i += 1

            a_new = index_i/1000

            # solve b
            sythentic_t = ( mpoints_t[:, j] - a_new * mpoints_s[:, j]).astype(np.int32)
            sythentic_histogram_t,bound = count_weighted_points_histogram_with_negative(weight[:,j], sythentic_t)

            sythentic_cumulative_t = calculate_cumulative_distribution(sythentic_histogram_t)
            index_i = 0
            while index_i != len(sythentic_histogram_t):
                if sythentic_cumulative_t[index_i] >= 0.5:
                    break
                index_i += 1
            b_new = index_i+bound
            print("index_b", index_i)
            object_old = np.sum(weight[:,j]*np.abs(a_old*mpoints_s[:, j]+b_old-mpoints_t[:,j]))
            object_new = np.sum(weight[:, j] * np.abs(a_new * mpoints_s[:, j] + b_new - mpoints_t[:, j]))
            if object_new>=object_old:
                break
            else:
                a_old=a_new
                b_old=b_new

        W[0, j] = a_old
        W[1, j] = b_old

    return mpoints_s*W[0:1,:]+W[1:2,:]

@njit
def solve_least_l1_parameters_from_weighted(mpoints_t,weight):
    # solve b
    sythentic_t = (mpoints_t).astype(np.int32)
    sythentic_histogram_t, bound = count_weighted_points_histogram_with_negative(weight, sythentic_t)

    sythentic_cumulative_t = calculate_cumulative_distribution(sythentic_histogram_t)
    index_i = 0
    while index_i != len(sythentic_histogram_t):
        if sythentic_cumulative_t[index_i] >= 0.5:
            break
        index_i += 1
    return index_i + bound



@njit
def obtain_least_l1_parameters_from_weighted_mpoints_to_mpoints_with_points_simple(weight,mpoints_s,mpoints_t):

    W = np.zeros((2, mpoints_s.shape[1]))

    for j in range(weight.shape[1]):
        b_old = 0.0
        a_old = 0.1
        while True:
            # solve a

            sythentic_weight = weight[:,j]*mpoints_s[:,j]/1000
            sythentic_t = (1000*mpoints_t[:,j]/mpoints_s[:,j]).astype(np.int32)
            sythentic_histogram_t = count_weighted_points_histogram(sythentic_weight,sythentic_t)

            sythentic_cumulative_t = calculate_cumulative_distribution(sythentic_histogram_t)
            index_i = 0
            while index_i != len(sythentic_histogram_t):
                if sythentic_cumulative_t[index_i] >= 0.5:
                  break
                index_i += 1
            a_new = index_i/1000

            # solve b
            sythentic_t = ( mpoints_t[:, j] - a_new * mpoints_s[:, j]).astype(np.int32)
            sythentic_histogram_t,bound = count_weighted_points_histogram_with_negative(weight[:,j], sythentic_t)

            sythentic_cumulative_t = calculate_cumulative_distribution(sythentic_histogram_t)
            index_i = 0
            while index_i != len(sythentic_histogram_t):
                if sythentic_cumulative_t[index_i] >= 0.5:
                    break
                index_i += 1
            b_new = index_i+bound

            object_old = np.sum(weight[:,j]*np.abs(a_old*mpoints_s[:, j]+b_old-mpoints_t[:,j]))
            object_new = np.sum(weight[:, j] * np.abs(a_new * mpoints_s[:, j] + b_new - mpoints_t[:, j]))
            if object_new>=object_old:
                break
            else:
                a_old=a_new
                b_old=b_new

        W[0, j] = a_old
        W[1, j] = b_old

    return mpoints_s*W[0:1,:]+W[1:2,:],W


if __name__ == "__main__":
    tif_file_a = r"G:\2021work\unchanged-points-relative-radiometric-normalization\使用数据集\LEVICD\train\A\train_1.tiff"
    tif_file_b = r"G:\2021work\unchanged-points-relative-radiometric-normalization\使用数据集\LEVICD\train\B\train_1.tiff"

    mapped_image= histogram_matching_from_images_to_images(read_tif(tif_file_a),read_tif(tif_file_b))
    plt.figure()
    plt.imshow(mapped_image)


    plt.figure()
    plt.imshow(read_tif(tif_file_a))


    plt.figure()
    plt.imshow(read_tif(tif_file_b))
    plt.show()