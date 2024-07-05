import numpy as np
from numba import njit
from image_enhancement.histogram.histogram import count_weighted_points_histogram,calculate_cumulative_distribution,count_weighted_points_histogram_with_negative
def obtain_least_square_parameters_from_points(mpoints_s,mpoints_t):
    # mpoints_s_1 = np.concatenate([mpoints_s,np.ones((len(mpoints_s),1))],axis=1)
    # mpoints_s_1_t = np.transpose(mpoints_s_1)
    # mpoints_s_1_t_mpoints_t = mpoints_s_1_t.dot(mpoints_t)
    # parameters = np.linalg.pinv(mpoints_s_1_t.dot(mpoints_s_1)).dot(mpoints_s_1_t_mpoints_t)
    W = np.zeros((2, mpoints_s.shape[1]))
    A = np.ones((mpoints_s.shape[0], 1 + 1))
    for j in range(mpoints_s.shape[1]):
        A[:, 0:1] = mpoints_s[:,j:j+1]
        GjY =  mpoints_t[:, j:j+1]
        # sum_matrix = np.zeros((mpoints_s.shape[1] + 1, mpoints_s.shape[1]+1))
        # for i in range(weight.shape[0]):
        #     sum_matrix += A[i:i+1, :].transpose().dot(weight[i, j] * A[i:i+1, :])
        GjA =   A
        W[:, j:j+1] = np.linalg.inv(A.transpose().dot(GjA)).dot(A.transpose().dot(GjY))
    return W

@njit
def obtain_least_square_parameters_from_weighted_mpoints_to_mpoints(weight,mpoints_s,mpoints_t):

    A = np.ones((mpoints_s.shape[0],mpoints_s.shape[1]+1))
    A[:,:mpoints_s.shape[1]]=mpoints_s
    GA = weight*A
    W = (np.linalg.inv((A.transpose()).dot(GA))).dot((A.transpose()).dot(weight*mpoints_t))


    return W

@njit
def least_square_mapping_from_weighted_mpoints_to_mpoints(weight,mpoints_s,mpoints_t):
    A = np.ones((mpoints_s.shape[0], mpoints_s.shape[1] + 1))
    A[:, :mpoints_s.shape[1]] = mpoints_s
    W = np.zeros((mpoints_s.shape[1] + 1, mpoints_s.shape[1]))
    for j in range(weight.shape[1]):
        GjY = weight[:, j:j+1] * mpoints_t[:, j:j+1]
        sum_matrix = np.zeros((mpoints_s.shape[1] + 1, mpoints_s.shape[1]+1))
        for i in range(weight.shape[0]):
            sum_matrix += A[i:i+1, :].transpose().dot(weight[i, j] * A[i:i+1, :])

        W[:, j:j+1] = np.linalg.inv(sum_matrix).dot(A.transpose().dot(GjY))
    print((A.dot(W)).shape)
    return A.dot(W)


@njit
def obtain_least_square_parameters_from_weighted_mpoints_to_mpoints_with_points(weight,mpoints_s,mpoints_t):
    A = np.ones((mpoints_s.shape[0], mpoints_s.shape[1] + 1))
    A[:, :mpoints_s.shape[1]] = mpoints_s
    W = np.zeros((mpoints_s.shape[1] + 1, mpoints_s.shape[1]))
    for j in range(weight.shape[1]):
        GjY = weight[:, j:j+1] * mpoints_t[:, j:j+1]
        sum_matrix = np.zeros((mpoints_s.shape[1] + 1, mpoints_s.shape[1]+1))
        for i in range(weight.shape[0]):
            sum_matrix += A[i:i+1, :].transpose().dot(weight[i, j] * A[i:i+1, :])

        W[:, j:j+1] = np.linalg.inv(sum_matrix).dot(A.transpose().dot(GjY))
    return A.dot(W),W

@njit
def least_square_mapping_from_weighted_mpoints_to_mpoints_simple(weight,mpoints_s,mpoints_t):
    W = np.zeros((2, mpoints_s.shape[1]))
    A = np.ones((mpoints_s.shape[0], 1 + 1))
    for j in range(weight.shape[1]):

        A[:, 0:1] = mpoints_s[:,j:j+1]
        GjY = weight[:, j:j+1] * mpoints_t[:, j:j+1]
        # sum_matrix = np.zeros((mpoints_s.shape[1] + 1, mpoints_s.shape[1]+1))
        # for i in range(weight.shape[0]):
        #     sum_matrix += A[i:i+1, :].transpose().dot(weight[i, j] * A[i:i+1, :])
        GjA = weight[:, j:j+1] *  A
        W[:, j:j+1] = np.linalg.inv(A.transpose().dot(GjA)).dot(A.transpose().dot(GjY))
        # if W[0,j]<0.2:
        #     W[1, j] = np.sum(GjY -0.2*weight[:, j:j+1]*mpoints_s[:,j:j+1])/np.sum(weight[:, j:j+1])
        # W[0, j:j + 1] = np.clip(W[0, j:j + 1], 0.2, 5)
    return mpoints_s*W[0:1,:]+W[1:2,:]


@njit
def obtain_least_square_parameters_from_weighted_mpoints_to_mpoints_with_points_simple(weight,mpoints_s,mpoints_t):
    W = np.zeros((2, mpoints_s.shape[1]))
    A = np.ones((mpoints_s.shape[0], 1 + 1))
    for j in range(weight.shape[1]):

        A[:, 0:1] = mpoints_s[:,j:j+1]
        GjY = weight[:, j:j+1] * mpoints_t[:, j:j+1]
        # sum_matrix = np.zeros((mpoints_s.shape[1] + 1, mpoints_s.shape[1]+1))
        # for i in range(weight.shape[0]):
        #     sum_matrix += A[i:i+1, :].transpose().dot(weight[i, j] * A[i:i+1, :])
        GjA = weight[:, j:j+1] *  A
        W[:, j:j+1] = np.linalg.inv(A.transpose().dot(GjA)).dot(A.transpose().dot(GjY))
        # if W[0,j]<0.2:
        #     W[1, j] = np.sum(GjY -0.2*weight[:, j:j+1]*mpoints_s[:,j:j+1])/np.sum(weight[:, j:j+1])
        # W[0, j:j + 1] = np.clip(W[0, j:j + 1], 0.2, 5)
    return mpoints_s*W[0:1,:]+W[1:2,:],W


def apply_regression_parameters_for_points(mpoints_s,parameters):
    mpoints_s_1 = np.concatenate([mpoints_s, np.ones((len(mpoints_s), 1))],axis=1)
    return mpoints_s_1.dot(parameters)

def apply_regression_parameters_for_images(images_s,parameters):
    # images_s_1 = np.concatenate([images_s,np.ones((images_s.shape[0],images_s.shape[1],1))],axis=2)
    # images_regressed = (images_s_1.reshape(images_s.shape[0]*images_s.shape[1],-1)).dot(parameters)
    # images_regressed = images_regressed.reshape(images_s.shape[0],images_s.shape[1],-1)
    return images_s*parameters[0:1,:]+parameters[1:2,:]
