import numpy as np
from tif_shp_utils.read_write_tifs import read_tif
from numba import njit

@njit
def obtain_images_mask(images):
    '''
    影像上的所有波段上有一个无效值，这个位置就被无效掩膜覆盖。
    :param images: W*H*C
    :return: 掩膜1表示有效值，0表示无效值
    '''
    shape = images.shape
    mask = np.zeros((shape[0],shape[1]), np.uint8)
    for w in range(shape[0]):
        for h in range(shape[1]):
            mask[w,h]=1
            for c in range(shape[2]):
                if images[w,h,c]==0 or images[w,h,c]==None:
                    mask[w,h]=0
    return mask


@njit
def obtain_image_mask(image):
    '''
    影像上的所有波段上有一个无效值，这个位置就被无效掩膜覆盖。
    :param images: W*H
    :return: 掩膜1表示有效值，0表示无效值
    '''
    shape = image.shape
    mask = np.zeros((shape[0],shape[1]), np.uint8)
    print(shape)
    if len(shape)==2:
        for w in range(shape[0]):
            for h in range(shape[1]):
                mask[w, h] = 1
                if image[w, h] == 0 or image[w, h] == None:
                    mask[w, h] = 0
    return mask

@njit
def obtain_mask(images):
    '''
    影像上的所有波段上有一个无效值，这个位置就被无效掩膜覆盖。
    :param images: W*H*C
    :return: 掩膜1表示有效值，0表示无效值
    '''
    shape = images.shape
    mask = np.zeros((shape[0],shape[1]), np.uint8)
    if len(shape)==2:
        obtain_image_mask(images)
    else:
        obtain_images_mask(images)
    return mask


if __name__== "__main__":
    tif_file = r"G:\2021work\unchanged-points-relative-radiometric-normalization\使用数据集\LEVICD\train\A\train_1.tiff"

    print(obtain_mask(read_tif(tif_file)).shape)