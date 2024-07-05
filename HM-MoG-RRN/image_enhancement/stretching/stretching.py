import numpy as np
from image_enhancement.mask.mask import obtain_image_mask,obtain_images_mask
from tif_shp_utils.read_write_tifs import read_tif
import matplotlib.pyplot as plt

def linear_stretching_for_image(image,max_t=255.0,min_t=1.0,mask=None):
    if mask is None:
        mask = obtain_image_mask(image)
    max_value = np.max(image[mask==1])+0.0
    min_value = np.min(image[mask==1])+0.0
    new_image = np.zeros_like(image,np.uint64)
    new_image[mask==1]=(image[mask==1]-min_value)*(max_t-min_t)/(max_value-min_value)+min_t
    return new_image

def linear_stretching_for_images(images,max_t=255,min_t=1,mask=None):
    if mask is None:
        mask = obtain_images_mask(images)
    new_images = np.zeros_like(images, np.uint64)
    for c in range(new_images.shape[2]):
        new_images[:,:,c]=linear_stretching_for_image(images[:,:,c],max_t=max_t,min_t=min_t,mask=mask)
    return new_images

def get_mins_maxs_for_images(images,mask=None):
    mins = np.zeros(images.shape[2])
    maxs = np.zeros(images.shape[2])

    if mask is None:
        mask = obtain_images_mask(images)
    for c in range(images.shape[2]):
        mins[c] =np.min(images[mask == 1][:,c]) + 0.0
        maxs[c] = np.max(images[mask == 1][:,c]) + 0.0
    return mins,maxs

def recover_scale_for_images_for_1_255_linear_stretching(images,mins,maxs,mask=None):
    if mask is None:
        mask = obtain_images_mask(images)
    new_images = np.zeros_like(images, np.uint64)
    print(images[mask == 1][0,0])
    for c in range(images.shape[2]):
        new_images[mask == 1,c] = (images[mask == 1][:,c] - 1.0) * (maxs[c] - mins[c]) / (255.0 - 1.0) + mins[c]
    return new_images

def percentile_stretching_for_images(images,high_ratio,low_ratio,mask=None):
    if mask is None:
        mask = obtain_images_mask(images)
    new_images = np.zeros_like(images, np.uint64)
    for c in range(new_images.shape[2]):
        new_images[:,:,c]=percentile_stretching_for_image(images[:,:,c],high_ratio,low_ratio,mask)
    return new_images

def percentile_stretching_for_image(image,high_ratio,low_ratio,mask=None):
    high_point= np.percentile(image[mask==1],100-high_ratio)
    low_point = np.percentile(image[mask==1],low_ratio)

    new_image = np.clip(image,low_point,high_point)
    new_image[mask==0] = 0.0
    return linear_stretching_for_image(new_image,mask=mask)

if __name__=="__main__" :
    tif_file_a = r"C:\Users\gujin.LAPTOP-SMECFOOS\Desktop\New_folder\imm4695NEW\GF1C_PMS_552_resample.tif"
    image_s = read_tif(tif_file_a)
    mins_s, maxs_s = get_mins_maxs_for_images(image_s)
    image_s_recover = recover_scale_for_images_for_1_255_linear_stretching(linear_stretching_for_images(image_s), mins_s, maxs_s)
    plt.imshow(image_s_recover[:,:,:3][:,:,::-1]/5000)
    plt.show()