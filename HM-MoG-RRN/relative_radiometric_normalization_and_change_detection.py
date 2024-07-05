import time
from image_enhancement.change.relative_radiometric_normalization import relative_radiometric_normalization_via_change_noise_model
from tif_shp_utils.read_write_tifs import read_tif,get_geo
from image_enhancement.stretching.stretching import get_mins_maxs_for_images,recover_scale_for_images_for_1_255_linear_stretching,linear_stretching_for_images,percentile_stretching_for_images
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


tif_file_a = r"GF1C_PMS_127_0.tif"#subject image
tif_file_b = r"GF1C_PMS_552_resample_0.tif" #reference image

image_s = read_tif(tif_file_a).astype(np.int32)
image_t = read_tif(tif_file_b).astype(np.int32)

max_factor = max(np.max(image_s),np.max(image_t))

mins_s,maxs_s = get_mins_maxs_for_images(image_s)
mins_t, maxs_t = get_mins_maxs_for_images(image_t)

t0 = time.time()
recover_mapped_image,change_point_image = relative_radiometric_normalization_via_change_noise_model(image_s, image_t, get_geo(tif_file_a), get_geo(tif_file_b),disable_bias=True,number_of_noise=2,mapping='HM',hypothesis="MoG")
t1 = time.time()
print("time cost",(t1-t0)/60)

plt.figure()
plt.imshow(recover_mapped_image[:,:,:3][:,:,::-1]/max_factor)
plt.imsave(r"mapped_image_GF1C_PMS_127_0.jpg",np.clip(recover_mapped_image[:,:,:3][:,:,::-1]/max_factor,0,1), dpi = matplotlib.rcParams['figure.dpi'])
plt.title("mapped image")
print(maxs_s)
print(maxs_t)

plt.figure()
plt.imshow(image_s[:,:,:3][:,:,::-1]/max_factor)
plt.title("subject image")
plt.figure()
plt.imshow(image_t[:,:,:3][:,:,::-1]/max_factor)

plt.title("reference image")
plt.figure()
plt.imshow(change_point_image,cmap="gray")
plt.imsave(r"change_point_GF1C_PMS_127_0.jpg",change_point_image,cmap='gray', dpi = matplotlib.rcParams['figure.dpi'])
plt.title("no-change point")


plt.show()