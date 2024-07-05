import numpy as np
from tif_shp_utils.read_write_tifs import get_geo,read_tif
from numba import njit

@njit
def obtain_x_y_from_geo_location(x_geo,y_geo,geo):
    x = int(np.round((x_geo-geo[0])/geo[1]))
    y = int(np.round((y_geo-geo[3])/geo[5]))
    return x,y


def generate_change_point_image(location, probability, geo, rows, cols):
    change_point_image = np.zeros((rows, cols))
    for i in range(len(location)):

        if probability[i,0] != np.max(probability[i,:]):
            col,row = obtain_x_y_from_geo_location(location[i,0],location[i,1],geo)
            if col>=0 and col<cols and row>=0 and row<rows:
                change_point_image[row,col]= np.argmax(probability[i,:])

    return change_point_image/(probability.shape[1]-1)

@njit
def obtain_location_from_x_y_geo(x,y,geo):
    x_geo = geo[0] + x * geo[1] + y * geo[2]
    y_geo = geo[3] + x * geo[4] + y * geo[5]
    return x_geo, y_geo

@njit
def obtain_x_new_y_new_from_x_y_with_different_geo(x,y,geo,geo_new):
    x_geo, y_geo = obtain_location_from_x_y_geo(x,y,geo)
    x_new, y_new = obtain_x_y_from_geo_location((x_geo,y_geo,geo_new))
    return x_new,y_new

@njit
def obtain_intersection_area(images_s,geo_s,images_t,geo_t):
    print(geo_s[0], geo_t[0])
    left_up_geo_0 = max(geo_s[0], geo_t[0])
    left_up_geo_3 = min(geo_s[3], geo_t[3])

    x_s_geo, y_s_geo = obtain_location_from_x_y_geo(images_s.shape[0],images_s.shape[1],geo_s)
    x_t_geo, y_t_geo = obtain_location_from_x_y_geo(images_t.shape[0], images_t.shape[1], geo_t)

    down_right_geo_0 = min(x_s_geo,x_t_geo)
    down_right_geo_3 = max(y_s_geo,y_t_geo)

    if (down_right_geo_0-left_up_geo_0)*geo_s[1]<0 or (down_right_geo_3-left_up_geo_3)*geo_s[5]<0:
        raise ValueError

    x_start_s,y_start_s = obtain_x_y_from_geo_location(left_up_geo_0,left_up_geo_3,geo_s)
    x_end_s,y_end_s = obtain_x_y_from_geo_location(down_right_geo_0,down_right_geo_3,geo_s)
    geo_s_o = (left_up_geo_0,geo_s[1],geo_s[2],left_up_geo_3,geo_s[4],geo_s[5])


    x_start_t,y_start_t = obtain_x_y_from_geo_location(left_up_geo_0,left_up_geo_3,geo_t)
    x_end_t,y_end_t = obtain_x_y_from_geo_location(down_right_geo_0,down_right_geo_3,geo_t)
    geo_t_o = (left_up_geo_0, geo_t[1], geo_t[2], left_up_geo_3, geo_t[4], geo_t[5])

    return images_s[x_start_s:x_end_s,y_start_s:y_end_s],geo_s_o,images_t[x_start_t:x_end_t,y_start_t:y_end_t],geo_t_o

@njit
def obtain_corresponding_point_and_location(images_s_o,geo_s_o,images_t_o,geo_t_o):
    points_s = np.zeros((images_s_o.shape[0]*images_s_o.shape[1],images_s_o.shape[2]))
    points_t = np.zeros((images_s_o.shape[0] * images_s_o.shape[1], images_s_o.shape[2]))
    locations = np.zeros((images_s_o.shape[0]*images_s_o.shape[1],2))
    count = 0
    for i_s in range(images_s_o.shape[0]):
        for j_s in range(images_s_o.shape[1]):
            # i_t,j_t = obtain_x_new_y_new_from_x_y_with_different_geo(i_s,j_s,geo_s_o,geo_t_o)
            x_geo, y_geo = obtain_location_from_x_y_geo(j_s, i_s, geo_s_o)
            j_t, i_t = obtain_x_y_from_geo_location(x_geo, y_geo, geo_t_o)

            if i_t>=0 and i_t<=images_t_o.shape[0] and j_t>=0 and j_t<=images_t_o.shape[1]:
                if np.sum(images_s_o[i_s,j_s])!=0 and np.sum(images_t_o[i_t,j_t])!=0:
                    points_s[count] = images_s_o[i_s,j_s]
                    points_t[count] = images_t_o[i_t,j_t]
                    locations[count,0] = x_geo
                    locations[count,1] = y_geo
                    count += 1
    return points_s[:count], points_t[:count], locations[:count]

if __name__=="__main__":
    tif_file_a = r"G:\2021work\unchanged-points-relative-radiometric-normalization\使用数据集\LEVICD\train\A\train_1.tiff"
    tif_file_b = r"G:\2021work\unchanged-points-relative-radiometric-normalization\使用数据集\LEVICD\train\B\train_1.tiff"

    obtain_intersection_area(read_tif(tif_file_a),get_geo(tif_file_a), read_tif(tif_file_b),get_geo(tif_file_b))

    obtain_corresponding_point_and_location(read_tif(tif_file_a), get_geo(tif_file_a), read_tif(tif_file_b), get_geo(tif_file_b))