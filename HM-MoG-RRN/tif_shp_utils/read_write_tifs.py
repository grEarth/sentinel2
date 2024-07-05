try:
    import  gdal
except:
    from osgeo import gdal
def read_tif(tif_file):
    tif_ds = gdal.Open(tif_file)
    image_width = tif_ds.RasterXSize
    image_height = tif_ds.RasterYSize
    image_band = tif_ds.RasterCount
    image = None
    if image_band == 1:
        image = tif_ds.ReadAsArray(0,0,image_width,image_height)
    if image_band >1:
        image = tif_ds.ReadAsArray(0, 0, image_width, image_height).transpose([1,2,0])
    return image

def get_geo(tif_file):
    tif_ds = gdal.Open(tif_file)
    return tif_ds.GetGeoTransform()


if __name__=="__main__":
    tif_file = r"G:\2021work\unchanged-points-relative-radiometric-normalization\使用数据集\LEVICD\train\A\train_1.tiff"
    print(read_tif(tif_file).shape)
    print(get_geo(tif_file))