# from pyramids.dataset import Dataset
from digitalrivers.terrain import Terrain
from osgeo_utils import gdal_calc
import pandas as pd
# path = r"\\MYCLOUDEX2ULTRA\satellite-data\landsat\lake-taho"
path = r"examples\data\landsat\lake-taho"
#%%
color_palette = pd.read_csv(f"{path}/beige_green.txt")
# %%
b2 = Terrain.read_file(rf"{path}\LC08_L2SP_043033_20210922_20210930_02_T1_SR_B2.TIF")
b3 = Terrain.read_file(rf"{path}\LC08_L2SP_043033_20210922_20210930_02_T1_SR_B3.TIF")
b5 = Terrain.read_file(rf"{path}\LC08_L2SP_043033_20210922_20210930_02_T1_SR_B5.TIF")
b6 = Terrain.read_file(rf"{path}\LC08_L2SP_043033_20210922_20210930_02_T1_SR_B6.TIF")
b7 = Terrain.read_file(rf"{path}\LC08_L2SP_043033_20210922_20210930_02_T1_SR_B7.TIF")
#%%
"""
gdal_calc.py 
-B tahoe_LC08_20210922_B2_SR_float.tif 
-C tahoe_LC08_20210922_B3_SR_float.tif 
-E tahoe_LC08_20210922_B5_SR_float.tif 
-F tahoe_LC08_20210922_B6_SR_float.tif 
-G tahoe_LC08_20210922_B7_SR_float.tif

tahoe_LC08_20210922_SR_AWEIsh
"""
# %%
"--calc and enclosed in double quotes. The operations are performed on the bands defined previously, and can be any function available in NumPy"
# gdal_calc.py will write the output file in the same format as the input files.
aweish = gdal_calc.Calc(
    calc="B  + 2.5 * C - 1.5 * (E + F) - 0.25 * G",
    B=b2.raster, C=b3.raster, E=b5.raster, F=b6.raster, G=b7.raster,
    format="MEM",
    outfile="", type="Float32", creation_options=["COMPRESS=DEFLATE", "PREDICTOR=2"]
)
#%%
aweish = Terrain(aweish)
print(aweish)
aweish.stats()
# color_scale="boundary-norm", bounds=[0, 0.2, 0.4, 0.6, 0.8, 1]
aweish.plot(color_scale="linear") #vmin=0, vmax=1,

# aweish = aweish.change_no_data_value(-9999, aweish.no_data_value[0])

aweish.to_file(rf"{path}\tahoe_LC08_20210922_SR_NDVI.tif")
color_relief = aweish.color_relief(band=0, color_table=color_palette)
color_relief.to_file(f"{path}/tahoe_LC08_20210922_SR_NDVI_color_relief.tif")
#%%
# ndvi = Terrain.read_file(r"examples\data\landsat\lake-taho\tahoe_LC08_20210922_SR_NDVI.tif", read_only=False)

color_relief.no_data_value = 0
color_relief.plot(rgb=[0, 1, 2, 3])
color_relief.read_array(band=0, window=[0, 0, 5, 5])

#%%
import inspect
import string

def my_function(x, y):
    return x + y

# List of input TIFF files
tif_files = ['input1.tif', 'input2.tif']
# Output TIFF file
output_file = 'output.tif'


# Extract the function's source code
function_source = inspect.getsource(my_function)

# Convert the function logic into a string expression
# Assuming the function is simple and contains only the return statement

# Remove the function definition line
expression = function_source.split("return")[-1].strip()
# Step 3: Extract parameter names from the user-defined function
param_names = inspect.signature(my_function).parameters.keys()

if len(param_names) != len(tif_files):
    raise ValueError("Number of function parameters and number of input TIFF files must match.")

# Step 4: Map the TIFF files to the parameter names
tif_mapping = {name: tif_files[i] for i, name in enumerate(param_names)}

# Step 4: Prepare the parameters for gdal_calc.Calc
calc_params = {
    'calc': expression,
    'outfile': output_file
}
calc_params.update(tif_mapping)

