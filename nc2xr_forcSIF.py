# this script creates an XARRAY from multiple netcdf files and saves it as .zarr

import pandas as pd
import numpy as np
import xarray as xr
import zarr
from datetime import datetime
import netCDF4
import datetime
import os

#####

path_to_files = '/burg/glab/users/os2328/data/cSIF/'

xr1_f = path_to_files + 'cSIF_xr0-500.zarr'
xr2_f = path_to_files + 'cSIF_xr1000_end.zarr'
xr3_f = path_to_files + 'cSIF_xr1000.zarr'

df_1 = xr.open_dataset(xr1_f)
#print(df_1.info)
df_2 = xr.open_dataset(xr2_f)
df_3 = xr.open_dataset(xr3_f)
#print(df_2.info)
#print(df_3.info)

comb = xr.concat([df_1, df_2, df_3], dim="time")
#print(comb.info)


files_list = [path_to_files + 'OCO2.SIF.clear.inst.2007001.v2.nc', path_to_files + 'OCO2.SIF.clear.inst.2007005.v2.nc',path_to_files + 'OCO2.SIF.clear.inst.2007009.v2.nc', path_to_files + 'OCO2.SIF.clear.inst.2007013.v2.nc', path_to_files + 'OCO2.SIF.clear.inst.2007017.v2.nc', path_to_files + 'OCO2.SIF.clear.inst.2007021.v2.nc', path_to_files + 'OCO2.SIF.clear.inst.2007025.v2.nc', path_to_files + 'OCO2.SIF.clear.inst.2007029.v2.nc', path_to_files + 'OCO2.SIF.clear.inst.2007033.v2.nc', path_to_files + 'OCO2.SIF.clear.inst.2007037.v2.nc', path_to_files + 'OCO2.SIF.clear.inst.2007057.v2.nc', path_to_files + 'OCO2.SIF.clear.inst.2007065.v2.nc', path_to_files + 'OCO2.SIF.clear.inst.2019365.v2.nc']

ds0 = xr.open_dataset(files_list[0])
ds1 = xr.open_dataset(files_list[1])
from_name = files_list[0].split(path_to_files+'OCO2.SIF.clear.inst.',1)[1]
times = datetime.datetime.strptime(from_name[:7], '%Y%j')
dst0 = ds0.expand_dims("time").assign_coords(time=("time",[times]))

from_name = files_list[1].split(path_to_files+'OCO2.SIF.clear.inst.',1)[1]
times = datetime.datetime.strptime(from_name[:7], '%Y%j')
dst1 = ds1.expand_dims("time").assign_coords(time=("time",[times]))

comb_add = xr.concat([dst0, dst1], dim="time")
#print(comb_add.info)


files_list = [path_to_files+f for f in os.listdir(path_to_files) if f.startswith('OCO2.SIF.clear.inst.2020')]
#print(files_list)
#print(len(files_list))
for ib in files_list:
    timestep_ds = xr.open_dataset(ib)
    from_name = ib.split(path_to_files+'OCO2.SIF.clear.inst.',1)[1]
    times = datetime.datetime.strptime(from_name[:7], '%Y%j')
    dst = timestep_ds.expand_dims("time").assign_coords(time=("time",[times]))

    comb_add = xr.concat([comb_add, dst], dim="time")

comb = xr.concat([comb, comb_add], dim="time")
print(comb.info)
comb = comb[['clear_inst_SIF', 'clear_daily_SIF']]



encoding = {'lat': { '_FillValue': None},
            'lon': { '_FillValue': None},
            'time':{ '_FillValue': None},
            'clear_inst_SIF':{  '_FillValue': np.nan},
            'clear_daily_SIF':{ '_FillValue': np.nan}
            }




comb.to_zarr('/burg/glab/users/os2328/data/cSIF/cSIF_full.zarr', encoding=encoding)



#res_flag = 0
#try:
#    one = comb.resample(time='3D').mean(dim='time')
#except:
#    print('xarray resample didnt work')
#    res_flag = 1

#one = comb.to_dataframe()
#one = one.reset_index()
#one = one.dropna()
#if res_flag:
#    print('resampling in pd')
#    one = one.groupby([pd.Grouper(key='time', freq='3D'), 'lat', 'lon']).mean()

#one.to_pickle('/burg/glab/users/os2328/data/cSIF/3dmSIF.pkl', protocol = 4)



def mean_xr2nc(xr_data, dim_to_mean, output_name):

    ds_mean = xr_data.mean(dim=(dim_to_mean))

    if dim_to_mean == 'time':
        ds_mean.to_netcdf(path_to_files + output_name + "_map.nc")
    else:
        ds_mean = ds_mean.sortby('time')
        ds_mean.to_netcdf(path_to_files + output_name + ".nc")

    return print('a new averaged dataset is saved')
output_name = 'cSIF_xr'


mean_xr2nc(comb, 'time', output_name)
mean_xr2nc(comb, 'lon', output_name+'_hov')
mean_xr2nc(comb, ['lat', 'lon'], output_name+'_ts')


min_lon = -10
min_lat = 36
max_lon = 35
max_lat = 54

mask_lon = (comb.lon >= min_lon) & (comb.lon <= max_lon)
mask_lat = (comb.lat >= min_lat) & (comb.lat <= max_lat)

cropped_ds = comb.where(mask_lon & mask_lat, drop=True)
#cci_ts = cropped_ds.mean(dim=('lat', 'lon'))
cropped_ds.to_netcdf(path_to_files + "cSIF_only_europe_xr.nc")

min_lon = 9
min_lat = 30
max_lon = 40
max_lat = 45
    
mask_lon = (comb.lon >= min_lon) & (comb.lon <= max_lon)
mask_lat = (comb.lat >= min_lat) & (comb.lat <= max_lat)
    
cropped_ds = comb.where(mask_lon & mask_lat, drop=True)
#cci_ts = cropped_ds.mean(dim=('lat', 'lon'))
cropped_ds.to_netcdf(path_to_files + "cSIF_only_mediterr_xr.nc")


quit()


res_flag = 0
try:
    one = comb.resample(time='3D').mean(dim='time')
except:
    print('xarray resample didnt work')
    res_flag = 1

one = comb.to_dataframe()
one = one.reset_index()
one = one.dropna()
if res_flag:
    print('resampling in pd')
    one = one.groupby([pd.Grouper(key='time', freq='3D'), 'lat', 'lon']).mean()

one.to_pickle('/burg/glab/users/os2328/data/cSIF/3dmSIF.pkl', protocol = 4)


