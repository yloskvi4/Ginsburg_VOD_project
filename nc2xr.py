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

#comb = xr.open_dataset('/burg/glab/users/os2328/data/YAOSM/yaoSM_xr.zarr')
#ds = ds0.sortby('time')
#print(ds.time[0])
#print(ds.time[-1])
#vod = ds0['Optical_Thickness_Nad']

#ds_mean = vod.mean(dim=(['lat', 'lon']))
#ds_mean = ds_mean.sortby('time')


#ds_mean.to_netcdf('/burg/glab/users/os2328/data/VOD_project/initial_vod_ts.nc')
#ds_mean = vod.mean(dim=('lon'))

#ds_mean = ds_mean.sortby('time')

#ds_mean.to_netcdf('/burg/glab/users/os2328/data/VOD_project/initial_vod_hov.nc')

#ds = ds0.sortby('time')
#print(ds.time[0])
#print(ds.time[-1])

#quit()
###


def nc2zar(path_to_files, file_name_starts_with, is_time_dimention, output_name, ymd = 1):


    files_list = [path_to_files+f for f in os.listdir(path_to_files) if f.startswith('IB_SM_CDF3AM_')]
    #print(files_list)
    print(len(files_list))
    def one_nc2xr(file_, path_to_files):
        timestep_ds = xr.open_dataset(file_)
        from_name = file_.split(path_to_files+file_name_starts_with,1)[1]
        if ymd:
            times = datetime.datetime.strptime(from_name[:8], '%Y%m%d')
        else:
            times = datetime.datetime.strptime(from_name[:7], '%Y%j')
        dst = timestep_ds.expand_dims("time").assign_coords(time=("time",[times]))
        return dst

    if is_time_dimention:

        ds0 = xr.open_dataset(files_list[0])
        ds1 = xr.open_dataset(files_list[1])
    else:

        ds0 = one_nc2xr(files_list[0], path_to_files)
        ds1 = one_nc2xr(files_list[1], path_to_files)

    comb = xr.concat([ds0, ds1], dim="time")

    
    for ib in files_list[2:]:

        if is_time_dimention:
            ds = xr.open_dataset(ib)
        else:
            ds = one_nc2xr(ib, path_to_files)

        comb = xr.concat([comb, ds], dim="time")

    print(comb.info())

    comb.to_zarr(path_to_files + output_name + ".zarr")
    print('the new version 2 vod  zarr file is ready')
    return comb


def mean_xr2nc(xr_data, dim_to_mean, output_name):

    ds_mean = xr_data.mean(dim=(dim_to_mean))

    if dim_to_mean == 'time':
        ds_mean.to_netcdf(path_to_files + output_name + "_map.nc")
    else:
        ds_mean = ds_mean.sortby('time')
        ds_mean.to_netcdf(path_to_files + output_name + ".nc")

    return print('a new averaged dataset is saved')



path_to_files = '/burg/glab/users/os2328/data/VOD_project/VOD-IB/'
file_name_starts_with = 'IB_SM_CDF3AM_'
is_time_dimention = 0
output_name = 'smap_ib_xr_ver2'
#comb = xr.open_dataset('/burg/glab/users/os2328/data/VOD_project/smos_ic_xr.zarr')

#comb = comb[['BT_H_IC_Fitted', 'BT_V_IC_Fitted','Scene_Flags' ]]
comb = nc2zar(path_to_files, file_name_starts_with, is_time_dimention, output_name, ymd = 1)

#c2013 = xr.open_dataset('/burg/glab/users/os2328/data/VOD_project/SMOS-IC/smos_ic_xr2013.zarr')
#c2013 = c2013[['BT_H_IC_Fitted', 'BT_V_IC_Fitted','Scene_Flags' ]]

#comb = xr.concat([comb, c2013], dim="time")

#c13 = comb.sel(time=comb.time.dt.year.isin([2013]))
#print(np.unique(c13.time.dt.month))

#try:
#    comb.to_zarr('/burg/glab/users/os2328/data/smos_ic_xr.zarr')
#    print('in data')
#except:
#    comb.to_zarr('/burg/glab/users/os2328/data/VOD_project/smos_ic_xr.zarr')
#    print('in vod project')
#print('new full file is saved')

mean_xr2nc(comb, 'time', output_name)
mean_xr2nc(comb, 'lon', output_name+'_hov')
mean_xr2nc(comb, ['lat', 'lon'], output_name+'_ts')
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

one.to_pickle('/burg/glab/users/os2328/data/YAOSM/3dmsm.pkl', protocol = 4)


