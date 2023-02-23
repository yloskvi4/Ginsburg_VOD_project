# this script creates an XARRAY from multiple netcdf files and saves it as .zarr

import pandas as pd
import numpy as np
import xarray as xr
import zarr
from datetime import datetime
import netCDF4
import datetime
import os

def mean_xr2nc(xr_data, dim_to_mean, output_name):

    ds_mean = xr_data.mean(dim=(dim_to_mean))

    if dim_to_mean == 'time':
        ds_mean.to_netcdf('/burg/glab/users/os2328/data/' + output_name + "_map.nc")
    else:
        ds_mean = ds_mean.sortby('time')
        ds_mean.to_netcdf('/burg/glab/users/os2328/data/' + output_name + ".nc")

    return print('a new averaged dataset is saved')

#path_to_files = '/burg/glab/users/rg3390/data/SMOS/LVOD/daily/'
# 2014-2021

#years = range(2016, 2021)
#smos_dir = []
#for year in years:
#    year_dir = path_to_files + str(year) + '_ASC/'
#    smos_dir.append(year_dir)

#SMOS_list_full = []
#for d in smos_dir:
#    SMOS_list = [d+f for f in os.listdir(d) if f.endswith('.nc')]
#    SMOS_list_full = SMOS_list_full + SMOS_list

#file_name_starts_with = 'SM_RE06_MIR_CDF3SA_'
#is_time_dimention = 0
output_name = 'SMOS_IC_xr'


#timestep_ds = xr.open_dataset( SMOS_list_full[0])
#file_ = SMOS_list_full[0]

#from_name = file_.split('_B.DBL.nc',1)[0]

#times = datetime.datetime.strptime(from_name[-23:-15], '%Y%m%d')
#ds0 = timestep_ds.expand_dims("time").assign_coords(time=("time",[times]))

#timestep_ds = xr.open_dataset( SMOS_list_full[1])
#file_ = SMOS_list_full[1]

#from_name = file_.split('_B.DBL.nc',1)[0]
        
#times = datetime.datetime.strptime(from_name[-23:-15], '%Y%m%d')
#ds1 = timestep_ds.expand_dims("time").assign_coords(time=("time",[times]))

#comb = xr.concat([ds0, ds1], dim="time")

#for ib in SMOS_list_full[2:]:

#    timestep_ds = xr.open_dataset(ib)
#    file_ = ib
    
#    from_name = file_.split('_B.DBL.nc',1)[0]
    
#    times = datetime.datetime.strptime(from_name[-23:-15], '%Y%m%d')
#    ds = timestep_ds.expand_dims("time").assign_coords(time=("time",[times]))

#    comb = xr.concat([comb, ds], dim="time")

comb = xr.open_dataset('/burg/glab/users/os2328/data/SMOS_IC_xr.zarr')
#ds1 = xr.open_dataset('/burg/glab/users/os2328/data/SMOS_IC_xr2016-21.zarr')
#comb = xr.concat([ds0, ds1], dim="time")




#print(comb.info())

#comb.to_zarr('/burg/glab/users/os2328/data/SMOS_IC_xr.zarr')


mean_xr2nc(comb, 'time', output_name)
comb = comb.sortby('time')
one_loc = comb.sel(lat=-51.90, lon =-71.45, method="nearest")
one_loc.to_netcdf("/burg/glab/users/os2328/data/vod-51-71_smosic.nc")
print('1 ts saved')

one_loc2 = comb.sel(lat=45.41, lon =0.38, method="nearest")
one_loc2.to_netcdf("/burg/glab/users/os2328/data/vod45038_smosic.nc")
print('2 ts saved')



try:
    one = comb.resample(time='1M').mean(dim='time')

    mean_xr2nc(one, 'lon', output_name+'_hov')
except:
    print('resemple didnt work')
    one = comb.to_dataframe()
    one = one.reset_index()
    one = one.dropna()
    one = one.groupby([pd.Grouper(key='time', freq='1M'), 'lat', 'lon']).mean()

    one.to_pickle('/burg/glab/users/os2328/data/1m_mean_smos_ic_vod.pkl', protocol = 4)

#one_loc = comb.sel(lat=-51.90, lon =-71.45, method="nearest")
#one_loc.to_netcdf("/burg/glab/users/os2328/data/vod-51-71_smosic.nc")

#one_loc2 = comb.sel(lat=45.41, lon =0.38, method="nearest")
#one_loc2.to_netcdf("/burg/glab/users/os2328/data/vod45038_smosic.nc")


quit()
#mean_xr2nc(comb, ['lat', 'lon'], output_name+'_ts')


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


