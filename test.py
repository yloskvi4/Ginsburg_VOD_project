import numpy as np
import pandas as pd
import xarray as xr


ds = xr.open_dataset('/burg/glab/users/os2328/data/VOD_project/smos_ic_xr_vod_ver2_andcasm_sc.zarr')
#ds = xr.open_dataset('/burg/glab/users/os2328/data/SMOS_IC_xr.zarr')
#ds = ds.to_dataframe()
#ds = ds.reset_index()
#print(ds.head())
#min_lon = -84.41 
#min_lat = 17.68 
#max_lon = -65.64 
#max_lat = 23.16 
#try:
#    ds = ds.sortby(['time', 'lat', 'lon'])
#except:
#    print('sort by all 3 didnt work')
#ds = ds.sortby('time')
#ds = ds[['time', 'lat', 'lon', 'Optical_Thickness_Nad']]
#ds = ds.dropna()

#cropped = ds[ds['lat']<max_lat]
#cropped = cropped[cropped['lat']>min_lat]
#cropped = cropped[cropped['lon']>min_lon]
#cropped = cropped[cropped['lon']<max_lon]
#print(cropped.head())
#print(cropped.shape)
#timed = cropped[cropped['time']>= '2017-08-30']
#before = timed[timed['time']<= '2017-09-03']

#before.to_pickle('/burg/glab/users/os2328/data/VOD_project/before_irma_smos.pkl', protocol = 4)

#timed = timed[timed['time']>= '2017-09-10']
#after = timed[timed['time']<= '2017-09-15']

#after.to_pickle('/burg/glab/users/os2328/data/VOD_project/after_irma_smos.pkl', protocol = 4)

#cropped_ds = ds.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon))

#avr_before = cropped_ds.sel(
#    time=slice('2017-08-30', '2017-09-03'))
#avr_before.to_netcdf('/burg/glab/users/os2328/data/VOD_project/before_irma.nc')

#avr_after = cropped_ds.sel(
#    time=slice('2017-09-10', '2017-09-15'))
#avr_after.to_netcdf('/burg/glab/users/os2328/data/VOD_project/after_irma.nc')

#quit()

#one_loc = ds.sel(lat = -55.177, lon = -68.602, method='nearest')

#one_loc.to_netcdf('/burg/glab/users/os2328/data/VOD_project/one_loc2.nc')
one = ds.get(['BT_H_IC_Fitted','BT_V_IC_Fitted', 'vod', 'casm', 'seas'])
one = one.sortby('time')

res_flag = 0
try:
    one = one.resample(time='3D').mean(dim='time')
except:
    print('xarray resample didnt work')
    res_flag = 1

one = one.to_dataframe()
one = one.reset_index()
one = one.dropna()
if res_flag:
    print('resampling in pd')
    one = one.groupby([pd.Grouper(key='time', freq='3D'), 'lat', 'lon']).mean()

one['resid_sm'] = one['casm'] - one['seas']
one = one[['time', 'lat', 'lon', 'BT_H_IC_Fitted','BT_V_IC_Fitted', 'vod', 'casm', 'resid_sm']]
one.to_pickle('/burg/glab/users/os2328/data/VOD_project/all_smos_vod.pkl', protocol = 4)             


