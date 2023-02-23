import pandas as pd
import numpy as np
import xarray as xr
import netCDF4
import datetime
import os
import xesmf as xe

path_to_files = '/burg/glab/users/os2328/data/VOD_project/'

#vod = xr.open_dataset(path_to_files + 'VOD-IB/smap_ib_vod_xr.zarr')
#comb = vod.sortby('time')
#one = comb.resample(time='1M').mean(dim='time')

#ds_mean = one.mean(dim=('lon'))
#ds_mean.to_netcdf('/burg/glab/users/os2328/data/' + 'smap_ib_vod_xr_1M_hov'  + ".nc")
#quit()


#def regrid():






path_to_files = '/burg/glab/users/os2328/data/VOD_project/'

#smos = xr.open_dataset(path_to_files + 'smos_ic_xr.zarr')
smos = xr.open_dataset('/burg/glab/users/os2328/data/VOD_project/smos_ic_xr_vod_ver2_andcasm.zarr')

#smos = smos.sortby(smos.time)
#vod = xr.open_dataset(path_to_files + 'VOD-IB/smap_ib_xr_ver2.zarr')
#vod = vod.sortby(vod.time)

##############
# assuming no regridding is needed
##############3

#vod_25 = vod['Optical_Thickness_Nad']
#smos['vod'] = vod_25

#print(smos.info)
#smos.to_zarr(path_to_files + 'smos_ic_xr_vod_ver2.zarr')

casm = xr.open_dataset('/burg/glab/users/os2328/data/casm_xr.zarr')
casm = casm.rename({'date': 'time'})
casm = casm.sel(time=slice("2010-01-01", "2020-08-27"))

# VOD is in EASE-2 36 km, SMOS is in 25 km
# Regridding VOD to SMOS

ds_out = xr.Dataset(
    {
        "lat": (["lat"], smos.coords['lat'].values),
        "lon": (["lon"], smos.coords['lon'].values),
    }
)

regridder = xe.Regridder(casm, ds_out, "bilinear", periodic=True)
sm = regridder(casm['seasonal_cycle'])
#print(vod_25)

smos['seas'] = sm
             
print(smos.info)
smos.to_zarr(path_to_files + 'smos_ic_xr_vod_ver2_andcasm_sc.zarr')
    
quit()

try:
    smos['vod'] = vod_25
    vod_25.to_netcdf(path_to_files + "VOD-IB/smap_ib_vod_xr_25.nc")
    print(smos.info)
    smos.to_zarr(path_to_files + 'smos_ic_xr_vod.zarr')

except Exception as e:
    print('zar didnt work')
    if hasattr(e, 'message'):
        print(e.message)
    else:
        print(e)
    to_ar = xt.DataArray(vod_25)
    to_ar.to_zarr(path_to_files + "VOD-IB/vod_xr_25.zarr")

