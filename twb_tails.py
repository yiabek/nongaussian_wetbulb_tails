#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" twb_tails.py

This file is part of the temp_extremes_distshape module of the MDTF code package (see mdtf/MDTF_v2.0/LICENSE.txt)
======================================================================
twb_tails.py

  Provide functions called by twb_calculations.py

  This file is modified from part of the Surface Temperature Extremes and Distribution Shape Package
  and the MDTF code package. See LICENSE.txt for the license.

Including:
  (1) concat_files
  (2) gauss_calc
  (4) array_reshape
  (5) wetbulb_anom
  (6) shift_ratio
  (7) warming_ratio
  
Modified by: Yianna Bekris 
Original code by: Dr. Arielle Catalano

======================================================================

"""

## Import Python packages
import sys
import numpy as np
import xarray as xr
import glob
from netCDF4 import Dataset
from cftime import num2date, date2num
import scipy
from scipy import signal
import jdatetime

## Import WetBulb.py function modified to reduce NaNs
from WetBulb_NoNaN import WetBulb

## Set print options
np.set_printoptions(threshold=sys.maxsize)

### concat_files
## infiles is the path given as a string
## outfile_name is the resulting file, given as a string
def concat_files(infiles, outfile_name):
    
    # Read files and sort
    nc_files = sorted(glob.glob(infiles))
    nc_list = []
    
    # Loop through and take a daily mean
    for entry in nc_files:
        
        # keep attributes of original files
        xr.set_options(keep_attrs=True)
        
        # open file
        ds = xr.open_dataset(entry)
        ds = ds.resample(time='1D').mean()
        nc_list.append(ds)      
    
    nc_concat = xr.concat(nc_list, dim='time')
    nc_concat.to_netcdf(outfile_name)
    
    print(f'Finished and file saved as {outfile_name}!')
    
    
### gauss_calc
## For calculating Gaussian exceedances over each grid cell
## Used by warming_ratio to calculate Gaussian exceedances
def gauss_calc(warming, shape):
    
    gauss_exceedances = []
    
    for reps in np.arange(0, 1000):
        
        randsamp = np.random.randn(shape)
        randsamp_shift = randsamp + (np.std(randsamp, ddof=1) * warming)
        gauss_pthresh = np.percentile(randsamp, 95, interpolation='midpoint')
        excd_inds = randsamp_shift[randsamp_shift > gauss_pthresh]
        vals = np.true_divide(len(excd_inds), len(randsamp_shift))
        gauss_exceedances.append(vals)
        
    return gauss_exceedances    
 
### array_reshape
def array_reshape(var_data, lat, lon, datatime):

    if lon[lon>180].size>0:
        lon[lon>180]=lon[lon>180]-360

    ## Reshape data to [lon, lat, time] dimensions for code to run properly
    if len(var_data.shape) == 4:
        var_data = np.squeeze(var_data)
    if var_data.shape == (len(lon),len(lat),len(datatime)):
        var_data = var_data
    elif var_data.shape == (len(lat),len(datatime),len(lon)):
        var_data = np.transpose(var_data,(2,0,1))
    elif var_data.shape == (len(lon),len(datatime),len(lat)):
        var_data = np.transpose(var_data,(0,2,1))
    elif var_data.shape == (len(datatime),len(lon),len(lat)):
        var_data = np.transpose(var_data,(1,2,0))
    elif var_data.shape == (len(lat),len(lon),len(datatime)):
        var_data = np.transpose(var_data,(1,0,2))
    elif var_data.shape == (len(datatime),len(lat),len(lon)):
        var_data = np.transpose(var_data,(2,1,0))

    return var_data

### date_subset
# files: files to loop through, should be a list of file names
# lon_var: longitude variable (str)
# lat_var: latitude variable (str)
# field_var: name of the data variable (str)
# time_var: name of the time variable (str)
# monthsub: months to subset by (list of ints)
# yearbeg: beginning year (int)
# yearend: end year (int)
def date_subset(
        files, lon_var, lat_var, field_var, 
        time_var, monthsub, yearbeg, yearend
        ):

    print((f"   Subsetting by months {monthsub} and {yearbeg}-{yearend}..."), end=' ')
    
    ## Opening one file to extract latitudes and longitudes
    var_netcdf = Dataset(files[0],"r")
    datearray = var_netcdf.variables[time_var][:]
    timeunits = var_netcdf.variables[time_var].units
    varunits = var_netcdf.variables[field_var].units
    lat = np.asarray(var_netcdf[lat_var][:],dtype="float")
    lon = np.asarray(var_netcdf[lon_var][:],dtype="float")
    var_netcdf.close()

    ## Initiating arrays for multi-file datasets
    var_data = np.empty([0,len(lat), len(lon)])
    datatime = np.empty(0, dtype='datetime64')

    ## Loop through one or more files and extract variables
    for k in range(len(files)):
        file = files[k]
        data = Dataset(file)
        var = np.array(data[field_var])
        datearray = data.variables[time_var][:]
        timeunits = data.variables[time_var].units
        datetemp = np.array([num2date(t,units=timeunits) for t in datearray]) #daily data
        datatime = np.concatenate([datatime,datetemp])
        var_data = np.concatenate([var_data,var], axis=0)
        data.close()

    ## Extract month and years for subsetting 
    mo = np.array([int('{0.month:02d}'.format(t)) for t in list(datatime)])
    yr = np.array([int('{0.year:04d}'.format(t)) for t in list(datatime)])
    leapstr = np.array([t.strftime('%m-%d') for t in list(datatime)])
    date = np.array([t.strftime('%Y-%m-%d')for t in list(datatime)])

    ## Reshape data for easier handling
    var_data = array_reshape(var_data,lat,lon,datatime)
    
    ## remove duplicates
    filler, drop_duplicate_dates = np.unique(datatime,return_index=True)
    datatime = datatime[drop_duplicate_dates]
    var_data = var_data[:,:,drop_duplicate_dates]

    ## Find year indices
    yearind = np.where(np.logical_and(yr>=yearbeg, yr<=yearend))[0]

    ## Subset to time range specified by "yearbeg,yearend" values
    var_data = var_data[:,:,yearind]
    leapstr = leapstr[yearind]
    mo = mo[yearind]
    yr = yr[yearind]
    date = date[yearind]

    ## Subset temperature to season specified by "monthsub"
    moinds = np.in1d(mo,monthsub)
    moinds = (np.where(moinds)[0])
    moinds = [np.int(indval) for indval in moinds]
    leapstr = leapstr[moinds]
    var_data = var_data[:,:,moinds]

    ## Subset yr to get indices later
    yr = yr[moinds]

    ## Remove leap days if needed
    dateind = (leapstr != '02-29')
    leapstr = leapstr[dateind]
    var_data = var_data[:,:,dateind]

    if varunits == 'K' or varunits == 'Kelvin':
       var_data = var_data - 273.15

    print("...Array subsetted to time period specified!")

    return var_data, leapstr, lat, lon


##================================================================

""" wetbulb_anom
 Calculates wet-bulb temperature anomalies with:
     
 Specific humidity (SH_array)
 Dry-bulb temperature (T_array)
 Surface pressure (SLP_array)
 leapstr is used to index days for calculating anomalies
 
 
"""
def wetbulb_anom(SH_array, T_array, SLP_array, leapstr):

    ## Calculate wet-bulb temperature
    TWBarray, Teqarray, epottarray = WetBulb(T_array, SLP_array, SH_array, 0)


    ### Extract the "ptile" percentile of the temperature anomaly distribution
    pthresh = np.percentile(TWBarray, 95, axis=2, interpolation='midpoint')
    days_uniq = np.unique(leapstr)
    var_anom = np.empty(TWBarray.shape)
    dayinds = [np.where(leapstr == dd)[0] for dd in days_uniq]
    for begval in np.arange(0, len(dayinds)):
        temp_clim = np.mean(TWBarray[:, :, dayinds[begval]], axis=2)
        temp_clim = temp_clim.reshape(temp_clim.shape[0], temp_clim.shape[1], 1)
        var_anom[:, :, dayinds[begval]] = TWBarray[:, :, dayinds[begval]] - temp_clim
        
    return var_anom, pthresh, TWBarray, dayinds


# ======================================================================
### shift_ratio
### Compute shift ratio of Non-Gaussian to Gaussian distribution tails specified using "ptile" percentile
# -----  ptile is percentile to define tail of distribution of interest
# -----  shift is the value used to shift the distribution as a warming scenario
# -----  msk is output from Region_Mask function, masking to land grid cells
# -----  T2Manom_data is 2-meter temperature anomaly data output from Seasonal_Anomalies function above
# -----  lon and lat are longitude and latitude arrays output from Seasonal_Anomalies function above
# ---------  Output is global shift ratios
def shift_ratio(ptile, shift, anom, lon, lat):
    
    print(("   Computing Non-Gaussian to Gaussian shift ratio..."), end=' ')

    ## Remove all NaNs or Infs if necessary
    ## It is better to deal with NaNs and Infs before 
    ## computing the shift raio so this is commented out
    # np.nan_to_num(anom, nan=0, neginf=0, posinf=0)
    # where_are_NaNs = np.isnan(anom)
    # anom[where_are_NaNs] = 0
    # where_inf = np.isinf(anom)
    # anom[where_inf] = 0

    ## Detrend temperature anomaly data output from wetbulb_anom function
    anom = signal.detrend(anom, axis=2, type='linear')

    ## Extract the "ptile" percentile of the temperature anomaly distribution
    pthresh = np.percentile(anom, ptile, axis=2, interpolation='midpoint')  

    ### Compute number of days exceeding pthresh after shift
    ## -----  Loop through each grid cell where 'thrshold[iloncell,ilatcell]' 
    ## -----  is the percentile threshold 'pthresh' of the 2-m temperature anomaly distribution
    ## -----  at grid cell defined by its longitude-latitude coordinate
    
    ## Cold-side tails
    if ptile < 50:
        exceedances = np.array([[len(np.where((anom[iloncell, ilatcell, :] 
            + shift * np.std(anom[iloncell, ilatcell, :], ddof=1)) < pthresh[iloncell, ilatcell])[0]) 
            if ~np.isnan(pthresh[iloncell, ilatcell]) 
            else np.nan for ilatcell in np.arange(0, len(lat))] 
                        for iloncell in np.arange(0, len(lon))])
        
        
    ## Warm-side tails            
    elif ptile > 50:
        exceedances = np.array([[len(np.where((anom[iloncell, ilatcell, :] 
            + shift * np.std(anom[iloncell, ilatcell, :], ddof=1)) > pthresh[iloncell, ilatcell])[0]) 
            if ~np.isnan(pthresh[iloncell, ilatcell]) 
            else np.nan for ilatcell in np.arange(0, len(lat))] 
                        for iloncell in np.arange(0, len(lon))])


    ## Convert exceedances into percentages by dividing by total number of days and multiplying by 100
    exceedances = np.divide(exceedances, anom.shape[2]) * 100

    ## Set zeros to NaNs
    exceedances = exceedances.astype(float)
    exceedances[exceedances == 0] = np.nan
    
    ## Draw random samples from Gaussian distribution the length of the time dimension, and repeat 10000 times
    ## -----  Compute 5th & 95th percentiles of random gaussian distribution shift 
    ## ----- to determine statistical significance of shift ratio
    
    ## Initiate empty list for appending Gaussian exceedances
    gauss_exceedances = []
    
    ## Repeat the shift 10,000 times
    for reps in np.arange(0, 10000):
        
        randsamp = np.random.randn(anom.shape[2])
        randsamp_shift = randsamp + (np.std(randsamp, ddof=1) * shift)
        gauss_pthresh = np.percentile(randsamp, ptile, interpolation='midpoint')
        
        if ptile < 50:
            excd_inds = randsamp_shift[randsamp_shift < gauss_pthresh]
        elif ptile > 50:
            excd_inds = randsamp_shift[randsamp_shift > gauss_pthresh]
            
        # Append Gaussian exceedances    
        gauss_exceedances.append(np.true_divide(len(excd_inds), len(randsamp)))
        
    ## Find thresholdw and take median of Gaussian exceedances 
    gaussp5 = np.percentile(gauss_exceedances, 5, interpolation='midpoint') * 100
    gaussp95 = np.percentile(gauss_exceedances, 95, interpolation='midpoint') * 100
    gaussp50 = np.percentile(gauss_exceedances, 50, interpolation='midpoint') * 100

    ### Find where exceedance percentiles are outside the 5th and 95th percentile of the random gaussian distribution
    # -----  Where values are not outside the 5th/95th percentiles, set to NaN
    # -----  Remaining grid cells are statistically significantly different from a Gaussian shift
    # -----  COMMENT OUT FOR MULTI-MODEL MEAN AND COMPARISONS TO NOT MASK SIGNIFICANCE
    # exceedances[(exceedances > gaussp5) & (exceedances < gaussp95)] = np.nan

    ### Compute ratio of exceedances from non-Gaussian shift to median (50th percentile) of shifts from randomly generated Gaussian distributions
    shiftratio = np.true_divide(exceedances, np.ones_like(exceedances) * gaussp50).transpose(1, 0)
    
    print("...Computed!")
    
    return shiftratio

# ======================================================================
### warming_ratio
### Compute shift ratio of Non-Gaussian to Gaussian distribution tails specified using "ptile" percentile
# -----  ptile is percentile to define tail of distribution of interest
# -----  shift is the value used to shift the distribution as a warming scenario
# -----  msk is output from Region_Mask function, masking to land grid cells
# -----  T2Manom_data is 2-meter temperature anomaly data output from Seasonal_Anomalies function above
# -----  lon and lat are longitude and latitude arrays output from Seasonal_Anomalies function above
# ---------  Output is global shift ratios
def warming_retio(ptile, future_anom, lon, lat, warming, past_anom): # took msk out
    print(("   Computing Non-Gaussian to Gaussian warming ratio..."), end=' ')

    ## Remove NaN and Inf if necessary
    # where_are_NaNs = np.isnan(future_anom)
    # future_anom[where_are_NaNs] = 0
    # where_inf = np.isinf(future_anom)
    # future_anom[where_inf] = 0

    # where_are_pNaNs = np.isnan(past_anom)
    # past_anom[where_are_pNaNs] = 0
    # where_pinf = np.isinf(past_anom)
    # past_anom[where_pinf] = 0

    ### Add axis 2 to warming for "warm ratio" calculation
    warming = (np.repeat(warming[:, :, np.newaxis], past_anom.shape[2], axis=2))

    ### Extract the "ptile" percentile of the temperature anomaly distribution
    pthresh = np.percentile(past_anom, ptile, axis=2, interpolation='midpoint') 


    ### Compute number of days exceeding pthresh after shift
    # -----  Loop through each grid cell where 'thrshold[iloncell,ilatcell]' is the percentile threshold 'pthresh'
    # -----  of the 2-m temperature anomaly distribution at grid cell defined by its longitude-latitude coordinate
    if ptile < 50:
        exceedances = np.array([[len(np.where((future_anom[iloncell, ilatcell, :]) 
                                  < pthresh[iloncell, ilatcell])[0]) 
                                 if ~np.isnan(pthresh[iloncell, ilatcell]) 
                                 else np.nan for ilatcell in np.arange(0, len(lat))] 
                                 for iloncell in np.arange(0, len(lon))])
        
    elif ptile > 50:
        exceedances = np.array([[len(np.where((future_anom[iloncell, ilatcell, :]) 
                                  > pthresh[iloncell, ilatcell])[0]) 
                                 if ~np.isnan(pthresh[iloncell, ilatcell]) 
                                 else np.nan for ilatcell in np.arange(0, len(lat))] 
                                 for iloncell in np.arange(0, len(lon))])

    ### Convert exceedances into percentages by dividing by total number of days and multiplying by 100
    exceedances = np.divide(exceedances, future_anom.shape[2]) * 100

    ### Set zeros to NaNs
    ### COMMENT OUT FOR MULTI-MODEL MEAN AND COMPARISONS
    exceedances = exceedances.astype(float)
    # exceedances[exceedances == 0] = np.nan

    ## Draw random samples from Gaussian distribution the length of the time dimension, and repeat 10000 times
    ## -----  Compute 5th & 95th percentiles of random gaussian distribution shift to determine statistical significance of shift ratio
    specshape = future_anom.shape[2]
    gauss_exceedances = np.apply_along_axis(gauss_calc, 2, warming, specshape)
    gaussp5 = np.percentile(gauss_exceedances, 5, interpolation='midpoint',axis=2) * 100
    gaussp95 = np.percentile(gauss_exceedances, 95, interpolation='midpoint',axis=2) * 100
    gaussp50 = np.percentile(gauss_exceedances, 50, interpolation='midpoint',axis=2) * 100

    # ### Find where exceedance percentiles are outside the 5th and 95th percentile of the random gaussian distribution
    # # -----  Where values are not outside the 5th/95th percentiles, set to NaN
    # # -----  Remaining grid cells are statistically significantly different from a Gaussian shift
    # # -----  COMMENT OUT FOR MULTI-MODEL MEAN AND COMPARISONS TO NOT MASK SIGNIFICANCE
    # exceedances[(exceedances > gaussp5) & (exceedances < gaussp95)] = np.nan
    # gaussp50[gaussp50 == 0] = 0.0000000001
    
    ### Compute ratio of exceedances from non-Gaussian shift to median (50th percentile) of shifts from randomly generated Gaussian distributions
    gauss50exc = np.ones_like(exceedances) * gaussp50
    warmratio = np.true_divide(exceedances, gauss50exc)
    # warmratio = np.true_divide(exceedances, gauss50exc).transpose(1, 0)
    print("...Computed!")
    
    return warmratio, exceedances

    

    
     