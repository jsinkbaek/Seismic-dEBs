#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 13:51:57 2020

@author: emil
"""

import os
import astropy.io.fits as pyfits
import numpy as np
import scipy.stats as sct
from astropy.modeling import models, fitting
from astropy import constants as const
from astropy.time import Time
from scipy.interpolate import CubicSpline
from scipy import stats#, interpolate
from scipy.signal import fftconvolve
import lmfit

# =============================================================================
# Templates
# =============================================================================

def read_phoenix(filename, wl_min=3600,wl_max=8000):
  
  fhdu = pyfits.open(filename)
  flux = fhdu[0].data
  fhdu.close()

  whdu = pyfits.open(os.path.dirname(filename)+'/'+'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
  wave = whdu[0].data
  whdu.close()

  #from Ciddor (1996) to go from vacuum to air wavelength
  sig2 = (1e4/wave)**2.0
  f = 1.0 + 0.05792105/(238.0185-sig2) + 0.00167917/(57.362-sig2)
  wave /= f
  
  keep = (wave > wl_min) & (wave < wl_max)
  wave, flux = wave[keep], flux[keep]
  flux /= np.amax(flux)

  #flux /= np.median()
  return wave, flux

def read_kurucz(filename):
  '''
  Extract template wavelength and flux.
  :params:
    filename : str, path to template
  :return:
    wave     : array, template wavelength array
    flux     : array, template flux array
  '''
  temp = pyfits.open(filename)
  th = temp[0].header
  flux = temp[0].data[0]
  temp.close()
  
  # create wavelengths
  wls = th['CRVAL1']
  wld = th['CDELT1']
  wave = np.arange(len(flux))*wld + wls
  return wave, flux 


# =============================================================================
# SONG specifics
# =============================================================================

def SONG_request(filename):
  '''
  Extract data and header from SONG file.
  
  :params:
    filename : str, name of .fits file
  
  :return:
    data     : array, observed spectrum 
    orders   : int, number of orders for spectrum
    bjd      : float, epoch for spectrum in BJD
    bvc      : float, barycentric velocity correction for epoch in km/s
    star     : str, name of star 
    date     : str, date of observations in UTC
    exp      : float, exposure time in seconds
  '''
  fits = pyfits.open(filename)
  hdr = fits[0].header
  data = fits[0].data
  fits.close()

  star = hdr['OBJECT']
  date = hdr['DATE-OBS']
  exp =  hdr['EXPTIME']

  bjd = hdr['BJD-MID'] + 2400000
  bvc = hdr['BVC']
  orders = hdr['NAXIS2']

  return data, orders, bjd, bvc, star, date, exp

def SING(data,order=30):
  '''
  Extract spectrum from SONG at given order.
  
  :params:
    data : array, observed spectrum
    order: int, extract spectrum at order, defaults to central order
  
  :return:
    wl   : array, observed wavelength 
    fl   : array, observed raw flux
    bl   : array, blaze function
  '''
  
  wl, fl = data[3,order,:], data[1,order,:] # wavelength, flux
  bl = data[2,order,:] # blaze
  return wl, fl, bl


# =============================================================================
# FIES specifics
# =============================================================================

def FIES_caliber(filename):
  '''
  Reads a wavelength calibrated spectrum in IRAF (FIEStool) format.
  Returns a record array with wavelength and flux order by order as a two column 
  array in data['order_i'].
  
  Adapted and extended from functions written by R. Cardenes and J. Jessen-Hansen.
  
  :params:
    filename : str, name of .fits file
  
  :return:
    data     : array, observed spectrum, wavelength and flux order by order
    no_orders: int, number of orders for spectrum
    star     : str, name of star 
    date     : str, date of observations in UTC
    exp      : float, exposure time in seconds
  
  '''
  try:
    hdr = pyfits.getheader(filename)
    star = hdr['OBJECT']
    date = hdr['DATE-OBS']
    date_mid = hdr['DATE-AVG']
    #bjd = Time(date_mid, format='isot', scale='utc').jd
    bjd = Time(date, format='isot', scale='utc').jd
    exp = hdr['EXPTIME']
    vhelio = hdr['VHELIO']
  except Exception as e:
    print('Problems extracting headers from {}: {}'.format(filename, e))
    print('Is filename the full path?')

  # Figure out which header cards we want
  cards = [x for x in sorted(hdr.keys()) if x.startswith('WAT2_')]
  # Extract the text from the cards
  # We're padding each card to 68 characters, because PyFITS won't
  # keep the blank characters at the right of the text, and they're
  # important in our case
  text_from_cards = ''.join([hdr[x].ljust(68) for x in cards])
  data = text_from_cards.split('"')

  #Extract the wavelength info (zeropoint and wavelength increment for each order) to arrays
  info = [x for x in data if (x.strip() != '') and ("spec" not in x)]
  zpt = np.array([x.split(' ')[3] for x in info]).astype(np.float64)
  wstep = np.array([x.split(' ')[4] for x in info]).astype(np.float64)
  npix = np.array([x.split(' ')[5] for x in info]).astype(np.float64)
  orders = np.array([x.split(' ')[0] for x in info])
  no_orders = len(orders)

  #Extract the flux
  fts  = pyfits.open(filename)
  data = fts[0].data
  fts.close()
  wave = data.copy()

  #Create record array names to store the flux in each order
  col_names = [ 'order_'+order for order in orders]

  #Create wavelength arrays
  for i,col_name in enumerate(col_names):
    wave[0,i] = zpt[i] + np.arange(npix[i]) * wstep[i]

  #Save wavelength and flux order by order as a two column array in data['order_i']
  data = np.rec.fromarrays([np.column_stack([wave[0,i],data[0,i]]) for i in range(len(col_names))],
    names = list(col_names))

  return data, no_orders, bjd, vhelio, star, date, exp

def getFIES(data,order=40):
  '''
  Extract calibrated spectrum from FIES at given order.
  
  :params:
    data : array, observed spectrum
    order: int, extract spectrum at order, defaults to central order
  
  :return:
    wl   : array, observed wavelength 
    fl   : array, observed raw flux
  '''
  arr = data['order_{:d}'.format(order)]
  wl, fl = arr[:,0], arr[:,1]
  return wl, fl

# =============================================================================
# Preparation/normalization
# =============================================================================

def normalize(wl,fl,bl=np.array([]),poly=1,gauss=True,lower=0.5,upper=1.5):
  '''
  Nomalization of observed spectrum.
  
  :params:
    wl   : array, observed wavelength 
    fl   : array, observed raw flux
    bl   : array, blaze function, recommended input
    poly : int, degree of polynomial fit to normalize, 
                set to None for no polynomial fit
    gauss: bool, only fit the polynomial to flux within
                 (mu + upper*sigma) > fl > (mu - lower*sigma)
    upper: float, upper sigma limit to include in poly fit
    lower: float, lower sigma limit to include in poly fit

  :return:
    wl   : array, observed wavelength 
    nfl  : array, observed normalized flux
  '''

  if len(bl) > 0:
    fl = fl/bl # normalize w/ blaze function
  
  keep = np.isfinite(fl)
  wl, fl = wl[keep], fl[keep] # exclude nans
  
  if poly is not None:
    if gauss:
      mu, sig = sct.norm.fit(fl)
      mid  = (fl < (mu + upper*sig)) & (fl > (mu - lower*sig))
      pars = np.polyfit(wl[mid],fl[mid],poly)
    else:
      pars = np.polyfit(wl,fl,poly)
    fit = np.poly1d(pars)
    nfl = fl/fit(wl)
  else:
    nfl = fl/np.median(fl)

  return wl, nfl

def crm(wl,nfl,iters=1,q=[99.0,99.9,99.99]):
  '''
  Cosmic ray mitigation.
  Excludes flux over qth percentile.
  
  :params:
    wl   : array, observed wavelength 
    nfl  : array, observed normalized flux
    iters: int, iterations of removing upper q[iter] percentile
    q    : list, percentiles
  
  :return:
    wl   : array, observed wavelength 
    nfl  : array, observed normalized flux
  '''
  assert iters <= len(q), 'Error: More iterations than specified percentiles.'

  for ii in range(iters):
    cut = np.percentile(nfl,q[ii]) # find upper percentile
    cosmic = np.array(n for n in range(len(nfl)) if nfl[n] < cut) # cosmic ray mitigation
    wl, nfl = wl[cosmic], nfl[cosmic]

  return wl, nfl
        
def resample(wl,nfl,twl,tfl,dv=1.0,edge=0.0):
  '''
  Resample wavelength and interpolate flux and 
  template flux.
  Flips flux, i.e., 1-flux.
  
  :params:
    wl   : array, observed wavelength 
    nfl  : array, observed normalized flux
    twl  : array, template wavelength
    tfl  : array, template flux
    dv   : float, RV steps in km/s
    edge : float, skip edge of detector - low S/N - in Angstrom
  
  :return:
    lam  : array, resampled wavelength
    rf_fl: array, resampled and flipped flux
    rf_tl: array, resampled and flipped template flux
  '''

  wl1, wl2 = min(wl) + edge, max(wl) - edge
  nn = np.log(wl2/wl1)/np.log(np.float64(1.0) + dv/(const.c.value/1e3))
  lam = wl1*(np.float64(1.0)  + dv/(const.c.value/1e3))**np.arange(nn,dtype='float64')
  if len(lam)%2 != 0: lam = lam[:-1] # uneven number of elements

  keep = (twl >= lam[0]) & (twl <= lam[-1]) # only use resampled wl interval
  twl, tfl_order = twl[keep], tfl[keep]
  
  flip_fl, flip_tfl = 1-nfl, 1-tfl_order
  rf_fl = np.interp(lam,wl,flip_fl)
  rf_tl = np.interp(lam,twl,flip_tfl)
  return lam, rf_fl, rf_tl

# =============================================================================
# Cross-correlation
# =============================================================================

def getCCF(fl,tfl,rvr=401,ccf_mode='full'):
  '''
  Perform the cross correlation and trim array
  to only include points over RV range.
  
  :params:
    fl   : array, flipped and resampled flux
    tfl  : array, flipped and resampled template flux
    rvr  : integer, range for RVs in km/s
  
  :return:
    rvs  : array, RV points
    ccf  : array, CCF at RV
  '''
  ccf = np.correlate(fl,tfl,mode=ccf_mode)
  ccf = ccf/(np.std(fl) * np.std(tfl) * len(tfl)) # normalize ccf
  rvs = np.arange(len(ccf)) - len(ccf)//2
  
  mid = (len(ccf) - 1) // 2 # midpoint
  lo = mid - rvr//2
  hi = mid + (rvr//2 + 1)
  rvs, ccf = rvs[lo:hi], ccf[lo:hi] # trim array

  ccf = ccf/np.mean(ccf) - 1 # shift 'continuum' to zero
  #cut = np.percentile(ccf,85)
  #ccf = ccf/np.median(ccf[ccf < cut]) - 1 # shift 'continuum' to zero
  return rvs, ccf

def getRV(rvs,ccf):
  '''
  Get radial velocity and projected rotational velocity
  from CCF by fitting a Gaussian and collecting the location of 
  the maximum and the width, respectively.
  
  :params:
    rvs  : array, RV points
    ccf  : array, CCF at RV
  
  :return:
    RV   : float, position of Gaussian, radial velocity in km/s
    vsini: float, width of Gaussian, rotational velocity in km/s
  '''
  
  ## starting guesses
  idx = np.argmax(ccf)
  amp, mu1 = ccf[idx], rvs[idx]# get max value of CCF and location
  ccf = ccf/amp
  g_init = models.Gaussian1D(amplitude=1.0, mean=mu1, stddev=1.)
  fit_g = fitting.LevMarLSQFitter()
  gauss = fit_g(g_init, rvs, ccf)
  rv, vsini = gauss.mean.value, gauss.stddev.value
  return rv, vsini

# =============================================================================
# Broadening function
# =============================================================================

def getBF(fl,tfl,rvr=401,dv=1.0):
  '''
  Carry out the SVD of the "design  matrix".

    This method creates the "design matrix" by applying
    a bin-wise shift to the template and uses numpy's
    `svd` algorithm to carry out the decomposition.

    The design matrix, `des` is written in the form:
    "`des` = u * w * transpose(v)". The matrices des,
    w, and u are stored in homonymous attributes.
  
  :params:  
    fl   : array, flipped and resampled flux
    tfl  : array, flipped and resampled template flux
    bn   : int, Width (number of elements) of the broadening function. Needs to be odd.
  :returns:
    vel  : array, velocity in km/s
    bf   : array, the broadening function
  '''
  bn = rvr/dv
  if bn % 2 != 1: bn += 1
  bn = int(bn) # Width (number of elements) of the broadening function. Must be odd.
  bn_arr = np.arange(-int(bn/2), int(bn/2+1), dtype=float)
  vel = -bn_arr*dv

  nn = len(tfl) - bn + 1
  des = np.matrix(np.zeros(shape=(bn, nn)))
  for ii in range(bn): des[ii,::] = tfl[ii:ii+nn]

  ## Get SVD deconvolution of the design matrix
  ## Note that the `svd` method of numpy returns
  ## v.H instead of v.
  u, w, v = np.linalg.svd(des.T, full_matrices=False)

  wlimit = 0.0
  w1 = 1.0/w
  idx = np.where(w < wlimit)[0]
  w1[idx] = 0.0
  diag_w1 = np.diag(w1)

  vT_diagw_uT = np.dot(v.T,np.dot(diag_w1,u.T))

  ## Calculate broadening function
  bf = np.dot(vT_diagw_uT,np.matrix(fl[int(bn/2):-int(bn/2)]).T)
  bf = np.ravel(bf)

  return vel, bf

def smoothBF(vel,bf,sigma=5.0):
  '''
  :params:
    vel  : array, velocity in km/s
    bf   : array, the broadening function
  '''
  nn = len(vel)
  gauss = np.zeros(nn)
  gauss[:] = np.exp(-0.5*np.power(vel/sigma,2))
  total = np.sum(gauss)

  gauss /= total

  bfgs = fftconvolve(bf,gauss,mode='same')

  return bfgs

# def rotbf_func(vel,ampl,vrad,vsini,gwidth,const,limbd):
#   '''
#   Rotational profile.

#   :params:
#     vel    : array, velocity in km/s
#   :returns:
#     rotbf  : array, rotational profile
#   '''

#   nn = len(vel)
#   bf = np.zeros(nn)
#   #bf = np.ones(nn)*const
#   bf_gs = np.zeros(nn)

#   # a1 = (vel - vrad)/vsini
#   # idxs = np.where(abs(a1) < 1.0)[0]
#   # asq = np.sqrt(1.0 - np.power(a1[idxs],2))
#   # bf[idxs] += ampl*asq*(1.0 - limbd + 0.25*np.pi*limbd*asq)
  
#   # gs[:] = np.exp(-0.5*np.power(vel/gwidth,2))
#   # total = np.sum(gs)
#   # gs = gs/total
#   ln = 0
#   for ii in range(nn):
#     a1 = (vel[ii] - vrad)/vsini
#     a_sq = 0
#     bf[ii] = const
#     if abs(a1) < 1.0:
#       a_sq = np.sqrt(1.0 - np.power(a1,2))
#       bf[ii] += ampl*a_sq*(1.0 - limbd + 0.25*np.pi*limbd*a_sq)
#       ln += 1 
#       stp = ii

  
#   st = stp-ln+1
#   total = 0
#   for ii in range(nn):
#     bf_gs[ii] = 0
#     for jj in range(ln):
#       gs = np.exp(-0.5*np.power((st + jj - ii)/gwidth,2))/(2.50662827463100*gwidth)
#       bf_gs[ii] += bf[st + jj]*gs
#       total += gs
#     bf_gs[ii] += (1.0 - total)*const
#     total = 0

#   #rotbf = fftconvolve(bf,gs,mode='same')

#   return bf_gs

# test for fitting two rotational profiles by kfb
def rotbf2_func(vel,ampl1,vrad1,vsini1,ampl2,vrad2,vsini2,gwidth,const,limbd):
  '''
  Rotational profile. Kaluzny 2006

  :params:
    vel    : array, velocity in km/s
  :returns:
    rotbf  : array, rotational profile
  '''

  nn = len(vel)
  #bf = np.zeros(nn)
  bf = np.ones(nn)*const

  a1 = (vel - vrad1)/vsini1
  idxs = np.where(abs(a1) < 1.0)[0]
  asq = np.sqrt(1.0 - np.power(a1[idxs],2))
  bf[idxs] += ampl1*asq*(1.0 - limbd + 0.25*np.pi*limbd*asq)
  
  a2 = (vel - vrad2)/vsini2
  idxs = np.where(abs(a2) < 1.0)[0]
  asq = np.sqrt(1.0 - np.power(a2[idxs],2))
  bf[idxs] += ampl2*asq*(1.0 - limbd + 0.25*np.pi*limbd*asq)

  gs = np.zeros(nn)
  cgwidth = np.sqrt(2*np.pi)*gwidth
  gs[:] = np.exp(-0.5*np.power(vel/gwidth,2))/cgwidth
  
  rotbf2 = fftconvolve(bf,gs,mode='same')

  return rotbf2

def rotbf2_res(params,vel,bf,wf):
  ampl1  = params['ampl1'].value
  vrad1  = params['vrad1'].value
  vsini1 = params['vsini1'].value
  ampl2  = params['ampl2'].value
  vrad2  = params['vrad2'].value
  vsini2 = params['vsini2'].value
  gwidth = params['gwidth'].value
  const =  params['const'].value
  limbd = params['limbd1'].value

  res = bf - rotbf2_func(vel,ampl1,vrad1,vsini1,ampl2,vrad2,vsini2,gwidth,const,limbd)
  return res*wf

def rotbf2_fit(vel,bf,fitsize,res=60000,smooth=5.0,vsini1=5.0,vsini2=5.0,vrad1=0.0,vrad2=0.0,ampl1=1.0,ampl2=0.5,print_report=True):
  
  bfgs = smoothBF(vel,bf,sigma=smooth/1.)
  c = np.float64(299792.458)
  gwidth = np.sqrt(((c/res)/(2.354*1.))**2 + (smooth/1.)**2) # 1. is current dv value!
 
  peak = np.argmax(bfgs)
  #idx = np.where((vel > vel[peak] - fitsize) & (vel < vel[peak] + fitsize+1))[0]

  #wf = np.zeros(len(bfgs))
  #wf[idx] = 1.0
  wf = np.ones(len(bfgs))	

  params = lmfit.Parameters()
  #params.add('ampl1', value = bfgs[peak])
  params.add('ampl1', value = ampl1)
  params.add('vrad1', value = vrad1)
  params.add('ampl2', value = ampl2)
  params.add('vrad2', value = vrad2)
  params.add('gwidth', value = gwidth,vary = False)
  params.add('const', value = 0.0)
  params.add('vsini1', value = vsini1)
  params.add('vsini2', value = vsini2)
  params.add('limbd1', value = 0.68,vary = False)  

  fit = lmfit.minimize(rotbf2_res, params, args=(vel,bfgs,wf),xtol=1.e-8,ftol=1.e-8,max_nfev=500)
  if print_report: print(lmfit.fit_report(fit, show_correl=False))
  

  ampl1, ampl2, gwidth = fit.params['ampl1'].value,fit.params['ampl2'].value, fit.params['gwidth'].value
  vrad1, vsini1 = fit.params['vrad1'].value, fit.params['vsini1'].value
  vrad2, vsini2 = fit.params['vrad2'].value, fit.params['vsini2'].value
  limbd, const = fit.params['limbd1'].value, fit.params['const'].value
  model = rotbf2_func(vel,ampl1,vrad1,vsini1,ampl2,vrad2,vsini2,gwidth,const,limbd)

  return fit, model, bfgs



# END test for fitting two rotational profiles by kfb

def rotbf_func(vel,ampl,vrad,vsini,gwidth,const,limbd):
  '''
  Rotational profile. Kaluzny 2006

  :params:
    vel    : array, velocity in km/s
  :returns:
    rotbf  : array, rotational profile
  '''

  nn = len(vel)
  #bf = np.zeros(nn)
  bf = np.ones(nn)*const

  a1 = (vel - vrad)/vsini
  idxs = np.where(abs(a1) < 1.0)[0]
  asq = np.sqrt(1.0 - np.power(a1[idxs],2))
  bf[idxs] += ampl*asq*(1.0 - limbd + 0.25*np.pi*limbd*asq)
  
  gs = np.zeros(nn)
  cgwidth = np.sqrt(2*np.pi)*gwidth
  gs[:] = np.exp(-0.5*np.power(vel/gwidth,2))/cgwidth
  
  #gs[:] = np.exp(-0.5*np.power(vel/gwidth,2))
  #total = np.sum(gs)
  #gs = gs/total

  # for ii in range(nn):
  #   a1 = (vel[ii] - vrad)/vsini
  #   a_sq = 0
  #   bf[ii] = const
  #   if abs(a1) < 1.0:
  #     a_sq = np.sqrt(1.0 - np.power(a1,2))
  #     bf[ii] += ampl*a_sq*(1.0 - limbd + 0.25*np.pi*limbd*a_sq)
  #   gs[ii] = np.exp(-0.5*np.power())

  rotbf = fftconvolve(bf,gs,mode='same')

  return rotbf

def rotbf_res(params,vel,bf,wf):
  ampl  = params['ampl1'].value
  vrad  = params['vrad1'].value
  vsini = params['vsini1'].value
  gwidth = params['gwidth'].value
  const =  params['const'].value
  limbd = params['limbd1'].value

  res = bf - rotbf_func(vel,ampl,vrad,vsini,gwidth,const,limbd)
  return res*wf

def rotbf_fit(vel,bf,fitsize,res=60000,smooth=5.0,vsini=5.0,print_report=True):
  
  bfgs = smoothBF(vel,bf,sigma=smooth/1.)
  c = np.float64(299792.458)
  gwidth = np.sqrt(((c/res)/(2.354*1.))**2 + (smooth/1.)**2) # 3. is current dv value!
 
  peak = np.argmax(bfgs)
  idx = np.where((vel > vel[peak] - fitsize) & (vel < vel[peak] + fitsize+1))[0]

  wf = np.zeros(len(bfgs))
  wf[idx] = 1.0

  params = lmfit.Parameters()
  params.add('ampl1', value = bfgs[peak])
  params.add('vrad1', value = vel[peak])
  params.add('gwidth', value = gwidth,vary = False)
  params.add('const', value = 0.0)
  params.add('vsini1', value = vsini)
  params.add('limbd1', value = 0.68,vary = False)  

  fit = lmfit.minimize(rotbf_res, params, args=(vel,bfgs,wf),xtol=1.e-8,ftol=1.e-8,max_nfev=500)
  if print_report: print(lmfit.fit_report(fit, show_correl=False))
  

  ampl, gwidth = fit.params['ampl1'].value, fit.params['gwidth'].value
  vrad, vsini = fit.params['vrad1'].value, fit.params['vsini1'].value
  limbd, const = fit.params['limbd1'].value, fit.params['const'].value
  model = rotbf_func(vel,ampl,vrad,vsini,gwidth,const,limbd)

  return fit, model, bfgs

# =============================================================================
# Collection of calls
# =============================================================================
def auto_RVs_BF(wl,fl,bl,twl,tfl,q=[99.99],
                dv=1.0,edge=0.0,rvr=201,
                fitsize=30,res=90000,smooth=5.0,
                bvc=0.0):
  '''
  Collection of function calls to get radial velocity and vsini 
  from the broadening function following the recipe by J. Jessen-Hansen.
  
  :params:
    wl      : array, observed wavelength 
    nfl     : array, observed normalized flux
    twl     : array, template wavelength
    tfl     : array, template flux
    dv      : float, RV steps in km/s
    edge    : float, skip edge of detector - low S/N - in Angstrom
    rvr     : integer, range for RVs in km/s
    fitsize : float, fitsize in km/s - only fit `fitsize` part of the rotational profile
    res     : float, resolution of spectrograph - affects width of peak
    smooth  : float, smoothing factor - sigma in Gaussian smoothing
    bvc     : float, barycentric velocity correction for epoch in km/s
    
  :return:
    RV      : float, position of Gaussian, radial velocity in km/s
    vsini   : float, width of Gaussian, rotational velocity in km/s
  '''  
  wl, nfl = normalize(wl,fl,bl=bl)
  wl, nfl = crm(wl,nfl,q=q)
  lam, rf_fl, rf_tl = resample(wl,nfl,twl,tfl)
  rvs, bf = getBF(rf_fl,rf_tl,rvr=201)
  #rot = smoothBF(rvs,bf,sigma=smooth)

  fit, _, _ = rotbf_fit(rvs,rot,fitsize,res=res,smooth=smooth)
  RV, vsini = fit.params['vrad1'].value, fit.params['vsini1'].value
  return RV, vsini

def auto_RVs(wl,fl,bl,twl,tfl,dv=1.0,edge=0.0,rvr=401,bvc=0.0):
  '''
  Collection of function calls to get radial velocity and vsini
  from the cross-correlation function.
  
  :params:
    wl   : array, observed wavelength 
    nfl  : array, observed normalized flux
    twl  : array, template wavelength
    tfl  : array, template flux
    dv   : float, RV steps in km/s
    edge : float, skip edge of detector - low S/N - in Angstrom
    rvr  : integer, range for RVs in km/s
    bvc  : float, barycentric velocity correction for epoch in km/s
    
  :return:
    RV   : float, position of Gaussian, radial velocity in km/s
    vsini: float, width of Gaussian, rotational velocity in km/s
  '''
  wl, nfl = normalize(wl,fl,bl)
  wl, nfl = crm(wl,nfl)
  lam, resamp_flip_fl, resamp_flip_tfl = resample(wl,nfl,twl,tfl,dv=dv,edge=edge)
  rvs, ccf = getCCF(resamp_flip_fl,resamp_flip_tfl,rvr=rvr)
  rvs = rvs + bvc
  RV, vsini = getRV(rvs,ccf)
  
  return RV, vsini

def get_val_err(vals,out=True,sigma=5):
  '''
  Value and error estimation with simple outlier rejection.
  
  :params:
    vals : array/list of values
    out  : bool, if True an outlier rejection will be performed - Gaussian
    sigma: float, level of the standard deviation
    
  :return:
    val  : float, median of the values
    err  : float, error of the value
  '''
  if out:
    mu, sig = sct.norm.fit(vals)
    sig *= sigma
    keep = (vals < (mu + sig)) & (vals > (mu - sig))
    vals = vals[keep]
  
  val, err = np.median(vals), np.std(vals)/np.sqrt(len(vals))
  return val, err

# =============================================================================
# Command line calls
# =============================================================================
  
if __name__=='__main__':
  import sys
  import argparse 
  import glob
  import os
  
  def str2bool(arg):
    if isinstance(arg, bool):
      return arg
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
      return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
      return False
    else:
      raise argparse.ArgumentTypeError('Boolean value expected.')
  
  def low(arg):
    return arg.lower()
  ### Command line arguments to parse to script    
  parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
  
  ### Input
  putin = parser.add_argument_group('Input')
  putin.add_argument('path', type=str, help='Path to .fits files.')
  putin.add_argument('-tpath', '--template_path', type=str, help='Path to template.',
                     default='/home/emil/Desktop/PhD/scripts/templates/6250_35_p00p00.ms.fits')
  putin.add_argument('-t','--template',type=low, default='kurucz',
                     choices=['kurucz','phoenix'], help='Type of template.')
  
  ### Output
  output = parser.add_argument_group('Output')
  output.add_argument('-w', '--write', type=str2bool, default=True,
                      nargs='?', help='Should RVs be written to file?')
  output.add_argument('-s','--store', default=os.getcwd() + '/',
                      type=str, help='Path to store the output - RV_star.txt.')

  ### Custom
  custom = parser.add_argument_group('Custom')
  custom.add_argument('-dv','--delta_velocity',type=float,default=1.0,
                      help='Velocity step for radial velocity arrray in km/s.')
  custom.add_argument('-rvr','--rv_range',type=int,default=401,
                      help='Radial velocity range for RV fit in km/s.')
  custom.add_argument('-edge','--remove_edge',type=float,default=0.0,
                      help='Omit spectrum at edge of detector in Angstrom.')
  custom.add_argument('-so','--start_order',type=int,default=0,
                      help='Order to start from - skip orders before.')
  custom.add_argument('-eo','--end_order',type=int,default=0,
                      help='Order to end at - skip orders after.')
  custom.add_argument('-xo','--exclude_orders',type=str,default='none',
                      help='Orders to exclude - e.g., "0,12,15,36".')
  custom.add_argument('-out','--outlier_rejection', type=str2bool, default=True,
                      nargs='?', help='Outlier rejection?')
  custom.add_argument('-sig','--sigma_outlier',type=float,default=5,
                      help='Level of outliger rejection.')
  
  args, unknown = parser.parse_known_args(sys.argv[1:])

  tfile = args.template_path
  template = args.template
  if template == 'kurucz':
    twl, tfl = read_kurucz(tfile)
  elif  template == 'phoenix':
    twl, tfl = read_phoenix(tfile)
  print(args.path + '*.fits')
  
  filepath = args.path
  if filepath[-1] != '/': filepath = filepath + '/'
  fits_files = glob.glob(filepath + '*.fits')

  edge = args.remove_edge
  out, sigma = args.outlier_rejection, args.sigma_outlier
  write = args.write

  if write:
    first = pyfits.open(fits_files[0])
    star = first[0].header['OBJECT']
    first.close()
    fname = args.store + '{}_rvs.txt'.format(star.replace(' ',''))
    rv_file = open(fname,'w')
    lines = ['# bjd rv (km/s) srv (km/s)\n']

  vsinis = np.array([])
  
  for file in fits_files:
    data, no_orders, bjd, bvc, star, date, exp = SONG_request(file)
    print(star,date)
    print('Exposure time: {} s'.format(exp))
    rv_range = args.rv_range
    vel_step = args.delta_velocity
    rv, cc = np.empty([no_orders,rv_range]), np.empty([no_orders,rv_range]) 
    RVs = np.array([])
    
    orders = np.arange(no_orders)
    so, eo = args.start_order, args.end_order
    if (eo > 0) & (no_orders > (eo+1)):
      assert eo > so, "Error: Starting order is larger than or equal to ending order."
      orders = orders[so:eo+1]
    else:
      orders = orders[so:]
    
    ex_orders = args.exclude_orders
    if ex_orders != 'none':
      ex_orders = [int(ex) for ex in ex_orders.split(',')]
      orders = [order for order in orders if order not in ex_orders]

    for order in orders:
      wl, fl, bl = SING(data,order)
      RV, vsini = auto_RVs(wl,fl,bl,twl,tfl,dv=vel_step,edge=edge,rvr=rv_range,bvc=bvc)
      RVs = np.append(RVs,RV)
      vsinis = np.append(vsinis,vsini)

    RV, sigRV = get_val_err(RVs,out=out,sigma=sigma)
    
    print('RV = {:.6f} +/- {:0.6f} km/s\n'.format(RV,sigRV))
    line = '{} {} {}\n'.format(bjd,RV,sigRV)
    if write: lines.append(line)

  vsini, sigvsini = get_val_err(vsinis,out=out,sigma=sigma)
  print('vsini = {:.3f} +/- {:0.3f} km/s'.format(vsini,sigvsini))
  
  if write: 
    rv_file.write('# {} \n# vsini = {} +/- {} km/s\n'.format(star,vsini,sigvsini))
    for line in lines: rv_file.write(line)
    rv_file.close()

