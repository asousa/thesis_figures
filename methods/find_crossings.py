import matplotlib
matplotlib.use('agg')
import numpy as np
import pandas as pd
import pickle
import gzip
from scipy import interpolate
import matplotlib.pyplot as plt
import os
import itertools
import random
import os
import time
import datetime as datetime
from raytracer_utils import read_rayfile, read_damp
from scipy.spatial import Delaunay
from scipy.integrate import nquad
from scipy import stats
import xflib
from graf_iono_absorp import total_input_power, lon2MLT, MLT2lon, input_power_scaling
import logging
import math






# ----------------------------- Methods -------------------------------
def calc_stix_parameters(rf, time_axis):
    EPS0 = 8.854e-12
    # ray keys:
    #['qs', 'ms', 'nus', 'pos', 'n', 'stopcond', 'vprel', 'B0', 'w', 'time', 'Nspec', 'Ns', 'vgrel']

    # Magnetic field, index of refraction, number densities
    Bmag =  interpolate.interp1d(rf['time'],np.linalg.norm(rf['B0'],axis=1)).__call__(time_axis)
    n_vec = interpolate.interp1d(rf['time'],rf['n'], axis=0).__call__(time_axis)
    Ns    = interpolate.interp1d(rf['time'],rf['Ns'], axis=0).__call__(time_axis)
    
    # Species charge and mass
    qs = np.array(rf['qs'])[0,:]
    ms = np.array(rf['ms'])[0,:]
    
    w = rf['w']

    wps2vec = Ns.T*((pow(qs,2.0)/ms)[:,np.newaxis])/EPS0
    
    whsvec = Bmag*((qs/ms)[:,np.newaxis])
    
    # print whsvec

    R = 1.0 - np.sum(wps2vec/(w*(w + whsvec)), axis=0);
    L = 1.0 - np.sum(wps2vec/(w*(w - whsvec)), axis=0);
    P = 1.0 - np.sum(wps2vec/(w*w), axis=0);

    return R, L, P


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

def rotate_latlon(raypos, itime, dlat, dlon, xf=None):
    if xf is None:
        xf = xflib.xflib(lib_path='/shared/users/asousa/WIPP/3dWIPP/python/libxformd.so')
        
    newpos = np.zeros_like(raypos)
    for ind in range(np.shape(raypos)[1]):
#         print ind
        tmp = xf.sm2rllmag(raypos[:,ind], itime)
        tmp[1] += dlat
        tmp[2] += dlon
        newpos[:,ind] = xf.rllmag2sm(tmp, itime)
    
    return newpos

def flatten_longitude_variation(raypos, itime, xf=None):
    if xf is None:
        xf = xflib.xflib(lib_path='/shared/users/asousa/WIPP/3dWIPP/python/libxformd.so')
        
    newpos = np.zeros_like(raypos)

    tmp = xf.sm2rllmag(raypos[:,0], itime)
    start_lon = tmp[2]

    for ind in range(np.shape(raypos)[1]):
#         print ind
        tmp = xf.sm2rllmag(raypos[:,ind], itime)
        # tmp[1] += dlat
        tmp[2] = start_lon
        newpos[:,ind] = xf.rllmag2sm(tmp, itime)
    
    return newpos

def voxel_vol_nd(points):
    '''
    volume of a polygon in n-dimensional space. Rad.
    '''
    n, m = np.shape(points)
    tri = Delaunay(points.T, qhull_options='QJ')
    v = 0
    for row in tri.simplices:
        mat = points[:, row[1:]].T - points[:, row[0]].T
        v += np.abs(np.linalg.det(mat)/math.factorial(n))
    return v





def gen_EA_array(Lshells, dlat_fieldline, center_lon, itime,  L_MARGIN = 0.1, xf = None):

    if xf is None:
        xf = xflib.xflib(lib_path='/shared/users/asousa/WIPP/WIPP_stencils/python/methods/libxformd.so')

    # Constants
    Hz2Rad = 2.*np.pi
    D2R = np.pi/180.
    R2D = 180./np.pi
    H_IONO_BOTTOM = 1e5
    H_IONO_TOP = 1e6
    R_E = 6371e3
    C = 2.997956376932163e8
    Q_EL = 1.602e-19
    M_EL = 9.1e-31
    E_EL = 5.105396765648739E5 
    MU0  = np.pi*4e-7
    EPS0 = 8.854E-12
    B0   = 3.12e-5

    # ------------------ Set up field lines ----------------------------
    fieldlines = []

    for L in Lshells:
        fieldline = dict()
        maxlat = np.floor(np.arccos(np.sqrt((R_E + H_IONO_TOP)/R_E/L))*R2D)
        n_lsteps = int(np.round(2.0*maxlat/dlat_fieldline))
        lat_divisions = np.linspace(maxlat, -1.0*maxlat, n_lsteps+1)
        lat_centers   = lat_divisions[0:-1] - dlat_fieldline/2.
    
        fieldline['lat'] = lat_centers
        fieldline['L'] = L
        # Radius of tube around field line:
        slat = np.sin(lat_centers*D2R)
        clat = np.cos(lat_centers*D2R)
        clat2 = pow(clat,2.)
        slat2 = pow(slat,2.)
        slat_term = np.sqrt(1. + 3.*slat2)

        radii = clat2*clat / slat_term * L_MARGIN
        R_centers = L*clat2
        
        fieldline['R'] = R_centers
        fieldline['xradius']= radii
        
        # Approximate each segment as a cylinder:
        seg_length = R_centers*dlat_fieldline*D2R
        seg_vol = np.pi*pow(radii,2.)*seg_length*pow(R_E,3.)  # cubic meters

        fieldline['vol'] = seg_vol
        fieldline['total_vol'] = np.sum(seg_vol)

        fieldline['x'] = R_centers*clat
        fieldline['y'] = R_centers*slat
        
        fieldline['x_unit_vect'] = (3*clat2 - 2) / slat_term ;
        fieldline['y_unit_vect'] = (3*slat*clat) / slat_term ;

        coords_rllmag = np.vstack([R_centers, lat_centers, np.ones(len(lat_centers))*center_lon])
        coords_sm = []
        for row in coords_rllmag.T:
            coords_sm.append(xf.rllmag2sm(row, itime))
        fieldline['pos'] = np.array(coords_sm)


        # Calculate loss cone angles:
        # (dipole field for now)
        fieldline['wh'] = (Q_EL*B0/M_EL)/pow(L, 3.)*slat_term/pow(clat,6.)
        fieldline['dwh_ds'] = 3.*fieldline['wh']/(L*R_E)*slat/slat_term*(1./(slat_term*slat_term) + 2./(clat*clat))

        # Equatorial loss cone
        epsm = (1./L)*(1. + H_IONO_BOTTOM/R_E)
        fieldline['alpha_eq'] = np.arcsin(np.sqrt(pow(epsm, 3.)/np.sqrt(1 + 3.*(1. - epsm))  ))

        # Local loss cone
        fieldline['alpha_lc'] = np.arcsin(np.sqrt( slat_term/pow(clat,6.) )*np.sin(fieldline['alpha_eq']));
        salph = np.sin(fieldline['alpha_lc'])
        calph = np.cos(fieldline['alpha_lc'])

        fieldline['ds'] = L*R_E*slat_term*clat*dlat_fieldline*np.pi/180.0
        fieldline['dv_para_ds'] = -0.5*(salph*salph/(calph*fieldline['wh']))*fieldline['dwh_ds']



        # Calculate flight time constants to ionosphere: (dipole field again)
        iono_lat = np.floor(np.arccos(np.sqrt((R_E + H_IONO_BOTTOM)/R_E/L))*R2D)
        lats_fine = np.arange(iono_lat, -iono_lat, -dlat_fieldline/100.)
        rads_fine = L*R_E*pow(np.cos(lats_fine*D2R),2.0)
        # Straight line step between each fine point. Whatever, it works.
        xx = rads_fine*np.cos(D2R*lats_fine)
        yy = rads_fine*np.sin(D2R*lats_fine)
        dx = np.diff(xx); dy = np.diff(yy)
        # 4.4.2017. This matches the anylitical version you did in mathematica
        dd = np.sqrt(dx**2 + dy**2)
        dist_n = np.cumsum(dd)   # Distance to northern hemi from lats_fine, in meters
        # print dd[0:5], lats_fine[0:5]
        # evaluate walt 4.25:
        Bs = np.sqrt(1.0 + 3.*pow(np.sin(D2R*lats_fine),2.))/pow(np.cos(D2R*lats_fine),6.0)
        Bmir = np.sqrt(1.0 + 3.*pow(np.sin(D2R*iono_lat),2.))/pow(np.cos(D2R*iono_lat),6.0)
        # print Bs, Bmir
        ftc_n_tmp = dd/np.sqrt(1.0 - Bs[:-1]/Bmir)
        ftc_n_tmp[np.isinf(ftc_n_tmp)] = 0
        ftc_n = np.cumsum(ftc_n_tmp)

        # Find locations of EA segments in the fine-scale integrated vector. Magic line.
        indices = np.abs(np.subtract.outer(lats_fine, lat_centers)).argmin(0)
        fieldline['ftc_n'] = ftc_n[indices]
        fieldline['ftc_s'] = ftc_n[-1] - fieldline['ftc_n']

        # fieldline['ftc_n'] = ftc_n[100*np.arange(0,len(lat_centers) + 1)]
        # fieldline['ftc_s'] = ftc_n[-1] - fieldline['ftc_n']
        # print fieldline['ftc_n']
        # print fieldline['ftc_s']

        fieldline['crossings'] = [[] for i in range(len(lat_centers))]   # Initialize an empty list to store crossings in
        # fieldline['crossings']  = []
        fieldline['stixR'] = np.zeros_like(lat_centers)
        fieldline['stixL'] = np.zeros_like(lat_centers)
        fieldline['stixP'] = np.zeros_like(lat_centers)
        fieldline['mu'] = np.zeros_like(lat_centers)
        fieldline['hit_counts'] = np.zeros_like(lat_centers)
        fieldlines.append(fieldline)
    return fieldlines

def find_crossings(ray_dir='/shared/users/asousa/WIPP/rays/2d/nightside/gcpm_kp0/',
                    mlt = 0,
                    tmax=10,
                    dt=0.1,
                    lat_low=None,
                    f_low=200, f_hi=30000,
                    center_lon = None,
                    lon_spacing = None,
                    itime = datetime.datetime(2010,1,1,0,0,0),
                    lat_step_size = 1,
                    n_sub_freqs=10,
                    Llims = [1.2, 8],
                    L_step = 0.2,
                    dlat_fieldline=1,
                    frame_directory=None,
                    DAMP_THRESHOLD = 1e-3):

    # Constants
    Hz2Rad = 2.*np.pi
    D2R = np.pi/180.
    R2D = 180./np.pi
    H_IONO_BOTTOM = 1e5
    H_IONO_TOP = 1e6
    R_E = 6371e3
    C = 2.997956376932163e8
    # DAMP_THRESHOLD = 1e-3  # Threshold below which we don't log a crossing




    lat_hi= lat_low + lat_step_size
   
    Lshells = np.arange(Llims[0], Llims[1], L_step)
    L_MARGIN = L_step/2.0
    # print "doing Lshells ", Lshells


    # Coordinate transform tools
    xf = xflib.xflib(lib_path='/shared/users/asousa/WIPP/WIPP_stencils/python/methods/libxformd.so')

    t = np.arange(0,tmax, dt)
    itime = datetime.datetime(2010,1,1,0,0,0)


    # Find available rays
    d = os.listdir(ray_dir)
    freqs = sorted([int(f[2:]) for f in d if f.startswith('f_')])
    d = os.listdir(os.path.join(ray_dir, 'f_%d'%freqs[0]))
    lons = sorted([float(f[4:]) for f in d if f.startswith('lon_')])
    d = os.listdir(os.path.join(ray_dir, 'f_%d'%freqs[0], 'lon_%d'%lons[0]))
    lats = sorted([float(s.split('_')[2]) for s in d if s.startswith('ray_')])

    # Latitude spacing:


    latln_pairs = [(lat_low, lat_hi)]

    # Adjacent frequencies to iterate over
    freqs =   [f for f in freqs if f >=f_low and f <= f_hi]
    freq_pairs = zip(freqs[0:-1],freqs[1:])
    

            
#--------------- Load and interpolate the center longitude entries ------------------------------------
    center_data = dict()
    for freq in freqs:
        logging.info("Loading freq: %d"%freq)
        for lat in [lat_low, lat_hi]:
            lon = center_lon
            filename = os.path.join(ray_dir,'f_%d'%freq,'lon_%d'%lon,'ray_%d_%d_%d.ray'%(freq,lat,lon))
            # print filename
            rf = read_rayfile(filename)[0]
            
            filename = os.path.join(ray_dir,'f_%d'%freq,'lon_%d'%lon,'damp_%d_%d_%d.ray'%(freq,lat,lon))
            df = read_damp(filename)[0]
            
            t_cur = t[t <= rf['time'].iloc[-1]]
            
            # Interpolate onto our new time axis:
            x = interpolate.interp1d(rf['time'],rf['pos']['x']).__call__(t_cur)/R_E
            y = interpolate.interp1d(rf['time'],rf['pos']['y']).__call__(t_cur)/R_E
            z = interpolate.interp1d(rf['time'],rf['pos']['z']).__call__(t_cur)/R_E

            d = interpolate.interp1d(df['time'],df['damping'], bounds_error=False, fill_value=0).__call__(t_cur)

            v = interpolate.interp1d(df['time'],rf['vgrel'], axis=0).__call__(t_cur)
            vmag = np.linalg.norm(v, axis=1)


            B = interpolate.interp1d(rf['time'],rf['B0'], axis=0).__call__(t_cur)
            Bnorm = np.linalg.norm(B, axis=1)
            Bhat = B/Bnorm[:,np.newaxis]

            stixR, stixL, stixP = calc_stix_parameters(rf, t_cur)

            n = interpolate.interp1d(df['time'],rf['n'],axis=0).__call__(t_cur)
            mu = np.linalg.norm(n, axis=1)

            # kvec = n*rf['w']/C
            # kz = -1.0*np.sum(kvec*Bhat, axis=1)  # dot product of rows
            # kx = np.linalg.norm(kvec + Bhat*kz[:,np.newaxis], axis=1)
            # psi = R2D*np.arctan2(-kx, kz)

            # kvec = n*rf['w']/C
            # kz = np.sum(kvec*Bhat, axis=1)  # dot product of rows
            # kx = np.linalg.norm(kvec - Bhat*kz[:,np.newaxis], axis=1)
            # psi = np.arctan2(kx, kz)

            # psi = R2D*np.arctan2(kx, kz)

            kvec = n*rf['w']/C
            kz = np.sum(kvec*Bhat, axis=1)  # dot product of rows
            kx = np.linalg.norm(np.cross(kvec, Bhat), axis=1) # Cross product of rows
            psi = np.arctan2(kx, kz)


            # Stash it somewhere:
            key = (freq, lat, lon)
            curdata = dict()

            # Flatten out any longitude variation, just to be sure:
            curdata['pos'] = flatten_longitude_variation(np.vstack([x,y,z]), itime, xf=xf)
            # curdata['pos'] = np.vstack([x,y,z])
            curdata['damp']= d
            curdata['nt'] = len(t_cur)
            curdata['stixR'] = stixR
            curdata['stixP'] = stixP
            curdata['stixL'] = stixL
            curdata['mu'] = mu
            curdata['psi'] = psi
            curdata['vgrel'] = vmag


            center_data[key] = curdata


#------------ Rotate center_longitude rays to new longitudes ---------------------------
    logging.info("Rotating to new longitudes")
    ray_data = dict()
    for key in center_data.keys():
        for lon in [center_lon - lon_spacing/2., center_lon + lon_spacing/2.]:
            newkey = (key[0], key[1], lon)
            dlon = lon - key[2] 
            d = dict()
            d['pos'] = rotate_latlon(center_data[key]['pos'],itime, 0, dlon, xf)
            d['damp']=center_data[key]['damp']
            d['stixR'] = center_data[key]['stixR']
            d['stixL'] = center_data[key]['stixL']
            d['stixP'] = center_data[key]['stixP']
            d['mu'] = center_data[key]['mu']
            d['psi'] = center_data[key]['psi']
            d['vgrel'] = center_data[key]['vgrel']
            ray_data[newkey] = d


# ------------------ Set up field lines ----------------------------
    logging.info("Setting up EA grid")
    fieldlines = gen_EA_array(Lshells, dlat_fieldline, lon, itime, L_MARGIN, xf = xf)


#----------- Step through and fill in the voxels (the main event) ---------------------
    logging.info("Starting interpolation")

    lat_pairs  = [(lat_low, lat_hi)]
    lon_pairs  = [(center_lon - lon_spacing/2., center_lon + lon_spacing/2.)]

    # output space
    nfl = len(fieldlines)
    nlons = 1
    nt = len(t)
    n_freq_pairs = len(freq_pairs)
    data_total = np.zeros([nfl, n_freq_pairs, nlons, nt])

    lon1 = center_lon - lon_spacing/2.
    lon2 = center_lon + lon_spacing/2.

    for t_ind in np.arange(nt - 1):
        # Per frequency
        data_cur = np.zeros(nfl)
        logging.info("t = %g"%(t_ind*dt))
        for freq_ind, (f1, f2) in enumerate(freq_pairs):
            # print "doing freqs between ", f1, "and", f2

            # Loop over adjacent sets:
            if n_sub_freqs == 0:
                ff = np.arange(0, (f2 - f1), 1)  # This version for uniform in frequency
            else:
                ff = np.arange(0, n_sub_freqs, 1)  # This version for constant steps per pair
            nf = len(ff)

            fine_freqs = f1 + (f2 - f1)*ff/nf
            # print fine_freqs

            for lat1, lat2 in lat_pairs:
                k0 = (f1, lat1, lon1)
                k1 = (f1, lat2, lon1)
                k2 = (f2, lat1, lon1)
                k3 = (f2, lat2, lon1)
                k4 = (f1, lat1, lon2)
                k5 = (f1, lat2, lon2)
                k6 = (f2, lat1, lon2)
                k7 = (f2, lat2, lon2)
                clat = (lat1 + lat2)/2.
                f_center = (f1 + f2)/2.

                tmax_local = min(np.shape(ray_data[k0]['pos'])[1], np.shape(ray_data[k1]['pos'])[1],
                                 np.shape(ray_data[k2]['pos'])[1], np.shape(ray_data[k3]['pos'])[1],
                                 np.shape(ray_data[k4]['pos'])[1], np.shape(ray_data[k5]['pos'])[1],
                                 np.shape(ray_data[k6]['pos'])[1], np.shape(ray_data[k7]['pos'])[1])
                if (t_ind < tmax_local - 1):

                    points_4d = np.hstack([np.vstack([ray_data[k0]['pos'][:,t_ind:t_ind+2],np.zeros([1,2])]),
                                           np.vstack([ray_data[k1]['pos'][:,t_ind:t_ind+2],np.zeros([1,2])]),
                                           np.vstack([ray_data[k2]['pos'][:,t_ind:t_ind+2],np.ones([1,2])*nf]),
                                           np.vstack([ray_data[k3]['pos'][:,t_ind:t_ind+2],np.ones([1,2])*nf]),
                                           np.vstack([ray_data[k4]['pos'][:,t_ind:t_ind+2],np.zeros([1,2])]),
                                           np.vstack([ray_data[k5]['pos'][:,t_ind:t_ind+2],np.zeros([1,2])]),
                                           np.vstack([ray_data[k6]['pos'][:,t_ind:t_ind+2],np.ones([1,2])*nf]),
                                           np.vstack([ray_data[k7]['pos'][:,t_ind:t_ind+2],np.ones([1,2])*nf])])
                    
                    voxel_vol = voxel_vol_nd(points_4d)*pow(R_E,3.)

                    # damps_2d = np.hstack([ray_data[k0]['damp'][t_ind:t_ind+2],
                    #                       ray_data[k1]['damp'][t_ind:t_ind+2],
                    #                       ray_data[k2]['damp'][t_ind:t_ind+2],
                    #                       ray_data[k3]['damp'][t_ind:t_ind+2]])
                    # damping_avg = np.mean(damps_2d)
                    damping_pts  = np.hstack([ray_data[kk]['damp'][t_ind:t_ind+2]  for kk in [k0, k1, k2, k3, k4, k5, k6, k7]])
                    damp_interp   = interpolate.NearestNDInterpolator(points_4d.T, damping_pts)


                    points_2d = np.hstack([np.vstack([ray_data[k4]['pos'][[0,2],t_ind:t_ind+2], np.zeros([1,2])]),
                                           np.vstack([ray_data[k5]['pos'][[0,2],t_ind:t_ind+2], np.zeros([1,2])]),
                                           np.vstack([ray_data[k6]['pos'][[0,2],t_ind:t_ind+2], np.ones([1,2])*nf]),
                                           np.vstack([ray_data[k7]['pos'][[0,2],t_ind:t_ind+2], np.ones([1,2])*nf])])

                    # We really should interpolate these 16 corner points instead of just averaging them.
                    stixR_pts  = np.hstack([ray_data[kk]['stixR'][t_ind:t_ind+2]  for kk in [k0, k1, k2, k3, k4, k5, k6, k7]])
                    stixL_pts  = np.hstack([ray_data[kk]['stixL'][t_ind:t_ind+2]  for kk in [k0, k1, k2, k3, k4, k5, k6, k7]])
                    stixP_pts  = np.hstack([ray_data[kk]['stixP'][t_ind:t_ind+2]  for kk in [k0, k1, k2, k3, k4, k5, k6, k7]])
                    mu_pts  = np.hstack([ray_data[kk]['mu'][t_ind:t_ind+2]  for kk in [k0, k1, k2, k3, k4, k5, k6, k7]])
                    psi_pts = np.hstack([ray_data[kk]['psi'][t_ind:t_ind+2] for kk in [k0, k1, k2, k3, k4, k5, k6, k7]])
                    vel_pts = np.hstack([ray_data[kk]['vgrel'][t_ind:t_ind+2] for kk in [k0, k1, k2, k3, k4, k5, k6, k7]])

                    stixR_interp = interpolate.NearestNDInterpolator(points_4d.T, stixR_pts)
                    stixL_interp = interpolate.NearestNDInterpolator(points_4d.T, stixL_pts)
                    stixP_interp = interpolate.NearestNDInterpolator(points_4d.T, stixP_pts)
                    mu_interp    = interpolate.NearestNDInterpolator(points_4d.T, mu_pts)
                    psi_interp   = interpolate.NearestNDInterpolator(points_4d.T, psi_pts)
                    vel_interp   = interpolate.NearestNDInterpolator(points_4d.T, vel_pts)

                    # tri = Delaunay(points_2d.T, qhull_options='QJ')
                    tri = Delaunay(points_4d.T, qhull_options='QJ')
                    
                    # Loop through the output fieldlines
                    for fl_ind, fl in enumerate(fieldlines):
                        ix = np.arange(0,len(fl['pos']))
                        ief= np.arange(0, nf)    
                        px, pf = np.meshgrid(ix, ief, indexing='ij')  # in 3d, ij gives xyz, xy gives yxz. dumb.
                        # newpoints = np.hstack([fl['pos'][px.ravel(),:][:,[0,2]], np.atleast_2d(ff[pf.ravel()]).T])
                        newpoints = np.hstack([fl['pos'][px.ravel(),:], np.atleast_2d(ff[pf.ravel()]).T])

                        mask = (tri.find_simplex(newpoints) >= 0)*1.0
                        # mask = mask.reshape([len(ix), len(ief)])

                        # Entries in newpoints are inside the volume if mask is nonzero
                        # (Mask gives the index of the triangular element which contains it) 
                        # for row in newpoints[mask > 0]:
                        #     print "L:", fl['L'], xf.sm2rllmag(row[:-1], itime)
                        #     fieldlines[fl_ind]['crossings'].append(xf.sm2rllmag(row[:-1], itime))


                        mask = mask.reshape([len(ix), len(ief)])
                        minds = np.nonzero(mask)
                        if len(minds[0]) > 0:
                            # unscaled_pwr = (damping_avg/voxel_vol)
                            hit_lats = fl['lat'][minds[0]]
                            hit_freqs= fine_freqs[minds[1]]
                        #     # print "t = ", t_ind, "L = ", fl['L']
                            # print hit_lats, hit_freqs

                            # hit latitude, hit frequency (indices)
                            for hl, hf in zip(minds[0], minds[1]):

                                cur_pos = np.hstack([fl['pos'][hl,:], ff[hf]])
                                psi = psi_interp(cur_pos)[0]
                                mu = mu_interp(cur_pos)[0]
                                damp = damp_interp(cur_pos)[0]
                                vel = vel_interp(cur_pos)[0]*C

                                #              [unitless][m/s][1/m^3] ~ 1/m^2/sec. Multiply by total input energy.
                                if (damp > DAMP_THRESHOLD):
                                    pwr_scale_factor = damp*vel/voxel_vol
                                    tt = np.round(100.*t_ind*dt)/100.
                                    fieldlines[fl_ind]['crossings'][hl].append((tt, fine_freqs[hf], pwr_scale_factor, psi, mu, damp, vel))
                                    # fl['crossings'].append([fl['L'], fl['lat'][hl], t_ind*dt, fine_freqs[hf]])
                            #         # Stix parameters are functions of the background medium only,
                            #         # but we'll average them because we're grabbing them from the
                            #         # rays at slightly different locations within the cell.
                            #         # print np.shape(fl['pos'])

                                    fieldlines[fl_ind]['stixR'][hl] += stixR_interp(cur_pos)[0]
                                    fieldlines[fl_ind]['stixL'][hl] += stixL_interp(cur_pos)[0]
                                    fieldlines[fl_ind]['stixP'][hl] += stixP_interp(cur_pos)[0]
                                    fieldlines[fl_ind]['hit_counts'][hl] += 1

    # logging.info("finished with interpolation")
    logging.info("finished with interpolation")

    # Average the background medium parameters:

    for fl_ind, fl in enumerate(fieldlines):
        for lat_ind in range(len(fl['crossings'])):
            n_hits = fl['hit_counts'][lat_ind]
            if n_hits > 0:
                # print fl['L'], ":", fl['lat'][lat_ind],": hit count: ", fl['hit_counts'][lat_ind]

                # average stixR, stixL, stixP
                fl['stixP'][lat_ind] /= n_hits
                fl['stixR'][lat_ind] /= n_hits
                fl['stixL'][lat_ind] /= n_hits
                fl['hit_counts'][lat_ind] = 1


    out_data = dict()
    out_data['fieldlines'] = fieldlines
    out_data['time'] = t
    out_data['Lshells'] = Lshells
    out_data['lat_low'] = lat_low
    out_data['lat_hi'] = lat_hi
    out_data['fmin'] = f_low
    out_data['fmax'] = f_hi
    out_data['freq_pairs'] = freq_pairs
    

    return out_data







if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s] %(message)s')  


    data = find_crossings(ray_dir='/shared/users/asousa/WIPP/rays/2d/nightside/ngo_v2/',
                            tmax = 20,
                            dt = 0.02,
                            lat_low = 31,
                            lat_step_size=1,
                            f_low = 200, f_hi = 230,
                            n_sub_freqs = 20,
                            Llims = [2,4],
                            L_step = 0.5,
                            dlat_fieldline = 1
                            )

    with open('test_dump.pkl','w') as f:
        pickle.dump(data, f)
