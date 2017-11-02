import xflib
import numpy as np
def input_power_scaling(flash_loc, ray_loc, mag_lat, w, i0, MLT, xf=None):
    if xf is None:
        xf = xflib.xflib(lib_path='/shared/users/asousa/WIPP/3dWIPP/python/libxformd.so')
    f = w/(2.0*np.pi);
    D2R = np.pi/180.
    
    Z0 = 377.0
    P_A= 5e3
    P_B= 1e5
    H_E= 5000.0

    H_IONO = 1e5
    R_E = 6371e3
    flash_loc_sph = xf.c2s(flash_loc)
    ray_loc_sph   = xf.c2s(ray_loc)
    out_lat = ray_loc_sph[1]
    out_lon = ray_loc_sph[2]
    inp_lat = flash_loc_sph[1]
    inp_lon = flash_loc_sph[2]
    
    dlat  = D2R*(out_lat - inp_lat)
    dlong = D2R*(out_lon - inp_lon)
    clat1 = np.cos(D2R*inp_lat)
    clat2 = np.cos(D2R*out_lat)
    slat1 = np.sin(D2R*inp_lat)
    slat2 = np.sin(D2R*out_lat)
    # Compute current (latitude and longitude dependent) weighting:
    # (Use Haversine formula since we're moving further out than reasonable for Cartesian)
    a = pow(np.sin(dlat/2.0),2)
    b = (clat1*clat2*pow(np.sin(dlong/2.0),2))
    gc_distance = 2.0*R_E * np.arcsin(np.sqrt(a + b))

    dist_tot= np.hypot(gc_distance, H_IONO)
    xi = np.arctan2(gc_distance, H_IONO)
    
    w_sq =  pow( w , 2 );
    S = ( (1/Z0) * pow( (H_E*i0*2e-7*(np.sin(xi)/dist_tot)*w*(P_A-P_B)) , 2 )
                   /  (  (w_sq+pow(P_A,2))*(w_sq+pow(P_B,2))  )      ) ;
    S_vert = S * np.cos(xi) ;  #// factor for vert prop.

#     // Ionosphere absorption model
    attn_factor = pow(10,-(graf_iono_model(mag_lat,f, MLT )/10)  );
    S_vert = S_vert * attn_factor;

    return S_vert




def total_input_power(flash_pos_sm, i0, 
                        latmin, latmax, lonmin, lonmax, wmin, wmax, itime_in):
    # // Determine the total input power tracked by the set of guide rays:

    xf = xflib.xflib(lib_path='/shared/users/asousa/WIPP/3dWIPP/python/libxformd.so')
    tot_pwr = 0;
    pwr = 0;
    dlat = 0.1; 
    dlon = 0.1;
    dw   = 100*2*np.pi;
    tmp_coords = np.zeros(3);
    x_sm = np.zeros(3)

    D2R = np.pi/180.
    H_IONO = 1e5
    R_E = 6371e3

    for w in np.arange(wmin+dw/2, wmax, dw):
    # for (double w = wmin + dw/2; w < wmax; w+=dw) {
        for lat in np.arange(latmin + dlat/2, latmax, dlat):
        # for (double lat = latmin + dlat/2; lat < latmax; lat+=dlat) {
            for lon in np.arange(lonmin + dlon/2, lonmax, dlon):
            # for (double lon=lonmin; lon < lonmax; lon+=dlon) {
                # // cout << "(" << lat << ", " << lon << ")\n";
                tmp_coords = [1 + H_IONO/R_E, lat, lon];
                x_sm = xf.rllmag2sm(tmp_coords, itime_in);
                mlt = lon2MLT(itime_in, lon, xf);
                pwr = input_power_scaling(flash_pos_sm, x_sm, lat, w, i0, mlt, xf);

                dist_lat = (R_E + H_IONO)*dlat*D2R;
                dist_lon = (R_E + H_IONO)*dlon*np.cos(D2R*lat)*D2R;
                                # // Latitude distance      longitude distance       freq dist
                # // cout << "dist_lat: " << dist_lat << ", dist_lon: " << dist_lon << "\n";
                tot_pwr += pwr * dist_lat * dist_lon * dw;
            
    return tot_pwr;




def graf_iono_model(lats, f, MLT=0):
    # A model of VLF wave power attenuation between 100 and 1000 km altitude.
    # Based on Graf and Cohen 2013, figure 7.
    # (Data picked from plots and fitted to an exponential function)
    
    mltslope = 0.5
    
    # Fit parameters:
    # function: f = p[0]*exp(-x/p[1]) + p[2]
    p20D = [ 215.74122661,   11.9624129,  9.01400095]
    p20N = [ 117.54370955,   7.40762459,  0.90050155]
    p2D  = [ 55.94274086,    11.91761368, 4.09353494]
    p2N  = [ 11.99682851,    9.53682009,  0.23617706]
    
    # Attenuation values at 20kHz and 2kHz, day and night
    a20D  = np.log10(p20D[0]*np.exp(-np.abs(lats)/p20D[1]) + p20D[2])
    a20N  = np.log10(p20N[0]*np.exp(-np.abs(lats)/p20N[1]) + p20N[2])
    a2D   = np.log10(p2D[0]*np.exp(-np.abs(lats)/p2D[1]) + p2D[2])
    a2N   = np.log10(p2N[0]*np.exp(-np.abs(lats)/p2N[1]) + p2N[2])
    
    # interpolate / extrapolate between the 20kHz and 2kHz values
    mD = (a20D - a2D)
    mN = (a20N - a2N)
    cD = (a2D + a20D)/2. - mD*0.8010
    cN = (a2N + a20N)/2. - mN*0.8010
    
    # Fit values for day and night
    aD = np.power(10, mD*np.log10(f*1e-3) + cD)
    aN = np.power(10, mN*np.log10(f*1e-3) + cN)
    
    # Weight day + night according to MLT (logistic function)
    s1 = 1.0/(1 + np.exp((np.mod(MLT, 24) - 18)/mltslope))
    s2 = 1.0/(1 + np.exp((np.mod(MLT, 24) - 6)/mltslope))
    s = s1 - s2
    
    # Select day curve for s = 1, night curve for s = 0
    a = s*aD + (1.0 - s)*aN
    
    return a



def lon2MLT(itime, lon, xf=None):
    # // Input: itime, lon in geomagnetic dipole coords.
    # // Output: MLT in fractional hours
    # // Ref: "Magnetic Coordinate Systems", Laundal and Richmond
    # // Space Science Review 2016, DOI 10.1007/s11214-016-0275-y
    if xf is None:
        xf = xflib.xflib(lib_path='/shared/users/asousa/WIPP/3dWIPP/python/libxformd.so')

    ut_hr = itime.hour + itime.minute/60 # /1000.0/60.0;  #// Milliseconds to fractional hours (UT)
    A1 = [1, 51.48, 0];         #// Location of Greenwich (for UT reference) 
    B1 = [0, 0, 0]; # B1[3]                         // Location of Greenwich in geomag

    xf.s2c(A1);
    xf.geo2mag(A1, itime);
    xf.c2s(A1);

    return np.mod(ut_hr + (lon - A1[2])/15.0,  24);

def MLT2lon(itime, mlt, xf=None):
    # // Input: itime, mlt in fractional hours
    # // Output: longitude in geomagnetic coordinates
    # // Ref: "Magnetic Coordinate Systems", Laundal and Richmond
    # // Space Science Review 2016, DOI 10.1007/s11214-016-0275-y
    if xf is None:
        xf = xflib.xflib(lib_path='/shared/users/asousa/WIPP/3dWIPP/python/libxformd.so')

    ut_hr = itime.hour + itime.minute/60 # /1000.0/60.0;  #// Milliseconds to fractional hours (UT)
    A1 = [1, 51.48, 0];         #// Location of Greenwich (for UT reference) 
    B1 = [0, 0, 0]; # B1[3]                         // Location of Greenwich in geomag

    xf.s2c(A1);
    xf.geo2mag(A1, itime);
    xf.c2s(A1);
    
    return 15.*(mlt - ut_hr) + A1[2]