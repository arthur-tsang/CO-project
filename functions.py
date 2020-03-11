import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy import constants as c

from scipy.special import iv, kv
from scipy.optimize import curve_fit

G = c.G.to(u.km**2 * u.kpc / u.Msun / u.s**2).value
def point_fit(r, m):
    # m in Msun
    return np.sqrt(G*m/r)

def uniform_fit(r, rho):
    # rho in Msun / kpc^3
    return r * np.sqrt(4/3*np.pi*G*abs(rho))

def bulge_fit(r, rb, mb):
    return np.sqrt(G * abs(mb) * r) / (r+abs(rb))

def expdisk_fit(r, rd, sig0):
    # sig0 in Msun/kpc^2
    # rd in kpc
    # y is unitless
    rd, sig0 = abs(rd), abs(sig0)
    y = r/rd/2
    bess = iv(0,y)*kv(0,y) - iv(1,y)*kv(1,y)
    return 4*np.pi*G*sig0*rd*y**2 * bess

def halo_fit(r, rh, rhoh):
    # rh in kpc
    # rhoh in Msun / kpc**2
    rh, rhoh = abs(rh), abs(rhoh)
    vel_inf = np.sqrt(4*np.pi*G*rhoh* rh**2)
    return vel_inf * (1 - rh/r * np.arctan(r/rh))

def bulgedisk_fit(r, rb, mb, rd, sig0):
    return np.sqrt(bulge_fit(r, rb, mb)**2 + expdisk_fit(r, rd, sig0)**2)

def diskhalo_fit(r, rd, sig0, rh, rhoh):
    return np.sqrt(expdisk_fit(r, rd, sig0)**2 + halo_fit(r,rh,rhoh)**2)

def total_fit(r, rb, mb, rd, sig0, rh, rhoh):
    return np.sqrt(bulge_fit(r, rb, mb)**2 + expdisk_fit(r, rd, sig0)**2 + halo_fit(r,rh,rhoh)**2)

def double_gaussian(x, mean, sigma, norm, mean2, sigma2, norm2):
    return norm * np.exp(-(x-mean)**2/(2*sigma**2)) + norm2 * np.exp(-(x-mean2)**2/(2*sigma2**2))