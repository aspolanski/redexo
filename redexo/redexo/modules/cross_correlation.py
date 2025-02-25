__all__ = ['CrossCorrelationModule']
from .base import Module
from ..core.dataset import CCF_Dataset
import astropy.constants as const
from scipy.interpolate import interp1d
import numpy as np
import sys
import matplotlib.pyplot as plt


class CrossCorrelationModule(Module):
    def initialise(self, template, template_wl, rv_range, drv, error_weighted=False, mask_rm=False,rm_mask_range=None,rm_mask_value=0.0,subtract_oot=False,target=None,phase=None):
        self.template = template
        self.template_wl= template_wl
        self.rv_range = rv_range
        self.drv = drv
        self.rv_grid = np.arange(-rv_range, rv_range+1e-4, drv)
        self.beta_grid = 1.0-self.rv_grid/const.c.to('km/s').value
        self.error_weighted = error_weighted
        self.mask_rm = mask_rm
        self.rm_mask_range = rm_mask_range
        self.rm_mask_value = rm_mask_value
        self.subtract_oot = subtract_oot
        self.target = target
        self.phase = phase


    def process(self, dataset, debug=False):
        CCF = np.zeros((dataset.num_exposures, dataset.num_orders, self.rv_grid.size))
        for exp in range(dataset.num_exposures):
            nans = np.isnan(dataset.spec[exp])
            wl_exp = dataset.wavelengths[exp][~nans]
            spec_exp = dataset.spec[exp][~nans]
            errors_exp = dataset.errors[exp][~nans]
            shifted_wavelengths = wl_exp*self.beta_grid[:,np.newaxis]
            
            shifted_template = interp1d(self.template_wl, self.template, bounds_error=True)(shifted_wavelengths)
            shifted_template = shifted_template.T - np.mean(shifted_template, axis=1)
            if self.error_weighted:
                CCF[exp] = (spec_exp/errors_exp**2).dot(shifted_template)
            else:
                CCF[exp] = spec_exp.dot(shifted_template)
        
        if self.subtract_oot:
            try:
                phases = self.target.orbital_phase(dataset.obstimes)
                phase_idx = np.where( (phases>self.phase) | (phases<-self.phase))[0]

                avg_oot=np.nanmean(CCF[phase_idx,0,:],axis=0)
                CCF = ((CCF[:,0,:]+np.max(CCF[:,0,:]))/(avg_oot[None,:]++np.max(CCF[:,0,:])))-1.0 #adding max values to prevent Div0 errors.
                CCF = CCF[:,np.newaxis,:]
            
            except:
                print("None value. Please provide both a target and phase cut-off")
                sys.exit(1)

        if self.mask_rm:
            try:
                rv_idx = np.where( (self.rv_grid>self.rm_mask_range[0]) & (self.rv_grid<self.rm_mask_range[1]) )[0]
                CCF[:,0,rv_idx] = self.rm_mask_value

            except:
                print("None value. Please provide both a mask range and mask value")
                sys.exit(1)

        
        
        rv_matrix= np.tile(self.rv_grid[np.newaxis, np.newaxis,:], (dataset.num_exposures, dataset.num_orders, 1))
        res = CCF_Dataset(spec=CCF, rv_grid=rv_matrix, vbar=dataset.vbar, obstimes=dataset.obstimes)
        
                
        return res
