"""Fit and export actual NCEP reanalysis records with day/datetime times."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.io import netcdf_file
from scipy.spatial.distance import cdist

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--data',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--space-unit-km',type=float,default=1.)
a=p.parse_args()
source=a.source.resolve();sys.path.insert(0,str(source))
import skgstat,gstools
from skgstat import SpatioTemporalVariogram
assert Path(skgstat.__file__).resolve().parent.parent==source
assert hashlib.sha256(a.data.read_bytes()).hexdigest()=='c606b89c35970a2983b914b76df4adbb409003ef34aa7cfd7f582e41f307482b'
with netcdf_file(a.data,mmap=False) as f:
    dates=np.datetime64('1800-01-01T00','h')+np.rint(f.variables['time'][:]).astype('timedelta64[h]')
    selected=np.linspace(0,len(dates)-1,24,dtype=int)
    dates=dates[selected]
    lat=f.variables['lat'][:].copy();lon=f.variables['lon'][:].copy()
    air=f.variables['air'][:].copy()*float(f.variables['air'].scale_factor)
    ij=np.array([(i,j) for i in [10,12,14] for j in [15,17,19]])
    # Explicit local equirectangular coordinates in km. This is an
    # algorithm/unit check, not a geographic model-quality assessment.
    lat0=np.mean(lat[ij[:,0]])
    xy=np.column_stack(((lon[ij[:,1]]-np.mean(lon[ij[:,1]]))*111.195*np.cos(np.deg2rad(lat0)),
                        (lat[ij[:,0]]-lat0)*111.195))
    values=np.array([air[selected,i,j] for i,j in ij]).ravel()
xy/=a.space_unit_km
times=np.tile(dates,len(xy));coordinates=np.repeat(xy,len(dates),axis=0)
days=(times-times.min())/np.timedelta64(1,'D')
train=np.arange(len(values))%4!=0
records=[];arrays={}


def cov(distance,name,sill):
    if name=='exponential':return sill*np.exp(-3*distance)
    return sill*np.exp(-4*distance**2)


for name in ['exponential','gaussian']:
    for nugget in [False,True]:
        for unit in ['days','datetime']:
            factor=1. if unit=='days' else 86400.
            c=pd.DataFrame(coordinates[train],columns=['x','y'])
            c['time']=days[train] if unit=='days' else times[train]
            v=SpatioTemporalVariogram(c,values[train],model='metric',marginal_model=name,
                                      use_nugget=nugget,x_lags=6,t_lags=6)
            params=v.parameters.copy(); normalized=params.copy();normalized['time_range']/=factor
            krige=v.to_gs_krige()
            cond=np.column_stack((v.coordinates,v.time))
            target_time=days[~train] if unit=='days' else (
                pd.DatetimeIndex(times[~train])-v.time_origin).total_seconds().to_numpy()
            target=np.column_stack((coordinates[~train],target_time))
            field,variance=krige(target.T)
            scales=np.array([params['space_range'],params['space_range'],params['time_range']])
            K=cov(cdist(cond/scales,cond/scales),name,params['sill'])
            K.flat[::len(K)+1]+=params['nugget']
            C=cov(cdist(cond/scales,target/scales),name,params['sill'])
            M=np.block([[K,np.ones((len(K),1))],[np.ones((1,len(K))),np.zeros((1,1))]])
            rhs=np.vstack((C,np.ones((1,len(target)))))
            weights=np.linalg.solve(M,rhs)
            expected=values[train]@weights[:-1]
            expected_variance=params['sill']+params['nugget']-np.sum(C*weights[:-1],axis=0)-weights[-1]
            key=f'{name}_{nugget}_{unit}'
            arrays[key+'_field']=field;arrays[key+'_variance']=variance
            arrays[key+'_bins_space']=v.xbins;arrays[key+'_bins_time']=v.tbins/factor
            arrays[key+'_experimental']=v.experimental
            model=v.to_gstools()
            # The actual exported covariance must agree at all conditioning
            # and held-out pairs, not just reproduce conditioning values.
            model_cov=model.cov_spatial((cond[:,None,:]-target[None,:,:]).reshape(-1,3).T).reshape(C.shape)
            residual=v.transform(np.column_stack([z.ravel() for z in v.meshbins]))-v.experimental.ravel()
            rec=dict(name=key,parameters=normalized,
                covariance_error=float(np.max(np.abs(model_cov-C))),
                prediction_error=float(np.max(np.abs(field-expected))),
                variance_error=float(np.max(np.abs(variance-expected_variance))),
                matrix_condition=float(np.linalg.cond(M)),
                sse=float(np.nansum(residual**2)),field=field.tolist(),variance=variance.tolist())
            records.append(rec);print({k:x for k,x in rec.items() if k not in ['field','variance']},flush=True)

r=dict(source=str(source),package=skgstat.__file__,gstools=gstools.__version__,numpy=np.__version__,
       observations=len(values),train=int(train.sum()),held_out=int((~train).sum()),dates=[str(dates.min()),str(dates.max())],
       space_unit_km=a.space_unit_km,
       data_hash=hashlib.sha256(a.data.read_bytes()).hexdigest(),records=records,
       source_hash=hashlib.sha256((source/'skgstat/SpatioTemporalVariogram.py').read_bytes()).hexdigest())
a.output.write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed(a.output.with_suffix('.npz'),**arrays)
