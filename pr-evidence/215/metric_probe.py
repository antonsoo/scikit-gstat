import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();source=a.source.resolve();sys.path.insert(0,str(source))
import skgstat
from skgstat import SpatioTemporalVariogram
assert Path(skgstat.__file__).resolve().parent.parent==source
rng=np.random.default_rng(215)
c=np.column_stack((rng.normal(size=(30,2)),np.arange(30)/10));y=rng.normal(size=30)
records=[]
for name in ['spherical','exponential','gaussian','cubic','stable','matern']:
 for metric,h in [('euclidean',np.sqrt(2.)),('cityblock',2.),('sqeuclidean',2.)]:
  v=SpatioTemporalVariogram(c,y,xdist_func=metric,model='metric',marginal_model=name)
  v.cof=np.array([4.,2.,1.]+([1.3] if name in ['stable','matern'] else []))
  expected=float(v.transform([[h,0]])[0]);actual=float(v.to_gstools().vario_spatial([[1.],[1.],[0.]])[0])
  records.append(dict(model=name,metric=metric,expected=expected,exported=actual,error=actual-expected))
r=dict(source=str(source),source_hash=hashlib.sha256((source/'skgstat/interfaces/gstools.py').read_bytes()).hexdigest(),records=records)
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(records,indent=2))
