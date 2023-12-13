# TReduct
This package makes ZTF transient data easier to handle. It allows the user to retrieve, display, and reduce dimensionality of transient detections. 

## Usage
```python
from ztf_data import ZtfData
ztf = ZtfData()
ztf.display_transient()

ztf.reduce(type='tsvd', num_dimensions=15)
ztf.reduce(type='tsne', num_dimensions=2)
ztf.normalize()
ztf.plot_2d_data()
```
