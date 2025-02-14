# `cump2` subpackage

The cump2 subpackage provides GPU-accelerated MP2 calculations. It exports two functions: `DFMP2` and `DFKMP2`, which support both open boundary conditions (OBC) and periodic boundary conditions (PBC). For PBC, only gamma-point calculations are currently supported.

Examples can be found in the following files:
* `byteqc/cump2/tests/mp2.py``
* `byteqc/cump2/tests/mp2_gamma_point.py``
