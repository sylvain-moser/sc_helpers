## Dependencies: 

This package actually depends on the following packages, but they are not listed in pyproject.toml because when installing the package with pip inside conda, I don't want pip to mess up with conda installed dependencies. So they have to be installed in your conda env before pip installing this package: 

- scanpy
- numpy
- pandas 
- joblib
- scipy
- sklearn
- harmonypy