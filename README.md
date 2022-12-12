# Final Project for ME 231A 
This repository contains the code for our final project of Prof. Francesco Borrelli's course "Experiential Advanced Control Design I" at UC Berkeley.

<img src="results/animation/animation.gif" width="200" height="400" />

## Installation
Install Python packages from the root directory:
```
pip install -r requirements.txt
```
Install IPOPT:
```
apt-get install -y -qq glpk-utils
wget -N -q "https://portal.ampl.com/dl/open/ipopt/ipopt-linux64.zip"
unzip -o -q ipopt-linux64
apt-get install -y -qq coinor-cbc
```
