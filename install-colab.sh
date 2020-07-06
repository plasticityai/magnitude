#!/bin/bash

git clone https://github.com/plasticityai/magnitude.git
cd magnitude/
python setup.py install -vvvv
cd ../
rm -rf magnitude/