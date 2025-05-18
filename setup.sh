#!/bin/bash
git clone --recurse-submodules https://github.com/mikelma/craftium.git
sudo chmod -R 777 craftium
cd craftium

sudo apt install g++ make libc6-dev cmake libpng-dev libjpeg-dev libgl1-mesa-dev libsqlite3-dev libogg-dev libvorbis-dev libopenal-dev libcurl4-gnutls-dev libfreetype6-dev zlib1g-dev libgmp-dev libjsoncpp-dev libzstd-dev libluajit-5.1-dev gettext libsdl2-dev

sudo apt install libpython3-dev

pip install -U setuptools
