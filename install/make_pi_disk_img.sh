# Manually make sure the camera and I2C are enabled.
sudo raspi-config


#standard updates (5 min)
sudo apt update -y
sudo apt upgrade -y
sudo rpi-update -y

#helpful libraries (2 min)
sudo apt install build-essential python3-dev python3-distlib python3-setuptools  python3-pip python3-wheel -y
sudo apt install libzmq-dev -y
sudo apt install xsel xclip -y
sudo apt install python3-h5py -y

#remove python2 (1 min)
sudo apt-get remove python2.7 -y
sudo apt-get autoremove -y


#create a python virtualenv (2 min)
sudo apt install virtualenv -y
virtualenv env --system-site-packages --python python3
echo '#start env' >> ~/.bashrc
echo 'source ~/env/bin/activate' >> ~/.bashrc
source ~/.bashrc

#install numpy and pandas (3 min)
sudo apt install libxml2-dev python3-lxml -y
sudo apt install libxslt-dev -y

# install numpy which is needed for opencv install
pip install pandas #also installs numpy


#install redis-server (1 min)
sudo apt install redis-server


#install opencv (1 hour)
#instructions from:https://raspberrypi.stackexchange.com/questions/69169/how-to-install-opencv-on-raspberry-pi-3-in-raspbian-jessie
sudo apt-get install build-essential git cmake pkg-config -y
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev -y
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y
sudo apt-get install libxvidcore-dev libx264-dev -y
sudo apt-get install libatlas-base-dev gfortran -y

# NOTE: this gets the dev version. Use tags to get specific version
git clone https://github.com/opencv/opencv.git --depth 1
git clone https://github.com/opencv/opencv_contrib.git --depth 1

cd ~/opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D BUILD_EXAMPLES=OFF ..
make -j4
sudo make install
sudo ldconfig


#install tensorflow (5 min)
tf_file=tensorflow-1.7.0-cp35-none-linux_armv7l.whl
wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.7.0/${tf_file}
pip install ${tf_file}
rm ${tf_file}


#install donkey (1 min)
git clone https://github.com/wroscoe/donkey.git donkeycar
pip install -e donkeycar/[pi]


