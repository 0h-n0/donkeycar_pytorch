# Donkeycar_pytorch

Pytorch compatible version.

## You can find more information on original donkeycar repo.

* https://github.com/autorope/donkeycar


## How to build pytorch on your raspberry pi 3

### increase swap size

change `CONV_SWAPSIZE` in `/etc/dphys-swapfile`.

```
CONF_SWAPSIZE=2048
```


```
$ sudo systemctl restart dphys-suwapfile
```

### 


```shell
$ sudo apt update && sudo apt upgrade
$ sudo apt install libopenblas-dev libblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools
```

```shell
$ export NO_CUDA=1
$ export NO_DISTRIBUTED=1
$ export NO_MKLDNN=1 
$ export NO_NNPACK=1
$ export NO_QNNPACK=1
```

```shell
$ git clone --recursive https://github.com/pytorch/pytorch
$ cd pytorch
```


```shell
$ nohup python3 setup.py build &
```

```shell
$ python setup.py install
```

### set up your environmnet

```
$ sudo apt-get install -y virtualenv build-essential python3-dev gfortran libhdf5-dev
$ cd donkeycar_pytorch
$ python setup.py install 
```

### Get driving.
After building a Donkey2 you can turn on your car and go to http://localhost:8887 to drive.

### How to train a PyTorch model.

```shell
$ donkey createcar --template torchdonkey torch_donkey
$ cd torch_donkey
$ python manage.py train --tub=data/log --model=models/mypilot
```

### How to drive with PyTorch model.

```shell
$ cd torch_donkey
$ python manage.py drive --model=models/mypilot
```

