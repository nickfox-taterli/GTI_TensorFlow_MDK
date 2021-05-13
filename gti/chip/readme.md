GTI USB dongle users that do not wish to install GTI SDK separately, should perform the following commands in terminal before inferencing on chip in TensorFlow MDK for the first time. These commands copy USB driver libraries and rules to the host system. 
```sudo cp libftd3xx.* /usr/lib/```
```sudo cp *.rules /etc/udev/rules.d/```