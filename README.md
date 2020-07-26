# self-driving-car-using-google-colab-full-schema-and-neural-network-

using Raspberry Pi and Machine Learning using Google Colab
SACHIN KUMAR
SACHIN KUMAR
Follow
Jun 29, 2019 · 11 min read



Image for post
In this tutorial, we will learn how to build a Self-Driving RC Car using Raspberry Pi and Machine Learning using Google Colab. Ever since the thought and discussion and hype about self-driving cars came into existence, I always wanted to build one on my own. This tutorial is a very baby step towards that reality and will also provide you with some in-depth analysis and knowledge into the basics of self-driving cars. This tutorial will also cover how to train your model using Google Colab
Image for post
Introduction to Donkey Car
Requirements and Parts Needed
Assemble the Hardware
Software Installation
Calibrate your Car
Take it for a spin
Introduction to Google Colab
Train an autopilot using Colab
Experiment and have fun
Conclusion
References
1. Introduction to Donkey Car
Image for post
Firstly, after a few research, I came across Donkey Car. Donkey is a high-level self-driving library written in Python. Also, it was developed with a focus on enabling fast experimentation and easy contribution. Hence, this was essential for me to quickly get started.
2. Requirements and Parts Needed
2.1 Buying the parts and components
We have 2 options for building our Donkey Car.
Option 1: Buying through an official Donkey Store
There are 2 official stores to buy your Donkey Care Starter Kit.
If you are in the US, you can buy from Donkey Store.
If you are in the Asia region and others, you can buy from the Robocar Store.
Option 2: Bottoms Up Build
If you want to buy the parts yourself, want to customize your donkey or live out to of the US, you may want to choose the bottoms up build. Furthermore, instructions regarding the same are mentioned in the documentation.
# 2.2 Major Components and Parts
Raspberry Pi 3
MicroSD card
RC Car
USB Battery with micro-USB cable
Donkey Partial Kit
# 3. Assemble the hardware
The starter kit includes everything you need to get started. It includes:
Image for post
HSP 94186 Brushed RC Car
An RC car fully tested with the Donkey Car platform
3D Printed Top Cage (in White or other color)
Laser-cut Base Plate (in White or transparent)
Raspberry Pi 3B or 3B+ (Depends on availability)
The brain of the self-driving RC car
Wide Angle Raspberry Pi Camera
The eye of the self-driving RC car
16GB Micro SD Card
Storage for the Donkey Car platform and enough to hold training data for autonomous driving
Servo Driver PCA 9685
Controls the throttle and steering of the RC car
DC-DC 5V/2A Voltage Converter
Provides power from the RC car battery to the Raspberry Pi
All other accessories
Wires and screws to build the Donkey Car
In addition, refer to the Donkey Car Assembly video below to follow detailed instructions to assemble all the parts and build your car.
# 4. Software Installation
The installation of the software and setup consists of the following 2 simple steps. In addition, we will be creating our Donkey car application.
Step 1: Install Software on Host PC
When controlling your Donkey via behavioral cloning, you will need to setup a host pc to train your machine learning model from the data collected on the robot. Choose a setup that matches your computer OS. In my case, it was a Mac.
Installation for Mac
mkdir projects cd projects
git clone https://github.com/robocarstore/donkey cd donkey
conda update -n base -c defaults conda conda env remove -n donkey
conda env create -f envs/mac.yml source activate donkey pip install -e .
pip install tensorflow
pip install -e .[pc] donkey createcar --path ~/d2
Step 2: Setup Raspberry Pi
To begin, we need to flash the micro SD card with an operating system.
Download Raspian Lite (300MB)
Download Etcher tool to write an image to the SD Card.
Connect as SD Card reader with the SD card inside it.
Open Etcher and select from your hard drive the Raspberry Pi .img or .zip file you wish to write to the SD card.
Select the SD card you wish to write your image to.
Review your selection and Click “Flash” to begin writing data to SD Card.
Step 3: Setup the Raspberry Pi’s WiFi for first boot.
Open a text editor or notepad and copy paste the following code into it.
country=US
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="<your network name>"
    psk="<your password>"
}
Replace <your network name> with the ID of your network. Leave the quotes. Replace <your password> with your password, leaving it surrounded by quotes. If it bothers you to leave your password unencrypted, you may change the contents later once you’ve gotten the pi to boot and log-in.
Save this file to the root of boot partition with the filename wpa_supplicant.conf. On first boot, this file will be moved to /etc/wpa_supplicant/wpa_supplicant.conf where it may be edited later. If you are using Notepad on Windows, make sure it doesn't have a .txt at the end.
Image for post
Step 4: Raspberry Pi’s Hostname
If yours is the only Pi in the network, you can find it with the following command.
ping d2.local
Step 5: Enable SSH on boot
Put a file named ssh in the root of your boot partition.
Now your SD card is ready. Eject it from your computer, put it in the Pi and plug in the Pi.
Step 6: Connecting to the Pi
Plug in your raspberry pi device to a monitor using HDMI cable and also connect a keyboard to the raspberry pi.
Once the initial boot is done, you will be prompted to enter the login credentials.
Then try the command below. This would show you the IP address of the raspberry pi.
ifconfig wlan0
Once we know the IP address and since we enabled SSH, we can directly remotely connect to the raspberry pi using our mac or computer. Open the Terminal and type in one of the following command options to connect.
#option 1 ssh pi@d2.local #option 2 ssh pi@<ipaddress> #replace <ipaddress> with the one found in the previous step
Type in the following commands in the raspberry pi via ssh to update and upgrade the pi.
sudo apt-get update sudo apt-get upgrade
Step 8: Configure the Raspberry Pi
sudo raspi-config
enable I2c
enable camera
expand filesystem
change hostname
change default password for pi
Image for post
Image for post
Image for post
Image for post
Image for post
Make sure to reboot after making the configuration changes.
Step 9: Install dependencies
sudo apt-get update sudo apt-get install build-essential python3 python3-dev python3-virtualenv python3-numpy python3-picamera python3-rpi.gpio i2c-tools avahi-utils joystick libopenjp2-7-dev libtiff5-dev gfortran libatlas-base-dev libopenblas-dev libhdf5-serial-dev git
# Step 10: Install optional OpenCV dependencies
sudo apt-get install libilmbase-dev libopenexr-dev libgstreamer1.0-dev libjasper-dev libwebp-dev libatlas-base-dev libavcodec-dev libavformat-dev libswscale-dev libqtgui4 libqt4-test
# Step 11: Setup virtual env
python3 -m virtualenv -p python3 env echo "source env/bin/activate" >> ~/.bashrc source ~/.bashrc
Step 12: Install Tensorflow
You can check this page to find the one you like. Or install this one:
wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.10.0/tensorflow-1.10.0-cp35-none-linux_armv7l.whl pip install tensorflow-1.10.0-cp35-none-linux_armv7l.whl
Step 13: Install Donkeycar Python code and install
This will pull the latest from Tawn’s fork:
git clone https://github.com/tawnkramer/donkey pip install -e donkey[pi]
Step 14: Create your car application.
donkey createcar --path ~/d2
# 6. Take it for a spin
Start your car and its time to take it for a spin. if you are planning on using your mobile phone to control your car, you can configure it under the config.py file.
cd ~/d2 sudo nano config.py
change the following line at the bottom of the script.
USE_JOYSTICK_AS_DEFAULT = True
Start your car and drive.
Open your car’s folder and start your car.
cd ~/d2 python manage.py drive
This script will start the drive loop in your car which includes a part that is a web server for you to control your car. You can now control your car from a web browser at the URL: <your car's IP's address>:8887
Image for post
You can load the URL on a mobile browser and control your vehicle remotely or alternatively use the desktop browser to control the device. The car will start recording automatically when you use it to move forward and it will capture the necessary sequence of images and JSON files by creating a tub folder inside the Raspberry Pi’s project data folder.
Driving with Web Controller
Features
Recording — Press record data to start recording images, steering angles and throttle values.
Throttle mode — Option to set the throttle as constant. This is used in races if you have a pilot that will steer but doesn’t control the throttle.
Pilot mode — Choose this if the pilot should control the angle and/or throttle.
Max throttle — Select the maximum throttle.
# 7. Introduction to Google Colab
Now that we are able to drive our car successfully, we need to start training our model so that we can have a self-driving car. In the quest towards the same, I started training it on my Mac and it took hours and I completely gave up on it. Then as an alternate, used my Gaming Rig at home with pretty high specs and it took around 25–30 mins to train the model. This was no good if we wanted to quickly train more models, so in my research wanted to use the Cloud to get the job done quickly but you need to pay for it. Finally came across Google Colab, which requires no setup and runs in the cloud and is completely free.
Colaboratory is a free Jupyter notebook environment that requires no setup and runs entirely in the cloud.
With Colaboratory you can write and execute code, save and share your analyses, and access powerful computing resources, all for free from your browser.

# 8. Train an autopilot using Colab
Using Google Colab’s accelerated hardware, I was able to train our model multiple times within a couple of minutes or less. We are one step away from having our fully functional self driving RC Car.
Image for post
Follow the instructions as shown in the Github repository.
https://github.com/sachindroid8/self-driving-car-using-google-colab
Image for post
Image for post
Image for post
Image for post
Copy the generated mypilot.h5 file to your Raspberry Pi under the following location. /home/pi/d2/models/mypilot.h5
Execute the following command in your Raspberry Pi.
cd ~/d2 python manage.py drive --model ~/d2/models/mypilot
Build a Self-Driving RC Car using Raspberry Pi and Machine Learning using Google Colab
The below image is a demo during one of my talks in Google IO Extended 2019.
Image for post
# 9. Experiment and have fun
Training Tips:
Mode & Pilot: Congratulations on getting it this far. The first thing to note after running the command above is to look at the options in the Mode & Pilot menu. It can be pretty confusing. So here’s what the different options mean:
User: As you guessed, this is where you are in control of both the steering and throttle control.
Local Angle: Not too obvious, but this is where the trained model (mypilot from above) controls the steering. The Local refers to the trained model which is locally hosted on the raspberry-pi.
Local Pilot: This is where the trained model (mypilot) assumes control of both the steering and the throttle. As of now, it’s purportedly not very reliable. Be sure to also check out the Max Throttle and Throttle Mode options, and play around with a few settings. Can help with training quite a lot.
Build a Simple Track: This isn’t very well-documented, but the car should (theoretically) be able to train against any kind of track. To start off with, it might not be necessary to build a two-lane track with a striped center-lane. Try with a single lane with no center-line, or just a single strip that makes a circuit! At the least, you’ll be able to do end-to-end testing and verify that the software pipeline is all properly functional. Of course, as the next step, you’ll want to create a more standard track, and compete at a meetup nearest to you!
Get help: Try to get some helping hands from a friend or two. Again, this helps immensely with building the track, because it is harder than it looks to build a two-line track on your own! Also, you can save on resources (and tapes) by using a ribbon instead of tapes. They’ll still need a bit of tape to hold them, but you can reuse them and they can be laid down with a lot less effort (Although the wind, if you’re working outside, might make it difficult to lay them down initially).
# 10. Conclusion
To conclude, we have successfully built a self-driving car using Raspberry Pi and Machine Learning models using Tensorflow and Google Colab. We can train more models using the same method and train faster using Google Colab for free. Also, check out some of my other tutorials.
How to build your own smart speaker — Google Assistant, Google Cloud, Actions on Google and…
In this tutorial, we will learn how to build your own smart speaker using Google Assistant, Google Cloud, Actions on…
medium.com
How to create a chatbot using Dialogflow Enterprise Edition and Dialogflow API V2
In this tutorial, we will learn how to create a chatbot using Dialogflow Enterprise Edition and Dialogflow API V2. This…
medium.com
How to build an App for Google Assistant using Dialogflow Enterprise Edition and Actions on Google
Users engage Google Assistant in conversation to get things done, like buying groceries or booking a ride or in our…
medium.com
# 11. References
Google Colab
Robocarstore Docs
DonkeyCar Docs
