# Do silhouettes dream?
<p>Code for reproducing our interactive art installation ["Do silhouettes dream?"](http://www.skamalas.com/#silhouettes). </p>

<p>It was exhibited at the [ArtScience Museum](http://www.marinabaysands.com/museum.html) in Singapore as part of the [Microbites of Innovation Showcase](http://microbites.me/) in conjuction with the [ACM Creativity & Cognition 2017](http://cc.acm.org/2017/) conference.</p>

## Requirements
* python 2.7 with numpy, scipy and matplotlib packages
* [kinect v2 drivers](https://github.com/OpenKinect/libfreenect2/)
* [primesense](https://pypi.python.org/pypi/primesense) package

### MacOS Sierra tested recipe:
* follow [the instructions here](http://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/) to install Python and OpenCV properly via homebrew
* install the kinect drivers following [these instructions](https://github.com/OpenKinect/libfreenect2/blob/master/README.md#mac-osx)
* install the [primesense](https://pypi.python.org/pypi/primesense) package

## Inputs

### Deep Dreaming
We provide a modified version of the tensorflow deepdream script that outputs gifs and the proper numpy input tensor needed by the main script. See folder:
```
deepdream/
```

### Video input
Day-to-night timelapse videos are ideal for this. We provide an iPython Notebook script that gets a video and creates the proper tensor needed from the main script:

```
video2input.ipynb
```

it requires the youtube_dl package that you can install with:

```
sudo pip install --upgrade youtube_dl
```


## Running the demo

In general, to run the demo you need to first connect the kinect to your laptop and then run the main script:
```
python silhouettes.py
```
While **having the python window selected**, you can exit the demo at any time by pressing the ESC key.

Before running the demo for the first time, you will need to specify the input and output parameters and perhaps adjust the min-max depth values for each specific location.

### Adjusting input and output parameters
Open silhouettes.py with your favorite editor and adjust the following (self-explanatory) parameters:

```
# input/output parameters

# input tensor
input_fn = 'tensors/black4.pkl'

# output window size
FINAL_OUTPUT_WIDTH = 1440
FINAL_OUTPUT_HEIGHT = 900
```

### Adjusting min/max depth and debuging kinect input

You may wat to adjust the min-max depth values for each specific location you run the demo. These parameters would be very different if eg you want to run this in a big warehouse vs your cluttered bedroom. In general the depth values for kinect v2 are between 500 and 4500.

To debug depth, first turn set parameter:
``` 
show_depth = True
```
This will only display the depth map (the input from the kinect). Then, while **having the python window selected/in focus**, you can change the min/max depth values using some keyboard shortcuts:
* keys "a" and "z" control the minimun depth cliping 
* keys "s" and "x" control the maximum depth cliping 
There are debug messages shown on the terminal, to let you pick the proper values that work for your space.
After you decide, you set the following two variables to the selected depths. Eg parameters:
```
MIN_DEPTH = 500.0
MAX_DEPTH = 2000.0 
```
will clip faraway objects and work better for smaller spaces.

**Don't forget to turn the debug depth parameter off** to start running the demo:
``` 
show_depth = False
```
