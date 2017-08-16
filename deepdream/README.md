
### Install dependencies

Install the dependencies in `requirements.txt`. You can substitute the CPU-only tensorflow if you are not on a GPU machine.

### Get Inception model

Download the inception model graph to your working directory:

```
wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip && unzip inception5h.zip
```

### Running

The `timedepth.py` script takes an image as input and allows configuration of various outputs and dreep dream parameters. The script will generate a given number of frames as output. This output can be stored as separate images in a directory, combined into a gif, and stored as a Pickle file containing the raw image data as a NumPy array which can be used with the interactive projection scripts. Refer to the commandline arguments in the script for details.

#### Example Usage

The output can be controlled by passing the name of a "parameter function" to the script with the `-e` flag. This is the name of a function defined in `timedepth.py` to be invoked to generate the parameters for each frame of the deep dream series.

A typical usage is:
```
python timedream.py \
    --img <input_file> \
    -g <gif_output> \
    -o <frame_output_dir> \
    -n <num_frames> \
    -d <param_func> \
    -e <param_config_string>
```

Below are some sample commands that tend to produce interesting results.

Setup output directories for gif files and Pickle files:
```
mkdir results_gif
mkdir results
```

For a fast gif (10 frames) through a number of laters with deep iteration in each:
```
python timedream.py -g results_gif/ -o results/ -n 10 -d layers_fast_dream \
    -e mixed3a,mixed3a,mixed3b,mixed4a,mixed4b,mixed4c,mixed4d,mixed4e,mixed5a,mixed5b \
    --img <input_file>
```

For a smooth gif (128 frames) that only optimizes a single layer slowly:
```
python timedream.py -g results_gif/ -o results/ -n 128 -d iteration_layers_dream \
    -e mixed5a_5x5_pre_relu \
    --img <input_file>
```

For a smooth gif that smoothly transitions from a deep layer to a shallower layer:
```
python timedream.py -g results_gif/ -o results/ -n 128 -d iteration_layers_dream \
    -e mixed5a_5x5_pre_relu,mixed3a_3x3_pre_relu \
    --img <input_file>
```
