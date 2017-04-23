# Do silhouettes dream of time?
Code for reproducing our interactive art installation "Do silhouettes dream of time?" that will be exhibited at the [ArtScience Museum](http://www.marinabaysands.com/museum.html) in Singapore as part of the [Microbites of Innovation Showcase](http://cc.acm.org/2017/calls/artworks.php) in conjuction with the [ACM Creativity & Cognition 2017](http://cc.acm.org/2017/) conference.

*Python scripts will be available here soon!*

## Requirements:
* python 2.7 with numpy, scipy and matplotlib packages
* kinect drivers installed
* [primesense](https://pypi.python.org/pypi/primesense) package


## Inputs

### Deep Dreaming
We will provide a modified version of the tensorflow deepdream script to create the proper imnput tensor needed by the scripts.

### Video input

Timelapse videos are ideal for this, especially day-to-night ones. We will provide a script that gets a video and creates the proper tensor for the scripts
