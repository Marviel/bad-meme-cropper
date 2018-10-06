# Bad Meme Cropper
Crops memes badly, but tries its best.

# Install
It's easiest to use docker, because OpenCV is hard to install everywhere, always.
1. Install Docker
2. clone this git repo

# Run
1. cd to this directory
1. `docker build . -t bad-meme-cropper && docker run -v `pwd`/images:/input_folder/ -v `pwd`/output:/output_folder/ --rm -it bad-meme-cropper /input_folder/ /output_folder/ .1 10`
