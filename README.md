# Bad Meme Cropper
Crops memes badly, but tries its best.

# Install
It's easiest to use docker, because OpenCV is hard to install everywhere, always.
1. Install Docker
2. clone this git repo

# Run
1. cd to this directory, then run: `docker build . -t bad-meme-cropper && docker run -v [full_path_to_input_image_folder]:/if/ -v [full_path_to_output_image_folder]:/of/ --rm -it bad-meme-cropper /if/ /of/ .1 10`
2. enjoy
