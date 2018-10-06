# @Author: Bechtel, Luke <Marviel>
# @Date:   2018-10-06T14:20:17-04:00
# @Email:  lukebechtel4@gmail.com
# @Filename: main.py
# @Last modified by:   lukebechtel
# @Last modified time: 2018-10-06T18:39:34-04:00
# @License: MIT
# @Copyright: You are given explicit rights to use this software in any fashion, so long as it is not used to operate nuclear facilities, life support or other mission critical applications where human life or property may be at stake.

import argparse
import arghelper
import numpy as np
import cv2
import os
import sys
import glob
np.random.seed(1)

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end


def text_detect(img, ele_size=(8, 3)):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_sobel = cv2.Sobel(img, cv2.CV_8U, 1, 0)
    img_threshold = cv2.threshold(
        img_sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, ele_size)
    img_threshold = cv2.morphologyEx(
        img_threshold[1], cv2.MORPH_CLOSE, element)
    contours = cv2.findContours(img_threshold, 0, 1)
    Rect = [cv2.boundingRect(i) for i in contours[1] if i.shape[0] > 100]
    RectP = [(int(i[0] - i[2] * 0.08), int(i[1] - i[3] * 0.08),
              int(i[0] + i[2] * 1.1), int(i[1] + i[3] * 1.1)) for i in Rect]
    return RectP


def main():
    pass


def rectangle_contains_set(r1, r2):
    """
    Determines if r1 contains r2.
    """
    np.all(r1[0, :])


def rectangle_contains_rectangles(container, rects):
    return np.logical_and(
        np.all(container[0] <= rects[:, 0], axis=1),
        np.all(container[1] >= rects[:, 1], axis=1)
    )

def determine_coverage(container, rects):
    """
    Get number of children which are contained in parent.
    """
    contained = rectangle_contains_rectangles(container, rects)

    return np.sum(contained)


def clip_rect(rect, clip_rect):
    rect[:, 0] = np.clip(rect[:, 0], clip_rect[0][0], clip_rect[1][0])
    rect[:, 1] = np.clip(rect[:, 1], clip_rect[0][1], clip_rect[1][1])

def determine_image_crop_by_text_coverage(img, rects, desired_percentage,
                                          expansion_step=10, min_thickness=20):
    desired_percentage = np.clip(desired_percentage, 0, 1)
    img_shape_xy = [img.shape[1], img.shape[0]]
    dest_rect = np.array([[0, 0], [img_shape_xy[0], img_shape_xy[1]]])

    # Create a random rectangle in the space of the image.
    start_x = np.random.randint(0, img_shape_xy[0]-min_thickness)
    start_y = np.random.randint(0, img_shape_xy[1]-min_thickness)
    rect = np.array([[start_x,
                      start_y],
                     [start_x + np.random.randint(min_thickness, img_shape_xy[0]),
                      start_y + np.random.randint(min_thickness, img_shape_xy[1])]])

    # Clip the rect so it's within our bounds.
    clip_rect(rect, dest_rect)

    covered_percentage = determine_coverage(rect, rects) * 1.0 / len(rects)


    # While we haven't covered the percentage of rectangles we want,
    # Randomly choose to expand one piece of the rectangle by expansion_step
    while covered_percentage < desired_percentage:
        # Expand randomly. Don't be choosy right now.
        # Future optimization could filter by which directions are expandable.
        expand_dims = np.random.randint(0, 2, size=(2,))

        rect[expand_dims[0]][expand_dims[1]] += (1 if expand_dims[0] else -1) * expansion_step

        # Clip so it's not outside of the image.
        clip_rect(rect, dest_rect)

        # Calculate coverage percentage
        covered_percentage = determine_coverage(rect, rects) * 1.0 / len(rects)


    # In the future, this could be improved speed wise by only
    # Expanding to include the next textual region seen in the expansion
    # direction.
    return rect

def main():
    parser = argparse.ArgumentParser(
        description='Convert a three.js json file to .stl.')
    parser.add_argument('in_folderpath',
                        type=arghelper.extant_dir,
                        help='the input directory from whence the files come.')
    parser.add_argument('out_folderpath',
                        type=arghelper.extant_dir,
                        help='the output directory in which to store the files.')
    parser.add_argument('min_coverage',
                        type=float,
                        choices=[Range(0.0, 1.0)],
                        help='The minimum percentage (0.0-1.0) of text to include before cropping.')
    parser.add_argument('iterations',
                        type=int,
                        help='the output directory in which to store the files.')
    args = parser.parse_args()

    in_folderpath = args.in_folderpath
    out_folderpath = args.out_folderpath
    debug = True
    accepted_types = ['.png', '.jpg', '.jpeg']
    min_coverage = args.min_coverage
    iterations = args.iterations

    # If output folder does not exist, try to create it now, so we fail early.
    if not os.path.exists(out_folderpath):
        os.makedirs(out_folderpath)

    # Get all files in in_folderpath tree that match the types we care about :)
    input_paths = sum(
        map(
            lambda t: list(glob.iglob('%s/**/*%s' % (in_folderpath, t),
                                      recursive=True)),
            accepted_types),
        [])

    input_paths = map(os.path.abspath, input_paths)

    # Process everything in input_paths.
    # Do this *as* you write the files, so you don't keep too much in memory.
    for input_path in input_paths:
        print("processing %s" % (input_path))
        try:
            basename = os.path.basename(input_path)
            img = cv2.imread(input_path)
            rects = np.array(text_detect(img))
            rects = rects.reshape((len(rects), 2, 2))

            if debug:
                text_img = img.copy()
                for rect in rects:
                    cv2.rectangle(text_img, tuple(rect[0]), tuple(rect[1]), (255, 255, 0), 2)
                    cv2.imwrite('debug/%s__text-detected.png' % (basename), text_img)

            for i in range(iterations):
                crop_rect = determine_image_crop_by_text_coverage(img, rects, min_coverage)
                new_img = img[crop_rect[0][1]:crop_rect[1][1],
                              crop_rect[0][0]:crop_rect[1][0]]
                cv2.imwrite(os.path.join(out_folderpath,
                                         '%s__out-%d.png' % (basename, i)),
                            new_img)

        except Exception as e:
            print("could not process file %s" % (input_path))
            print("Exception was: %s" % (e))


if __name__ == "__main__":
    main()
