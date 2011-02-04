import hadoopy
import Image
import numpy as np
import cStringIO as StringIO
import cv
import base64

"""
An introductory example of hadoopy using OpenCV face detection.
https://code.ros.org/svn/opencv/trunk/opencv/samples/python/facedetect.py
"""

# Parameters for haar detection
# From the API:
# The default parameters (scale_factor=2, min_neighbors=3, flags=0) are tuned
# for accurate yet slow object detection. For a faster operation on real video
# images the settings are:
# min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING,
# min_size=<minimum possible face size

min_size = (20, 20)
haar_scale = 1.2
min_neighbors = 2
haar_flags = 0

_desired_size = 320*240   # A desired image size in pixels
_cascade = cv.Load("haarcascade_frontalface_alt.xml")
_face_size = 64


def mapper(key, value):
    """
    Args:
        key: Unused
        value: JPEG Image Data

    Yields:
        Tuple of (key, value) where
        key: Unused
        value: (face, (x, y)) where face is a numpy array for a detected
            face patch  (_face_size, _face_size) and x,y is the normalized
            location of the face center in the image (between 0 and 1)
    """
    try:
        image = Image.open(StringIO.StringIO(value))
    except (IOError):
        return

    # Resize the input image, converting to grayscale
    width, height = image.size
    factor = float(_desired_size) / (width*height)
    grey = np.asarray(image.convert('L').resize((int(width*factor),
                                                  int(height*factor))))

    # Detect faces in the grayscale image
    faces = cv.HaarDetectObjects(grey, _cascade, cv.CreateMemStorage(0),
                        haar_scale, min_neighbors, haar_flags, min_size)

    # Output each detected face
    for ((x, y, w, h), _) in faces:
        # Extract the face patch, and resize it to _face_size
        face = grey[y:y+h, x:x+h]
        resized = np.empty((_face_size, _face_size), 'u1')
        cv.Resize(face, resized)

        # Find the normalized coordinates
        nx, ny = float(x + w/2)/width, float(y + h/2)/height
        yield base64.b64encode(key),(resized, (nx, ny))


def combiner(key, values):
    """
    TODO explain here, right now combiner does nothing interesting
    Args:

    Yields:

    """
    # TODO combines some of the mean faces
    for v in values:
        yield key, v


class Reducer(object):

    def __init__(self):
        self.sum_face = np.zeros((_face_size, _face_size))
        self.sum_x = 0
        self.sum_y = 0
        self.face_counter = 0

    def _image_to_str(self, img):
        out = StringIO.StringIO()
        img.save(out, 'JPEG')
        out.seek(0)
        return out.read()

    def reduce(self, key, values):
        """
        TODO right now the mapper only computes the mean_face but we should
        make it do something more interesting
        Args:

        Yields:

        """
        for (face, (x, y)) in values:
            self.sum_x += x
            self.sum_y += y
            self.sum_face += face
            self.face_counter += 1
            filename = 'faces/%s.jpg' % str(key)
            yield filename, self._image_to_str(Image.fromarray(face))

        mean_face = (self.sum_face/self.face_counter).astype('u1')
        yield 'mean_face.jpg', self._image_to_str(Image.fromarray(mean_face))

if __name__ == '__main__':
    hadoopy.run(mapper, Reducer, combiner)
