import hadoopy
import Image
import numpy as np
import cStringIO as StringIO
import cv

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
_bin_count = 10  # The number of (normalized) bins in each dimension


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

        # Find the binned coordinates
        nx = (x + w/2)*_bin_count/grey.shape[1]
        ny = (y + h/2)*_bin_count/grey.shape[0]
        
        yield (nx,ny), resized


def reducer(key, values):
    """
    The mapper computes the face for each bin
    Args:

    Yields:

    """
    def _image_to_str(img):
        out = StringIO.StringIO()
        img.save(out, 'JPEG')
        out.seek(0)
        return out.read()

    x, y = key

    sum_face = np.zeros((_face_size, _face_size))
    face_counter = 0

    for (face) in values:
        sum_face += face
        face_counter += 1

    mean_face = (sum_face/face_counter).astype('u1')
    yield ('mean_%d_%d.jpg' % (x, y),
           _image_to_str(Image.fromarray(mean_face)))
    yield (x, y), face_counter


if __name__ == '__main__':
    hadoopy.run(mapper, reducer)
