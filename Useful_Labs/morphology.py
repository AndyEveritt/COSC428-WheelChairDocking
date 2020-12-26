# morphology.py

import cv2
import numpy as np

def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass

def erosion():
    """
	 Shrinks object. If ALL pixels in kernel are 1 then pixel remains 1, otherwise its 0
	 """
    cv2.namedWindow('Erosion Example')
    cv2.createTrackbar('Gaussian Blur', 'Erosion Example', 1, 100, nothing)
    cv2.createTrackbar('Iterations', 'Erosion Example', 1, 10, nothing)

    img = cv2.imread('../images/chips.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    while True:
        # Note that the value in gaussian_kernel is rounded up to the next odd number.
        gaussian_kernel = cv2.getTrackbarPos('Gaussian Blur', 'Erosion Example') // 2 * 2 + 1
        morph_iterations = cv2.getTrackbarPos('Iterations', 'Erosion Example')

        # Gaussian blur to reduce noise in the image.
        gray_blur = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)
        # Use adaptive thresholding to "binarize" the image.
        thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)

        # Perform some morphological operations to help distinguish some of the features in the image.
        kernel = np.ones((3,3), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=morph_iterations)

        cv2.imshow('Erosion Example', erosion)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows

def dilation():
    """
	 Expands object. If ANY pixels in kernel are 1 then pixel remains 1, otherwise its 0
	 """
    cv2.namedWindow('Dilation Example')
    cv2.createTrackbar('Gaussian Blur', 'Dilation Example', 1, 100, nothing)
    cv2.createTrackbar('Iterations', 'Dilation Example', 1, 10, nothing)

    img = cv2.imread('../images/chips.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    while True:
        # Note that the value in gaussian_kernel is rounded up to the next odd number.
        gaussian_kernel = cv2.getTrackbarPos('Gaussian Blur', 'Dilation Example') // 2 * 2 + 1
        morph_iterations = cv2.getTrackbarPos('Iterations', 'Dilation Example')

        # Gaussian blur to reduce noise in the image.
        gray_blur = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)
        # Use adaptive thresholding to "binarize" the image.
        thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 1)

        # Perform some morphological operations to help distinguish some of the features in the image.
        kernel = np.ones((3,3), np.uint8)
        dilation = cv2.dilate(thresh, kernel, iterations=morph_iterations)

        cv2.imshow('Dilation Example', dilation)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows

def closing():
    """
	 Fills gaps and holes. Dilation followed by Erosion
	 """
    cv2.namedWindow('Closing Example')
    cv2.createTrackbar('Gaussian Blur', 'Closing Example', 1, 100, nothing)
    cv2.createTrackbar('Iterations', 'Closing Example', 1, 10, nothing)

    img = cv2.imread('../images/chips.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    while True:
        # Note that the value in gaussian_kernel is rounded up to the next odd number.
        gaussian_kernel = cv2.getTrackbarPos('Gaussian Blur', 'Closing Example') // 2 * 2 + 1
        morph_iterations = cv2.getTrackbarPos('Iterations', 'Closing Example')

        # Gaussian blur to reduce noise in the image.
        gray_blur = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)
        # Use adaptive thresholding to "binarize" the image.
        thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 1)

        # Perform some morphological operations to help distinguish some of the features in the image.
        kernel = np.ones((3,3), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

        cv2.imshow('Closing Example', closing)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows

def opening():
    """
	 Smoothes, removes spurs, breaks narrow lines. Erosion followed by Dilation.
	 """
    cv2.namedWindow('Opening Example')
    cv2.createTrackbar('Gaussian Blur', 'Opening Example', 1, 100, nothing)
    cv2.createTrackbar('Iterations', 'Opening Example', 1, 10, nothing)

    img = cv2.imread('../images/chips.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    while True:
        # Note that the value in gaussian_kernel is rounded up to the next odd number.
        gaussian_kernel = cv2.getTrackbarPos('Gaussian Blur', 'Opening Example') // 2 * 2 + 1
        morph_iterations = cv2.getTrackbarPos('Iterations', 'Opening Example')

        # Gaussian blur to reduce noise in the image.
        gray_blur = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)
        # Use adaptive thresholding to "binarize" the image.
        thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 1)

        # Perform some morphological operations to help distinguish some of the features in the image.
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)

        cv2.imshow('Opening Example', opening)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows

if __name__ == "__main__":
#     erosion()
#     dilation()
#     closing()
     opening()
