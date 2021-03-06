import numpy as np
import cv2
from PyQt5 import QtGui

class Mask(object):

    def __init__(self,cam_type, shape):

        try:
            self.cam_type = cam_type
            self.mask = None
            #size = 1944, 2592, 3
            empty_img = np.zeros(shape, dtype=np.uint8)
            if self.cam_type is 1:
                self.mask = self.cmask([790, 1190], 1000, empty_img)   # 20181012_1: [790, 1190], 1000,
            if self.cam_type is 2:
                self.mask = self.cmask([985, 1340], 1089, empty_img)  # 20181022_2:  [985, 1340], 1089, ; [1093, 1299],1080

        except Exception as e:
            print("init Mask: ", str(e))

    def cmask(self, index, radius, array):
        """Generates the mask for a given input image.
        The generated mask is needed to remove occlusions during post-processing steps.

        Args:
            index (numpy array): Array containing the x- and y- co-ordinate of the center of the circular mask.
            radius (float): Radius of the circular mask.
            array (numpy array): Input sky/cloud image for which the mask is generated.

        Returns:
            numpy array: Generated mask image."""
        try:

            a, b = index
            is_rgb = len(array.shape)

            if is_rgb == 3:
                ash = array.shape
                nx = ash[0]
                ny = ash[1]
            else:
                nx, ny = array.shape

            s = (nx, ny)
            image_mask = np.zeros(s)
            y, x = np.ogrid[-a:nx - a, -b:ny - b]
            mask = x * x + y * y <= radius * radius
            image_mask[mask] = 1

            return (image_mask)

        except Exception as e:
            print("cmask: ", str(e))

    def maske_OpenCV_Image(self, input_image):

        try :
            red   = input_image[:, :, 0]
            green = input_image[:, :, 1]
            blue  = input_image[:, :, 2]

            # Show mask in red
            if False:
                h = input_image.shape[0]
                w = input_image.shape[1]

                for y in range(0, h):
                    for x in range(0, w):
                        if self.mask[y, x] == 0:
                            red[y, x] = 225
                r_img = red
            else:
                r_img = red.astype(float) * self.mask
            g_img = green.astype(float) * self.mask
            b_img = blue.astype(float) * self.mask

            dimension = (input_image.shape[0], input_image.shape[1], 3)
            output_img = np.zeros(dimension, dtype=np.uint8)

            output_img[..., 0] = r_img[:, :]
            output_img[..., 1] = g_img[:, :]
            output_img[..., 2] = b_img[:, :]

            return output_img

        except Exception as e:
            print("cmask: ", str(e))

    def get_mask_as_one_channel_img(self):
        # https://stackoverflow.com/questions/40119743/convert-a-grayscale-image-to-a-3-channel-image
        # for 3 channels np.stack((self.mask,) * 3, -1)
        try:
            mask = np.stack((self.mask,) * 1, -1)
            return mask.astype(np.uint8)

        except Exception as e:
            print("get_mask_as_one_channel_img: ", str(e))

class Segmentation(object):

    def __init__(self):
        # Create image mask
        size = 1944, 2592, 3
        empty_img = np.zeros(size, dtype=np.uint8)
        self.mask = self.cmask([880, 1190], 1117, empty_img)

    def ycbcr_substractor(self, image):

        try:
            #imageYCbCr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            imageYCbCr = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB) # sollte doch COLOR_RGB2YCR_CB sein ?
            imageYCbCr = imageYCbCr.astype(np.double)

            Y  = imageYCbCr[..., 0]
            Cb = imageYCbCr[..., 1]
            Cr = imageYCbCr[..., 2]

            TrCol = [150, 117]

            LimitVal = 9

            red   = image[..., 0]
            green = image[..., 1]
            blue  = image[..., 2]

            index_Img = ((Cb - TrCol[0]) ** 2 + (Cr - TrCol[1]) ** 2) < LimitVal ** 2

            red[index_Img]   = 255
            green[index_Img] = 0
            blue[index_Img]  = 0

            dimension = (image.shape[0], image.shape[1], 3)
            subtracted_img = np.zeros(dimension, dtype=np.uint8)

            subtracted_img[..., 0] = red[:,:]
            subtracted_img[..., 1] = green[:,:]
            subtracted_img[..., 2] = blue[:,:]

            masked_img = self.maske_OpenCV_Image(subtracted_img)

            return masked_img

        except Exception as e:
            print('Backgroundsubstraction: {}'.format(e))

    def cv2qpixmap(self,imgage):
        height, width, channel = imgage.shape
        qt_img = QtGui.QImage(imgage, width, height, QtGui.QImage.Format_RGB888)
        qpixmap_img = QtGui.QPixmap.fromImage(qt_img)
        return qpixmap_img

    def cmask(self,index, radius, array):
        """Generates the mask for a given input image.
        The generated mask is needed to remove occlusions during post-processing steps.

        Args:
            index (numpy array): Array containing the x- and y- co-ordinate of the center of the circular mask.
            radius (float): Radius of the circular mask.
            array (numpy array): Input sky/cloud image for which the mask is generated.

        Returns:
            numpy array: Generated mask image."""

        a, b = index
        is_rgb = len(array.shape)

        if is_rgb == 3:
            ash = array.shape
            nx = ash[0]
            ny = ash[1]
        else:
            nx, ny = array.shape

        s = (nx, ny)
        image_mask = np.zeros(s)
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x * x + y * y <= radius * radius
        image_mask[mask] = 1

        return (image_mask)

    def maske_OpenCV_Image(self, input_image):

        red   = input_image[:, :, 0]
        green = input_image[:, :, 1]
        blue  = input_image[:, :, 2]

        r_img = red.astype(float)   * self.mask
        g_img = green.astype(float) * self.mask
        b_img = blue.astype(float) * self.mask

        dimension = (input_image.shape[0], input_image.shape[1], 3)
        output_img = np.zeros(dimension, dtype=np.uint8)

        output_img[..., 0] = r_img[:, :]
        output_img[..., 1] = g_img[:, :]
        output_img[..., 2] = b_img[:, :]

        return output_img

    def getBR_Ratio_Image(self, input_image):
        """Extracts the ratio of red and blue blue channel from an input sky/cloud image.
           It is used in the clustering step to generate the binary sky/cloud image.

           Args:
               input_image (numpy array): Input sky/cloud image.
               mask_image (numpy array): Mask to remove occlusions from the input image.
               This mask contains boolean values indicating the allowable pixels from an image.

           Returns:
               numpy array: Ratio image using red and blue color channels, normalized to [0,255]."""

        red   = input_image[:, :, 2]
        green = input_image[:, :, 1]
        blue  = input_image[:, :, 0]

        r_img = red.astype(float)   * self.mask
        g_img = green.astype(float) * self.mask
        b_img = blue.astype(float) * self.mask

        BR = (b_img - r_img) / (b_img + r_img)
        BR[np.isnan(BR)] = 0

        normalized_img = (BR - np.amin(BR)) / (np.amax(BR) - np.amin(BR)) * 255

        dimension = (input_image.shape[0], input_image.shape[1], 3)
        output_img = np.zeros(dimension, dtype=np.uint8)

        output_img[..., 0] = normalized_img[:, :]
        output_img[..., 1] = normalized_img[:, :]
        output_img[..., 2] = normalized_img[:, :]


        #print('normalized_BR shape : {}'.format(normalized_img.shape))

        return self.cv2qpixmap(output_img)

    def findContour(self, image, TrCol = [150, 117], minArea = 300.0):

        imageYCbCr = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        imageYCbCr = imageYCbCr.astype(np.double)

        Y = imageYCbCr[..., 0]
        Cb = imageYCbCr[..., 1]
        Cr = imageYCbCr[..., 2]

       # TrCol = [140, 120]

        LimitVal = 9

        index_Img = ((Cb - TrCol[0]) ** 2 + (Cr - TrCol[1]) ** 2) < LimitVal ** 2

        imgray = np.array(index_Img, dtype=np.uint8)


        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        closing = cv2.morphologyEx(imgray, cv2.MORPH_CLOSE, kernel)

        erode = cv2.morphologyEx(closing, cv2.MORPH_ERODE, kernel)

        im2, contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        cleaned_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (area > minArea):
                cleaned_contours.append(cnt)

        img_cleand_contours = cv2.drawContours(image, cleaned_contours, -1, (0, 0, 0), 3)
        img_contour_masked = self.maske_OpenCV_Image(img_cleand_contours)

        return img_contour_masked

class Optical_Flow(object):

    def __init__(self):
        # Create image mask
        size = 1944, 2592, 3
        empty_img = np.zeros(size, dtype=np.uint8)
        self.mask = self.cmask([880, 1190], 1117, empty_img)

    def cmask(self,index, radius, array):
        """Generates the mask for a given input image.
        The generated mask is needed to remove occlusions during post-processing steps.

        Args:
            index (numpy array): Array containing the x- and y- co-ordinate of the center of the circular mask.
            radius (float): Radius of the circular mask.
            array (numpy array): Input sky/cloud image for which the mask is generated.

        Returns:
            numpy array: Generated mask image."""

        a, b = index
        is_rgb = len(array.shape)

        if is_rgb == 3:
            ash = array.shape
            nx = ash[0]
            ny = ash[1]
        else:
            nx, ny = array.shape

        s = (nx, ny)
        image_mask = np.zeros(s)
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x * x + y * y <= radius * radius
        image_mask[mask] = 1

        return (image_mask)

    def maske_OpenCV_Image(self, input_image):

        red   = input_image[:, :, 0]
        green = input_image[:, :, 1]
        blue  = input_image[:, :, 2]

        r_img = red.astype(float)   * self.mask
        g_img = green.astype(float) * self.mask
        b_img = blue.astype(float) * self.mask

        dimension = (input_image.shape[0], input_image.shape[1], 3)
        output_img = np.zeros(dimension, dtype=np.uint8)

        output_img[..., 0] = r_img[:, :]
        output_img[..., 1] = g_img[:, :]
        output_img[..., 2] = b_img[:, :]

        return output_img

    def draw_flow(self, img, flow, step=16):
        try:
            h, w = img.shape[:2]
            y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
            fx, fy = flow[y.astype(np.int64), x.astype(np.int64)].T
            lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
            lines = np.int32(lines + 0.5)

            cv2.polylines(img, lines, 0, (0, 135, 0),thickness=3)
            '''
            for (x1, y1), (x2, y2) in lines:
                cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)
            '''
            return img

        except Exception as e:
            print('draw_flow : {}'.format(str(e)))

    def opt_flow(self, prev, next):
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(next, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        imgWithFlow = self.draw_flow(prev,flow)

        masked_img = self.maske_OpenCV_Image(imgWithFlow)
        return masked_img



