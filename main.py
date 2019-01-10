from __future__ import division

import cv2
import numpy as np
import sqlite3
from glob import glob
import os
import re
import sys
import math
from os.path import join
from copy import deepcopy
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog

from matplotlib import pyplot as plt
import pyqtgraph as pg

import ImageProcessingLibrary
from dialog import Ui_dialog

TIMER_INTERVAL = 100
SLIDESHOW_STEP = 1        # 0 if used with lists but 1 if with sqlite
SLIDESHOW_INTERVAL = 50   # milliseconds
MAX_LOAD_IMAGES = 200     # max number of images loaded from source dir

######################################################################
## Hoa: 10.01.2019 Version 5 : Image Analysis
######################################################################
# Collects all images hdr and ldr, generates color mapped images
# and histogram and shows them as slide show.
# Using SQLite database to hold all loaded images.
#
# Images of one day have to be in own folder. Each image folder must
# contain a 'temp' folder. Each folder of a day must contain a
# 'output' folder with the preprocessed hdr images.
#
# Images from camera 1 respective camera 2 have to be contained within
# a folder named 'camera_1' respective 'camera_2'.
# Path Example: C:image_analysis\camera_1\20181012_raw_cam1\temp\20181012_090032
#
# New /Changes:
# ----------------------------------------------------------------------
# Remarks: additional libraries:
#          - pyqtgraph
#
# Resolve the HDF5 error by:
#           - conda uninstall hdf5
#           - conda install hdf5
# ----------------------------------------------------------------------
# 18.05.2018 : first implemented
# 21.05.2018 : using SQLite database
# 21.05.2018 : Alle images shown are preprocesed and saved in db
# 10.01.2019 : Loads now images from a SQLite database (still minor errors!)
#
#
######################################################################

class MyForm(QMainWindow):

    def __init__(self):
        super().__init__()
        self.ready = False
        self.ui = Ui_dialog()
        self.ui.setupUi(self)
        self.threadpool = QThreadPool()
        self.show()

        ###########################################################
        # Use QSettings to save states
        ###########################################################

        self.settings = QSettings('__settings.ini', QSettings.IniFormat)
        self.settings.setFallbacksEnabled(False)

        ###########################################################
        # Connect Signal
        ###########################################################
        self.ui.lbl_JPG_img.installEventFilter(self)
        self.ui.lbl_JPG_img.setMouseTracking(True)

        ####################################################################
        # Buttons Signals
        ####################################################################
        self.ui.pushButton_Step_Left.clicked.connect(self.backward_oneImage)
        self.ui.pushButton_Step_Right.clicked.connect(self.forward_oneImage)
        self.ui.pushButton_browse.clicked.connect(self.getOutputFolderName)
        self.ui.pushButton_RunImgSeq.clicked.connect(self.startStopSlideShow)
        self.ui.pushButton_close.clicked.connect(QCoreApplication.instance().quit)
        self.ui.pushButton_close.clicked.connect(self.close)

        ###########################################################
        # ListView Signal
        ###########################################################
        self.ui.listWidget_cam1.currentItemChanged.connect(self.on_item_changed_cam1)
        self.ui.listWidget_cam2.currentItemChanged.connect(self.on_item_changed_cam2)

        self.ui.listWidget_cam1.doubleClicked.connect(self.on_item_doubleclicked_cam1)
        self.ui.listWidget_cam2.doubleClicked.connect(self.on_item_doubleclicked_cam2)

        ###########################################################
        # Labels
        ###########################################################
        self.ui.lbl_info.setText('Info')

        ###########################################################
        # QLabel - containers for images and plots
        ###########################################################
        # Set scaled properties
        self.ui.lbl_JPG_img.setScaledContents(True)
        self.ui.lbl_HDR_img.setScaledContents(True)
        self.ui.lbl_JPG_COLMAP.setScaledContents(True)
        self.ui.lbl_HDR_COLMAP.setScaledContents(True)

        self.ui.lbl_JPG_RGB_hist.setScaledContents(True)
        self.ui.lbl_HDR_RGB_hist.setScaledContents(True)

        ###########################################################
        # Timer update slider values
        ###########################################################
        self.timer_update = QTimer()
        self.timer_update.start(TIMER_INTERVAL)

        ###########################################################
        # Timer for image slide show
        ###########################################################
        self.timer_slideshow = QTimer()
        self.timer_slideshow.timeout.connect(self.runSlideShow)
        # self.timer_slideshow.setInterval(SLIDESHOW_DELAY)
        self.slideshow_step = SLIDESHOW_STEP

        ###########################################################
        #  Lists with all camera 1/2 directories
        ###########################################################
        self.cam1_dirs_path_list = []
        self.cam2_dirs_path_list = []

        ###########################################################
        #  Lists concerning one directory of JPG and HDR images
        ###########################################################
        self.jpg_img_path_list = []
        self.hdr_img_path_list = []

        self.openCVImg_JPG_img_list = []
        self.openCVImg_HDR_img_list = []

        self.pixMapImg_JPG_img_list = []  # shown in first upper label
        self.pixMapImg_HDR_img_list = []  # shown in first lower label

        ###########################################################
        #  Plot Widgets used with pygraph
        ###########################################################
        self.pw_rgb_hist = pg.PlotWidget(name='RGB_HIST')  # lbl_JPG_RGB_hist
        self.pw_rgb_hist.setXRange(1,  350, padding=0)
        #self.pw_rgb_hist.setYRange(0, 1400, padding=0)

        self.hist_plot_rgb_r = self.pw_rgb_hist.plot(pen = 'r')
        self.hist_plot_rgb_g = self.pw_rgb_hist.plot(pen = 'g')
        self.hist_plot_rgb_b = self.pw_rgb_hist.plot(pen = 'b')
        self.ui.gridLayout.addWidget(self.pw_rgb_hist, 0, 2, 1, 1)

        self.pw_hdr_hist = pg.PlotWidget(name='HDR_HIST')  # lbl_HDR_RGB_hist
        self.pw_hdr_hist.setXRange(1,  350, padding=0)

        self.hist_plot_hdr_r = self.pw_hdr_hist.plot(pen = 'r')
        self.hist_plot_hdr_g = self.pw_hdr_hist.plot(pen = 'g')
        self.hist_plot_hdr_b = self.pw_hdr_hist.plot(pen = 'b')
        self.ui.gridLayout.addWidget(self.pw_hdr_hist, 1, 2, 1, 1)

        ###########################################################
        #  Settings and initial values
        ###########################################################
        self.hdr_img_shape = 0
        self.jpg_img_shape = 0
        self.qimage_width  = 2592
        self.qimage_height = 1944
        self.MAP = 'HSV'                   # Set type of color map used 'JET','HSV'
        self.database_name = "img_analysis.db"
        self.database_avaiable = False

        ###########################################################
        #  Misc Variables
        ###########################################################
        self.tot_numb_of_images = 0
        self.name_of_current_imgProcFunc = None  # name (string) of currently used img proc function
        self.image_mask = None
        self.imageLoaded = False
        self.scale_fac_width  = None
        self.scale_fac_height = None
        self.qLable_width = None
        self.qLable_height = None
        self.pass_this_imgProcFunc = None  # Placeholder for currently used img proc function
        self.curr_item_slec_cam1 = None
        self.curr_item_slec_cam2 = None
        self.curr_item_slec_path = None
        self.CAM = None                    # indicating weather camera_1 or camera_2 was used

        ###########################################################
        #  Boolean variables
        ###########################################################
        self.optFlowList_exists = False  # If a list of optical flow img's exists
        self.loading_complete   = False  # process of loading images is completed

        ###########################################################
        # Load images from last session as background task
        ###########################################################
        self.root_path_to_images = None   # root containing folders camera_1 resp camera_2
        self.root_path_to_images = self.settings.value("path_to_images")
        temp_text = self.load_last_root_path_from_settings(self.root_path_to_images)
        self.ui.lineEdit.setText(temp_text)

        ###########################################################
        # All variables declared and ready to be used
        ###########################################################
        self.ready = True  # If iInitialization completed

    #########################################################
    # File Dialoge
    #########################################################
    def load_last_root_path_from_settings(self, last_root_path):

        self.enable_disable_run_but(False)

        if (os.path.isfile("img_analysis.db")):
            self.database_avaiable = True
            text = 'Found Database. Would you like to load it?'
            ret = self.showMsgBox(text)
            if ret == QMessageBox.Yes:
                print('Loading database')
                self.ui.lbl_progress_status.setText('Loading database')

                ok = self.load_database()

                if(ok):
                    self.enable_disable_run_but(True)
                    self.loading_complete = True
                else:
                    self.database_avaiable = True
                    self.load_last_root_path(last_root_path)

                return last_root_path
        else:
            self.load_last_root_path(last_root_path)


    def load_last_root_path(self, last_root_path):
        self.ui.lbl_progress_status.setText('Select a image folder')

        if last_root_path:
            last_root_path = re.sub(r'\s+', '', last_root_path)
            self.root_path_to_images = last_root_path
            self.populateListViews(last_root_path)
        else:
            self.showErrorMsgBox("Path to images not set!", "")

        return last_root_path

    def populateListViews(self, cam_root_path):
        try:
            self.ui.listWidget_cam1.clear()
            self.ui.listWidget_cam2.clear()

            if os.path.exists(join(cam_root_path, 'camera_1')):
                self.cam1_dirs_path_list = os.listdir(join(cam_root_path,'camera_1'))

                if self.cam1_dirs_path_list:
                    self.ui.listWidget_cam1.addItems(self.cam1_dirs_path_list)
                else:
                    self.showErrorMsgBox('Empty camera_1 folder !', '')
            else:
                self.showErrorMsgBox('No Camera 1 files found !','')
                return

            if os.path.exists(join(cam_root_path, 'camera_2')):
                self.cam2_dirs_path_list = os.listdir(join(cam_root_path, 'camera_2'))

                if self.cam2_dirs_path_list:
                    self.ui.listWidget_cam2.addItems(self.cam2_dirs_path_list)
                else:
                    self.showErrorMsgBox('Empty camera_2 folder !', '')
            else:
                self.showErrorMsgBox('No Camera 2 files found !', '')
                return

        except Exception as e:
            print('collectSubCamDirs: Error: ' + str(e))

    def load_all_images(self):

        try:
            if(not self.jpg_img_path_list or not self.hdr_img_path_list):
                self.showErrorMsgBox("Could not load images.","No images found.")
                return

            self.tot_numb_of_images = len(self.jpg_img_path_list)
            totnum_of_imgs = self.tot_numb_of_images // 10

            print('Start after {} images'.format(totnum_of_imgs))

            img_cnt = 0
            ready = True
            self.ui.lbl_progress_status.setText('Loading Images')
            status_txt = 'Loading Images: .'

            for jpg_path, hdr_path in zip(self.jpg_img_path_list,self.hdr_img_path_list):

                self.loading_complete = False

                jpg_img = self.readImg_as_BGR2RGB(jpg_path)
                hdr_img = self.readImg_as_BGR2RGB(hdr_path)

                self.openCVImg_JPG_img_list.append(jpg_img)
                self.openCVImg_HDR_img_list.append(hdr_img)

                self.pixMapImg_JPG_img_list.append(self.cv2qpixmap(jpg_img))
                self.pixMapImg_HDR_img_list.append(self.cv2qpixmap(hdr_img))

                img_cnt += 1

                if len(status_txt) >= 200:
                    status_txt = 'Loading Images: .'
                else:
                    status_txt = status_txt + '.'

                self.ui.lbl_progress_status.setText(status_txt)
                self.ui.lbl_info.setText("Load {} | {}".format(img_cnt,self.tot_numb_of_images))

                if (img_cnt >= totnum_of_imgs) & ready:
                    self.enable_disable_run_but(True)
                    ready = False

            self.loading_complete = True
            self.ui.lbl_info.setText('Done')

        except Exception as e:
            print("Error: could not load image: ", str(e))

    def getImgDir_from_lineEdit(self):
        pathString = self.jpg_img_path_list[0]
        filePath = pathString.split('/')
        filePath = filePath[:-1]
        imageDirectory = ''
        for item in filePath:
            imageDirectory = os.path.join(imageDirectory, item)

        bs = r'\ '

        imageDirectory = imageDirectory.replace(':', ':' + bs).strip()
        imageDirectory = re.sub(r'\s+', '', imageDirectory)
        return imageDirectory

    def getOutputFolderName(self,base_path="", message="Select a folder"):
        try:
            if not base_path:
                base_path = self.ui.lineEdit.text()
                if not base_path:
                    base_path = self.root_path_to_images
                    if not base_path:
                        base_path = "C:\\"

            dir = str(QFileDialog.getExistingDirectory(self,message,str(base_path),QFileDialog.ShowDirsOnly))
            self.ui.lineEdit.setText(dir)
            self.root_path_to_images = dir
            self.populateListViews(dir)

        except Exception as e:
            QMessageBox.about(self, "Error: could not open requested path: ", str(e))

    def enable_disable_run_but(self,on_off):
        try:
            if on_off:
                self.ui.lbl_progress_status.setText('')
                self.ui.pushButton_RunImgSeq.setText('Ready')
                self.ui.pushButton_RunImgSeq.setStyleSheet("background-color: yellow")
                self.ui.pushButton_RunImgSeq.setEnabled(True)
            else:
                self.ui.pushButton_RunImgSeq.setStyleSheet("background-color: #f0f0f0")
                self.ui.pushButton_RunImgSeq.setEnabled(False)

        except Exception as e:
           print("enable_disable_run_but:  ", str(e))

    #########################################################
    # SQLite database related
    #########################################################
    def load_database(self):
        try:
            db_ok = False
            db = sqlite3.connect(self.database_name)
            cursor = db.cursor()
            cursor.execute("SELECT * FROM jpg_img ORDER BY id DESC LIMIT 1")
            last_id = cursor.fetchone()[0]

            if last_id > 0:
                self.tot_numb_of_images = last_id
                db_ok = True
            else:
                db_ok = False

            return db_ok

        except Exception as e:
            print('load_database: Error: ' + str(e))

    def create_DB(self):
        try:
            db = sqlite3.connect(self.database_name)
            cursor = db.cursor()

            cursor.execute("DROP TABLE IF EXISTS jpg_img")
            cursor.execute("CREATE TABLE jpg_img(id INT, img BLOB)")

            cursor.execute("DROP TABLE IF EXISTS hdr_img")
            cursor.execute("CREATE TABLE hdr_img(id INT, img BLOB)")

            cursor.execute("DROP TABLE IF EXISTS jpg_hsv")
            cursor.execute("CREATE TABLE jpg_hsv(id INT, img BLOB)")

            cursor.execute("DROP TABLE IF EXISTS jpg_jet")
            cursor.execute("CREATE TABLE jpg_jet(id INT, img BLOB)")

            cursor.execute("DROP TABLE IF EXISTS hdr_hsv")
            cursor.execute("CREATE TABLE hdr_hsv(id INT, img BLOB)")

            cursor.execute("DROP TABLE IF EXISTS hdr_jet")
            cursor.execute("CREATE TABLE hdr_jet(id INT, img BLOB)")

            db.commit()
            db.close()
        except Exception as e:
            print('create_DB: Error: ' + str(e))

    def getImgByObjID(self, objid, type):
        try:
            con = sqlite3.connect(self.database_name)
            cur = con.cursor()
            query = "SELECT img FROM " +type+ " WHERE id ="+str(objid)
            cur.execute(query)
            blob = cur.fetchone()[0]

            cur.close()
            con.close()

            if blob:
                nparr = np.fromstring(blob, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = None

            return img

        except Exception as e:
            print('Error getImgByObjID: id:{} : {} '.format(objid,e))

    def img2MAP_byteStr(self,img,map,with_mask = True):
        mask = ImageProcessingLibrary.Mask(self.CAM, img.shape)

        if map == 'HSV':
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.applyColorMap(img_gray, cv2.COLORMAP_HSV)

        if map == 'JET':
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)

        if with_mask:
            img_masked = mask.maske_OpenCV_Image(img)
        else:
            img_masked = img

        img_bytestr = cv2.imencode('.jpg', img_masked)[1].tostring()

        return img_bytestr

    def load_all_images_toDB(self):

        try:
            if(not self.jpg_img_path_list or not self.hdr_img_path_list):
                self.showErrorMsgBox("Could not load images.","No images found.")
                return

            self.tot_numb_of_images = len(self.jpg_img_path_list)
            totnum_of_imgs = math.ceil(self.tot_numb_of_images * (1/3))

            print('Start after {} images'.format(totnum_of_imgs))

            img_cnt = 0
            ready = True
            self.ui.lbl_progress_status.setText('Loading Images')
            status_txt = 'Loading Images: .'

            self.create_DB()
            con = sqlite3.connect(self.database_name)
            cur = con.cursor()

            for jpg_path, hdr_path in zip(self.jpg_img_path_list,self.hdr_img_path_list):

                img_cnt += 1

                self.loading_complete = False

                jpg_img = self.readImg_as_BGR2RGB(jpg_path)
                hdr_img = self.readImg_as_BGR2RGB(hdr_path)

                jpg_bytes = cv2.imencode('.jpg', jpg_img)[1].tostring()
                hdr_bytes = cv2.imencode('.jpg', hdr_img)[1].tostring()

                JPG2HSV = self.img2MAP_byteStr(jpg_img,'HSV')
                JPG2JET = self.img2MAP_byteStr(jpg_img,'JET')

                HD2RHSV = self.img2MAP_byteStr(hdr_img,'HSV',with_mask=False)
                HDR2JET = self.img2MAP_byteStr(hdr_img,'JET',with_mask=False)

                cur.execute("insert into jpg_img VALUES(?,?)", (img_cnt, sqlite3.Binary(jpg_bytes)))
                cur.execute("insert into hdr_img VALUES(?,?)", (img_cnt, sqlite3.Binary(hdr_bytes)))

                cur.execute("insert into jpg_hsv VALUES(?,?)", (img_cnt, sqlite3.Binary(JPG2HSV)))
                cur.execute("insert into jpg_jet VALUES(?,?)", (img_cnt, sqlite3.Binary(JPG2JET)))

                cur.execute("insert into hdr_hsv VALUES(?,?)", (img_cnt, sqlite3.Binary(HD2RHSV)))
                cur.execute("insert into hdr_jet VALUES(?,?)", (img_cnt, sqlite3.Binary(HDR2JET)))

                con.commit()



                if len(status_txt) >= 200:
                    status_txt = 'Loading Images: .'
                else:
                    status_txt = status_txt + '.'

                self.ui.lbl_progress_status.setText(status_txt)
                self.ui.lbl_info.setText("Load {} | {}".format(img_cnt,self.tot_numb_of_images))

                if (img_cnt >= totnum_of_imgs) & ready:
                    self.enable_disable_run_but(True)
                    ready = False

            cur.close()
            con.close()
            self.ui.lbl_progress_status.setText("")
            self.loading_complete = True
            self.ui.lbl_info.setText('Done')

        except Exception as e:
            print("Error: could not load images to database: ", str(e))

    #########################################################
    # Image Processing
    #########################################################
    def create_mask(self,cam_type):

        try:
            if not cam_type:
                return

            img_mask = ImageProcessingLibrary.Mask(cam_type)
            self.image_mask = img_mask.get_mask_as_one_channel_img()

        except Exception as e:
            print('create_mask: Error: ' + str(e))

    def set_new_imgList_exists(self):

        if self.name_of_current_imgProcFunc == 'Optical Flow':
            self.optFlowList_exists = True

    def readImg_as_BGR2RGB(self, path_to_img):
        try:
            img = cv2.imread(path_to_img)
            opencv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #return deepcopy(opencv_img)
            return opencv_img

        except Exception as e:
            self.showErrorMsgBox("Could not read image: ", str(e))

    #########################################################
    # Plot functions
    #########################################################
    def plot_rgb_histogram(self, opencv_img):

        try:
            hist_r = cv2.calcHist([opencv_img], [0], self.image_mask, [256], [1, 256])
            hist_g = cv2.calcHist([opencv_img], [1], self.image_mask, [256], [1, 256])
            hist_b = cv2.calcHist([opencv_img], [2], self.image_mask, [256], [1, 256])

            x = np.arange(1, 257, dtype=int)

            self.hist_plot_rgb_r.setData(x,hist_r[:,0])
            self.hist_plot_rgb_g.setData(x,hist_g[:,0])
            self.hist_plot_rgb_b.setData(x,hist_b[:,0])


        except Exception as e:
            print("plot_rgb_histogram: ", str(e))

    def plot_hdr_histogram(self, hdr_img):

        try:
            hist_r = cv2.calcHist([hdr_img], [0], self.image_mask, [256], [1, 256])
            hist_g = cv2.calcHist([hdr_img], [1], self.image_mask, [256], [1, 256])
            hist_b = cv2.calcHist([hdr_img], [2], self.image_mask, [256], [1, 256])

            x = np.arange(1, 257, dtype=int)

            self.hist_plot_hdr_r.setData(x,hist_r[:,0])
            self.hist_plot_hdr_g.setData(x,hist_g[:,0])
            self.hist_plot_hdr_b.setData(x,hist_b[:,0])


        except Exception as e:
            print("plot_hdr_histogram: ", str(e))

    #########################################################
    # Navigating through images
    #########################################################
    def runSlideShow(self):

        try:
            if self.slideshow_step >= self.tot_numb_of_images:
                self.slideshow_step = 1

            jpg_img = self.getImgByObjID(self.slideshow_step, 'jpg_img')
            hdr_img = self.getImgByObjID(self.slideshow_step, 'hdr_img')

            if  self.MAP is 'HSV':
                jpg_MAP = self.getImgByObjID(self.slideshow_step, 'jpg_hsv')
                hdr_MAP = self.getImgByObjID(self.slideshow_step, 'hdr_hsv')

            if  self.MAP is 'JET':
                jpg_MAP = self.getImgByObjID(self.slideshow_step, 'jpg_jet')
                hdr_MAP = self.getImgByObjID(self.slideshow_step, 'hdr_jet')

            self.ui.lbl_JPG_img.setPixmap(self.cv2qpixmap(jpg_img))
            self.ui.lbl_HDR_img.setPixmap(self.cv2qpixmap(hdr_img))

            self.ui.lbl_JPG_COLMAP.setPixmap(self.cv2qpixmap(jpg_MAP))
            self.ui.lbl_HDR_COLMAP.setPixmap(self.cv2qpixmap(hdr_MAP))

            self.plot_rgb_histogram(jpg_img)
            self.plot_hdr_histogram(hdr_img)

            self.slideshow_step += 1

            if self.loading_complete:
                self.ui.lbl_progress_status.setText(
                'Showing image {} of {} '.format(self.slideshow_step, self.tot_numb_of_images))

        except Exception as e:
            print('Error in runSlideShow: {}'.format(e))

    def img2Colmapd(self, opencv_img, map ='HSV'):
        #https://www.tutorialspoint.com/opencv/opencv_color_maps.htm
        try:
            mask = ImageProcessingLibrary.Mask(self.CAM, opencv_img.shape)

            if map == 'HSV':
                img_gray = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2GRAY)
                img = cv2.applyColorMap(img_gray, cv2.COLORMAP_HSV)

            if map == 'JET':
                img_gray = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2GRAY)
                img = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)

            img_masked = mask.maske_OpenCV_Image(img)

            return img_masked

        except Exception as e:
            print('Error in img2Colmpd: {}'.format(e))

    def startStopSlideShow(self):

        if (len(self.jpg_img_path_list) == 0 and not self.database_avaiable):
            return

        btn_text = self.ui.pushButton_RunImgSeq.text()

        if btn_text == 'Ready':
            self.imageLoaded = True
            self.ui.pushButton_RunImgSeq.setText('Stop')
            self.timer_slideshow.start(SLIDESHOW_INTERVAL)
            self.ui.pushButton_RunImgSeq.setStyleSheet("background-color: red")
            return

        if btn_text == 'Run':
            self.ui.pushButton_RunImgSeq.setText('Stop')
            self.timer_slideshow.start(SLIDESHOW_INTERVAL)
            self.ui.pushButton_RunImgSeq.setStyleSheet("background-color: red")
            self.ui.pushButton_Step_Right.setEnabled(False)
            self.ui.pushButton_Step_Left.setEnabled(False)
            self.ui.pushButton_browse.setEnabled(False)
            return

        if btn_text == 'Stop':
            self.ui.pushButton_RunImgSeq.setText('Run')
            self.ui.pushButton_RunImgSeq.setStyleSheet("background-color: green")
            self.timer_slideshow.stop()
            self.ui.pushButton_Step_Right.setEnabled(True)
            self.ui.pushButton_Step_Left.setEnabled(True)
            self.ui.pushButton_browse.setEnabled(True)

    def backward_oneImage(self):
        try:
            self.slideshow_step -= 1

            if self.slideshow_step < 0:
                self.slideshow_step = self.tot_numb_of_images - 1
                current_image = self.slideshow_step
            if self.slideshow_step == 0:
                current_image = self.tot_numb_of_images
            else:
                current_image = self.slideshow_step

            jpg_img = self.getImgByObjID(self.slideshow_step, 'jpg_img')
            hdr_img = self.getImgByObjID(self.slideshow_step, 'hdr_img')

            if  self.MAP is 'HSV':
                jpg_MAP = self.getImgByObjID(self.slideshow_step, 'jpg_hsv')
                hdr_MAP = self.getImgByObjID(self.slideshow_step, 'hdr_hsv')

            if  self.MAP is 'JET':
                jpg_MAP = self.getImgByObjID(self.slideshow_step, 'jpg_jet')
                hdr_MAP = self.getImgByObjID(self.slideshow_step, 'hdr_jet')

            self.ui.lbl_JPG_img.setPixmap(self.cv2qpixmap(jpg_img))
            self.ui.lbl_HDR_img.setPixmap(self.cv2qpixmap(hdr_img))

            self.ui.lbl_JPG_COLMAP.setPixmap(self.cv2qpixmap(jpg_MAP))
            self.ui.lbl_HDR_COLMAP.setPixmap(self.cv2qpixmap(hdr_MAP))

            self.plot_rgb_histogram(jpg_img)
            self.plot_hdr_histogram(hdr_img)


            self.ui.lbl_progress_status.setText(
                'Showing image {} of {} '.format(current_image, self.tot_numb_of_images))

        except Exception as e:
            self.slideshow_step += 1
            print('Slideshow : {}'.format(e))

    def forward_oneImage(self):

        try:
            self.slideshow_step += 1

            if self.slideshow_step >= self.tot_numb_of_images:
                self.slideshow_step = 0
                current_image = self.tot_numb_of_images
            else:
                current_image = self.slideshow_step

            jpg_img = self.getImgByObjID(self.slideshow_step, 'jpg_img')
            hdr_img = self.getImgByObjID(self.slideshow_step, 'hdr_img')

            if  self.MAP is 'HSV':
                jpg_MAP = self.getImgByObjID(self.slideshow_step, 'jpg_hsv')
                hdr_MAP = self.getImgByObjID(self.slideshow_step, 'hdr_hsv')

            if  self.MAP is 'JET':
                jpg_MAP = self.getImgByObjID(self.slideshow_step, 'jpg_jet')
                hdr_MAP = self.getImgByObjID(self.slideshow_step, 'hdr_jet')

            self.ui.lbl_JPG_img.setPixmap(self.cv2qpixmap(jpg_img))
            self.ui.lbl_HDR_img.setPixmap(self.cv2qpixmap(hdr_img))

            self.ui.lbl_JPG_COLMAP.setPixmap(self.cv2qpixmap(jpg_MAP))
            self.ui.lbl_HDR_COLMAP.setPixmap(self.cv2qpixmap(hdr_MAP))

            self.plot_rgb_histogram(jpg_img)
            self.plot_hdr_histogram(hdr_img)

            self.ui.lbl_progress_status.setText(
                'Showing image {} of {} '.format(current_image, self.tot_numb_of_images))

        except Exception as e:
            self.slideshow_step -= 1
            print('Foreward one image: {}'.format(e))

    def getDirectories(self, pathToDirectories, max_img_to_load = 2000):
        try:
            allDirs = []
            img_cnt = 1

            for dirs in sorted(glob(os.path.join(pathToDirectories, "*", ""))):
                if img_cnt > max_img_to_load:
                    break
                elif os.path.isdir(dirs):
                    if dirs.rstrip('\\').rpartition('\\')[-1] :
                        allDirs.append(dirs)
                        # print('{}'.format(str(dirs)))
                        img_cnt += 1

            print('All images loaded! - Found {} images.'.format(img_cnt))

            return allDirs

        except Exception as e:
            print('getDirectories: Error: ' + str(e))

    #########################################################
    # Misc
    #########################################################
    def updateScaleFac(self):
        self.scale_fac_width  = np.rint(self.qimage_width  / self.qLable_width)
        self.scale_fac_height = np.rint(self.qimage_height / self.qLable_height)

    def calcTruePixCoordinates(self, mouse_x, mouse_y):
        true_x = mouse_x * self.scale_fac_width
        if true_x > self.qimage_width:
            true_x = self.qimage_width

        true_y = mouse_y * self.scale_fac_height
        if true_y > self.qimage_height:
            true_y = self.qimage_height

        trueCoordinates = QPoint(true_x,true_y)

        return trueCoordinates

    def getPixelValue(self,true_x,true_y):

        try:
            if self.imageLoaded:
                if len(self.pixMapImg_HDR_img_list) > 0:

                    qimage = self.pixMapImg_HDR_img_list[self.slideshow_step - 1].toImage()
                    pixel_val = qimage.pixel(true_x, true_y)

                    _red   = qRed(pixel_val)
                    _green = qGreen(pixel_val)
                    _blue  = qBlue(pixel_val)

                    RGB = namedtuple("RGB", "red green blue")
                    pixel_rgb = RGB(red=_red, green=_green, blue=_blue)

                return pixel_rgb

        except Exception as e:
            print('Error getPixelValue: {}'.format(e))

    def closeEvent(self, event):

        if self.root_path_to_images is not None:
            self.settings.setValue("path_to_images", self.root_path_to_images)
            event.accept()

    def cv2qpixmap(self, openCV_img):
        height, width, channel = openCV_img.shape
        qt_img = QImage(openCV_img, width, height, QImage.Format_RGB888)
        qpixmap_img = QPixmap.fromImage(qt_img)
        return qpixmap_img

    def extractObjID(self, path):
        filename = path.rpartition('\\')[-1]
        objid = int(os.path.splitext(filename)[0])

        return objid

    def set_img5_size(self, img):
        if img is None:
            print('Error set_img5_size: missing image')
            return
        self.jpg_img_shape = img.shape

    def set_hdr_size(self, img):
        if img is None:
            print('Error set_hdr_size: missing image')
            return
        self.hdr_img_shape = img.shape


    #########################################################
    # Widget events
    #########################################################
    def draw_circle(self, event, x, y, flags, param):
        global ix, iy
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # cv2.circle(img, (x, y), 100, (255, 0, 0), -1)
            self.mouseX, self.mouseY = x, y

    def eventFilter(self, srcEvent, event):
        try:

            x = 0
            y = 0

            if srcEvent == self.ui.lbl_HDR_img:
                if event.type() == QEvent.Leave:
                    string = 'empty'

                if event.type() == QEvent.Resize:
                    self.qLable_width  = srcEvent.width()
                    self.qLable_height = srcEvent.height()

                    self.updateScaleFac()
                    #print('scaleFactor : width: {}  height: {}'.format(self.scale_fac_width,self.scale_fac_height))

                if event.type() == QEvent.MouseMove:
                    x = event.pos().x()
                    y = event.pos().y()
                    self.updatePixelText(x,y)

                elif event.type() == QEvent.MouseButtonPress:
                    if event.button() == Qt.RightButton:
                        # print('Rightclick')
                        return False

                    elif event.button() == Qt.LeftButton:
                        # print('Leftclick ')

                        x = event.pos().x()
                        y = event.pos().y()
                        self.updatePixelText(x, y)

                        return False
                    else:
                        return False

            return False


        except Exception as e:
            print('Error evenFilter: {}'.format(e))

    def showMsgBox(self, str_msg, infoTxt='Message'):
        try:
            qm = QMessageBox()
            ret = qm.question(self, infoTxt, str_msg,qm.Yes | qm.No)
            return ret

        except Exception as e:
            print('showMsgBox: {}'.format(e))

    def showErrorMsgBox(self, str_msg, error):
        try:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText(str_msg)
            msg.setInformativeText(error)
            msg.setWindowTitle("Error")
            msg.setTextInteractionFlags(Qt.TextSelectableByMouse)
            msg.exec()
        except Exception as e:
            print('showErrorMsgBox: {}'.format(e))

    def on_item_changed_cam1(self,curr, prev):
        self.curr_item_slec_cam1 = curr.text()

    def on_item_changed_cam2(self,curr, prev):
        self.curr_item_slec_cam2 = curr.text()

    def on_item_doubleclicked_cam1(self):
        try:
            new_path = join(self.root_path_to_images,'camera_1',self.curr_item_slec_cam1)
            self.collectAllImgSubDirectories(new_path)
            self.CAM = 1
            #self.create_mask(self.CAM)
        except Exception as e:
            print('on_item_doubleclicked_cam1: {}'.format(e))

    def on_item_doubleclicked_cam2(self):
        try:
            new_path = join(self.root_path_to_images,'camera_2',self.curr_item_slec_cam2)
            self.collectAllImgSubDirectories(new_path)
            self.CAM = 2
            #self.create_mask(self.CAM)
        except Exception as e:
            print('on_item_doubleclicked_cam2: {}'.format(e))

    def collectAllImgSubDirectories(self,path_to_sourceDir):
        try:
            sourceDir = path_to_sourceDir.replace('/','\\')

            if not os.path.exists(join(sourceDir, 'temp')):
                msg = "could not find temp directory. Missing preprocessed files?"
                error = ""
                self.showErrorMsgBox(msg, error)
                return

            self.ui.pushButton_browse.setEnabled(False)
            allDirs = self.getDirectories(join(sourceDir, 'temp'), max_img_to_load = MAX_LOAD_IMAGES)

            name_img5 = "raw_img5.jpg"
            name_hdr  = "hdr_data.jpg"

            for dir in allDirs:
                path_img = join(dir,name_img5)
                path_hdr = join(dir,'output',name_hdr)
                self.jpg_img_path_list.append(path_img)
                self.hdr_img_path_list.append(path_hdr)

            jpg_img = self.readImg_as_BGR2RGB(self.jpg_img_path_list[0])
            hdr_img = self.readImg_as_BGR2RGB(self.hdr_img_path_list[0])
            self.set_img5_size(jpg_img)
            self.set_hdr_size(hdr_img)

            pool = ThreadPoolExecutor(max_workers=3)
            # pool.submit(self.load_all_images)
            pool.submit(self.load_all_images_toDB)

        except Exception as e:
            print('collectAllImgSubDirectories: Error: ' + str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
