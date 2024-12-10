import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import matplotlib.ticker as mticker
from IPython import get_ipython
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.widgets import RectangleSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import threading
from decimal import Decimal
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QHBoxLayout, QWidget, QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSizePolicy
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.figure import Figure
from threading import Event
import io
from PIL import Image as PILImage
from IPython.display import display, Image as IPImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget
from PyQt5 import uic
from scipy.optimize import curve_fit
import os

   
from .logic import *
from .live_plot import *



class MplCanvas(FigureCanvasQTAgg):
    """
    labels = [xlabel, ylabel]
    """
    def __init__(self, parent=None, labels=None, mode=None, scale=1):

        change_to_inline(params_type = 'nbagg', scale=scale)

        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(111)
        if labels is not None:
            self.axes.set_xlabel(labels[0])
            self.axes.set_ylabel(labels[1])
            
        if mode=='PL':
            
            line = self.axes.imshow(1e4*np.random.random((100, 100)), animated=True, cmap='inferno')
            cbar = self.fig.colorbar(line)
            cbar.set_label(labels[2])
        elif mode=='PLdis':
            
            divider = make_axes_locatable(self.axes)
            axright = divider.append_axes("top", size="20%", pad=0.25)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            
            line = self.axes.imshow(1e4*np.random.random((100, 100)), animated=True, cmap='inferno')
            cbar = self.fig.colorbar(line, cax = cax)
            cbar.set_label(labels[2])
        elif mode=='PLE':
            self.axes.set_ylim(0, 1000)
            
        self.fig.tight_layout()
            
        super(MplCanvas, self).__init__(self.fig)

class DetachedWindow(QMainWindow):
    def __init__(self, widget, parent=None):
        super().__init__()
        self.widget = widget
        self.setCentralWidget(self.widget)
        self.setWindowTitle('Live')
        self.parent_window = parent
        self.resize(640, 700)

        self.scale = self.parent_window.config_instances['display_scale']
        self.init_size = self.size()
        self.setMaximumSize(self.init_size.width()*self.scale, self.init_size.height()*self.scale)
        self.setMinimumSize(self.init_size.width()*self.scale, self.init_size.height()*self.scale)

        self.widget.show()



    def closeEvent(self, event):
        self.parent_window.reattach_page()
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self, config_instances):
        super().__init__()
        self.config_instances = config_instances
        self.scanner = self.config_instances['scanner']
        self.wavemeter = self.config_instances.get('wavemeter')
        self.is_running = False
        self.selector_PLE = []
        self.selector_PL = []
        self.selector_Live = []
        self.is_wavelength_PL_flag = False
        self.is_wavelength_Live_flag = False
        self.data_figure_PL = None
        self.data_figure_PLE = None
        self.cur_plot = 'PL'
        self.is_fit = False
        self.spl = 299792458
        self.is_save_to_jupyter_flag = True
        self.time_PL = 0
        self.time_PLE = 0
        self.laser_stabilizer = None

        ui_path = os.path.join(os.path.dirname(__file__), "GUIv2.ui")
        uic.loadUi(ui_path, self)

        self.scale = self.config_instances['display_scale']
        self.init_size = self.size()
        self.setMaximumSize(self.init_size.width()*self.scale, self.init_size.height()*self.scale)
        self.setMinimumSize(self.init_size.width()*self.scale, self.init_size.height()*self.scale)
        self.scale_widgets(self.centralwidget, self.scale, is_recursive=True)
        # set size
        
        
        self.widget_figure_PLE = self.findChild(QWidget, 'widget_figure_PLE')
        layout = QVBoxLayout(self.widget_figure_PLE)
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas_PLE = MplCanvas(self.widget_figure_PLE, labels=['Wavelength (nm)', 'Counts'], mode='PLE', scale=self.scale)
        # makes sure overwirte format
        layout.addWidget(self.canvas_PLE)
        
        
        self.widget_figure_PL = self.findChild(QWidget, 'widget_figure_PL')
        layout = QVBoxLayout(self.widget_figure_PL)
        layout.setContentsMargins(0, 0, 0, 0) 

        self.canvas_PL = MplCanvas(self.widget_figure_PL, labels=['X', 'Y', 'Counts'], mode='PLdis', scale=self.scale)
        layout.addWidget(self.canvas_PL)
        
        self.widget_figure_Live = self.findChild(QWidget, 'widget_figure_Live')
        layout = QVBoxLayout(self.widget_figure_Live)
        layout.setContentsMargins(0, 0, 0, 0) 

        self.canvas_Live = MplCanvas(self.widget_figure_Live, labels=['Data', 'Counts'], mode='PLE', scale=self.scale)
        layout.addWidget(self.canvas_Live)
        

                

        self.pushButton_start_PLE.clicked.connect(self.start_plot_PLE)
        self.pushButton_start_PL.clicked.connect(self.start_plot_PL)
        self.pushButton_start_Live.clicked.connect(self.start_plot_Live)
        self.pushButton_stop_PLE.clicked.connect(self.stop_and_show)
        self.pushButton_stop_PL.clicked.connect(self.stop_and_show)
        self.pushButton_stop_Live.clicked.connect(self.stop_and_show)
        
        #self.radioButton_is_wavelength_PL.toggled.connect(self.is_wavelength_PL)
        self.checkBox_log.toggled.connect(self.is_save_to_jupyter)
        
        self.pushButton_wavelength.clicked.connect(self.read_wavelength)
        self.pushButton_range_PL.clicked.connect(self.read_range_PL)
        self.pushButton_range_PLE.clicked.connect(self.read_range_PLE)
        self.pushButton_lorent.clicked.connect(self.fit_lorent)
        self.pushButton_unit.clicked.connect(self.change_unit)
        self.pushButton_save_PL.clicked.connect(self.save_PL)
        self.pushButton_save_PLE.clicked.connect(self.save_PLE)
        self.pushButton_XY.clicked.connect(self.read_xy)

        self.pushButton_scanner.clicked.connect(self.move_scanner)

        self.doubleSpinBox_exposure_PL.valueChanged.connect(self.estimate_PL_time)
        self.doubleSpinBox_xl.valueChanged.connect(self.estimate_PL_time)
        self.doubleSpinBox_xu.valueChanged.connect(self.estimate_PL_time)
        self.doubleSpinBox_yl.valueChanged.connect(self.estimate_PL_time)
        self.doubleSpinBox_yu.valueChanged.connect(self.estimate_PL_time)
        self.doubleSpinBox_step_PL.valueChanged.connect(self.estimate_PL_time)
        # recalculate plot time of PL

        self.doubleSpinBox_exposure_PLE.valueChanged.connect(self.estimate_PLE_time)
        self.doubleSpinBox_wl.valueChanged.connect(self.estimate_PLE_time)
        self.doubleSpinBox_wu.valueChanged.connect(self.estimate_PLE_time)
        self.doubleSpinBox_step_PLE.valueChanged.connect(self.estimate_PLE_time)
        self.doubleSpinBox_step_PLE.valueChanged.connect(self.step_PLE_in_MHz)
        # recalculate plot time of PLE


        self.pushButton_detach.clicked.connect(self.detach_page)
        self.pushButton_reattach.clicked.connect(self.reattach_page)
        self.pushButton_reattach.setEnabled(False)  # Initially disabled

        # Placeholder for detached window
        self.detached_window = None

        self.timer_scanner = QtCore.QTimer()
        self.timer_scanner.setInterval(200)  # Interval in milliseconds
        self.timer_scanner.timeout.connect(self.read_scanner)
        self.timer_scanner.start()
        # real time scanner read

        self.timer_wavemeter = QtCore.QTimer()
        self.timer_wavemeter.setInterval(500)  # Interval in milliseconds
        self.timer_wavemeter.timeout.connect(self.read_wavemeter)
        self.timer_wavemeter.start()
        # real time wavemeter read


        self.doubleSpinBox_X.valueChanged.connect(self.bind_set)
        self.doubleSpinBox_Y.valueChanged.connect(self.bind_set)
        self.checkBox_is_bind.toggled.connect(self.bind_set)


        self.checkBox_is_stabilizer.toggled.connect(self.is_stabilizer)
        self.doubleSpinBox_wavelength.valueChanged.connect(self.is_stabilizer)



        self.init_widget()
        self.show()

    def is_stabilizer(self):

        if not self.checkBox_is_stabilizer.isChecked():
            if self.laser_stabilizer is not None:
                self.laser_stabilizer.stop()
            return
        # else, stabilizer is checked

        if (self.laser_stabilizer is None) or (not self.laser_stabilizer.is_alive()):
            self.laser_stabilizer = LaserStabilizer(config_instances = self.config_instances)
            self.laser_stabilizer.start()


        self.wavelength = self.doubleSpinBox_wavelength.value()
        self.laser_stabilizer.set_wavelength(self.wavelength)


    def bind_set(self):
        if self.checkBox_is_bind.isChecked():
            self.move_scanner()


    def read_scanner(self):
        x = self.scanner.x
        y = self.scanner.y

        self.lineEdit_X.setText(f'{x}')
        self.lineEdit_Y.setText(f'{y}')


    def read_wavemeter(self):
        if self.wavemeter is not None:
            wavelength = self.wavemeter.wavelength
            self.lineEdit_wavelength.setText(f'{wavelength:.5f}')


    def detach_page(self):
        # Remove the second tab and detach it into a new window
        page_widget = self.tabWidget.widget(1)
        #self.tabWidget.removeTab(1)

        # Create and show the detached window
        self.detached_window = DetachedWindow(page_widget, parent=self)
        self.detached_window.show()

        # Update button states
        self.pushButton_detach.setEnabled(False)
        self.pushButton_reattach.setEnabled(True)

    def reattach_page(self):
        # Add the widget back to the tab widget
        if self.detached_window:
            page_widget = self.detached_window.widget
            self.tabWidget.addTab(page_widget, "Live")
            self.detached_window.close()
            self.detached_window = None


        # Update button states
        self.pushButton_detach.setEnabled(True)
        self.pushButton_reattach.setEnabled(False)

    def scale_widgets(self, widget, scale_factor, is_recursive):
        #print('name',widget.objectName(), 'init width', widget.font().pointSizeF(), 'fa', fa.objectName())


        widget.setGeometry(widget.x()* scale_factor, widget.y()* scale_factor, 
                            widget.width() * scale_factor, widget.height() * scale_factor)

        font = widget.font()
        if isinstance(widget, QDoubleSpinBox):
            font.setPointSizeF(16 * scale_factor)
        else:
            font.setPointSizeF(12 * scale_factor)
        widget.setFont(font)


        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        widget.setSizePolicy(size_policy)

        #print('name',widget.objectName(), 'width', widget.font().pointSizeF())
        #widget.updateGeometry()
        if is_recursive:
            for child in widget.findChildren(QWidget):
                self.scale_widgets(child, scale_factor, is_recursive=False)


    def step_PLE_in_MHz(self):

        step_in_MHz = 1000*np.abs(self.spl/((self.wl + self.wu)/2) - self.spl/((self.wl + self.wu)/2 + self.step_PLE))
        self.lineEdit_step_PLE.setText(f'{step_in_MHz:.2f}MHz')

    def estimate_PL_time(self):
        if self.is_running and self.cur_plot=='PL':
            points_total = self.live_plot_PL.points_total
            points_done = self.live_plot_PL.points_done
            ratio = points_done/points_total
            self.lineEdit_time_PL.setText(f'PL finishes in {(ratio*self.time_PL):.2f}s / {self.time_PL:.2f}s, {ratio*100:.2f}%')

        else:
            self.read_data_PL()
            time = self.exposure_PL * len(np.arange(self.xl, self.xu, self.step_PL)) * len(np.arange(self.yl, self.yu, self.step_PL))
            self.lineEdit_time_PL.setText(f'new PL finishes in {time:.2f}s')
            self.time_PL = time

    def estimate_PLE_time(self):
        if self.is_running and self.cur_plot=='PLE':
            points_total = self.live_plot_PLE.points_total
            points_done = self.live_plot_PLE.points_done
            ratio = points_done/points_total
            self.lineEdit_time_PLE.setText(f'PLE finishes in {(ratio*self.time_PLE):.2f}s / {self.time_PLE:.2f}s, {ratio*100:.2f}%')

        else:
            self.read_data_PLE()
            time = (self.exposure_PLE + 0.5)* len(np.arange(self.wl, self.wu, self.step_PLE)) 
            # considering the overhead of stabilizing laser frequency
            self.lineEdit_time_PLE.setText(f'new PLE finishes in {time:.2f}s')
            self.time_PLE = time


    def init_widget(self):
        current_time = time.localtime()
        current_date = time.strftime("%Y-%m-%d", current_time)
        time_str = current_date.replace('-', '_')

        self.lineEdit_save.setText(f'{time_str}/')

        self.estimate_PLE_time()

        step_in_MHz = 1000*np.abs(self.spl/((self.wl + self.wu)/2) - self.spl/((self.wl + self.wu)/2 + self.step_PLE))
        self.lineEdit_step_PLE.setText(f'{step_in_MHz:.2f}MHz')

        self.estimate_PL_time()




    def move_scanner(self):
        x = self.doubleSpinBox_X.value()
        y = self.doubleSpinBox_Y.value()

        self.scanner.x = x
        self.scanner.y = y
        #self.print_log(f'moved scanner to (x = {x}, y = {y})')

    def move_scanner_Live(self):

        x = self.doubleSpinBox_X_Live.value()
        y = self.doubleSpinBox_Y_Live.value()

        self.scanner.x = x
        self.scanner.y = y
        self.print_log(f'moved scanner to (x = {x}, y = {y})')

        
    def print_log(self, text, is_print_jupyter = True):
        self.lineEdit_print.setText(text)
        if is_print_jupyter:
            print(text)
        # out put in jupyter cell
        
    def read_xy(self):
        if self.selector_PL == []:
            return
        _xy = self.selector_PL[1].xy #cross selector
        
        if _xy is not None:
            self.doubleSpinBox_X.setValue(_xy[0])
            self.doubleSpinBox_Y.setValue(_xy[1])
            self.print_log(f'read x = {_xy[0]}, y = {_xy[1]}')
        else:
            self.print_log(f'read x = None, y = None')
                
    def is_save_to_jupyter(self, checked):
        if checked:
            self.is_save_to_jupyter_flag = True
        else:
            self.is_save_to_jupyter_flag = False
                
    def save_to_jupyter(self, fig):

        current_time = time.localtime()
        current_date = time.strftime("%Y-%m-%d", current_time)
        current_time_formatted = time.strftime("%H:%M:%S", current_time)
        time_str = current_date.replace('-', '_') + '_' + current_time_formatted.replace(':', '_')


        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        img_data = buffer.getvalue()
        img = PILImage.open(io.BytesIO(img_data))
        print(
        f"""
        {time_str}
        """)
        display(IPImage(img_data, format='png'))
        #fig.canvas.draw()
        
        
    def save_PL(self):
        if self.data_figure_PL is None:
            return
        addr = self.lineEdit_save.text()
        self.data_figure_PL.save(addr = addr)

        info = self.data_figure_PL.info
        self.print_log(f'info: {info}')

        if self.is_save_to_jupyter_flag:
            self.save_to_jupyter(self.canvas_PL.fig)
            
        self.print_log('saved PL')
        
    def save_PLE(self):
        if self.data_figure_PLE is None:
            return
        addr = self.lineEdit_save.text()
        self.data_figure_PLE.save(addr = addr) 

        info = self.data_figure_PLE.info
        self.print_log(f'info: {info}')

        if self.is_save_to_jupyter_flag:
            self.save_to_jupyter(self.canvas_PLE.fig)
            
        self.print_log('saved PLE')
        
        
    def change_unit(self):
        if self.data_figure_PLE is None:
            self.print_log(f'No figure to change unit')
            return
        if self.data_figure_PLE.unit == 'nm':
            self.data_figure_PLE.to_GHz()
            self.print_log(f'changed unit to GHz')
        else:
            self.data_figure_PLE.to_nm()
            self.print_log(f'changed unit to nm')
        
    def fit_lorent(self):
        if self.data_figure_PLE is None:
            self.print_log(f'No figure to fit')
            return
        if self.is_fit:
            self.data_figure_PLE.clear()
            self.is_fit = False
            self.print_log(f'fit cleared')
        else:
            self.data_figure_PLE.lorent()
            self.is_fit = True
            log_info = self.data_figure_PLE.log_info
            self.print_log(f'curve fitted, {log_info}')
        
            
    def read_wavelength(self):
        if self.selector_PLE == []:
            self.print_log(f'No wavelength to read')
            return
        _wavelength = self.selector_PLE[1].wavelength
        
        if _wavelength is not None:
            if self.data_figure_PLE.unit == 'nm':
                self.doubleSpinBox_wavelength.setValue(_wavelength)
            else:
                self.doubleSpinBox_wavelength.setValue(self.spl/_wavelength)
            self.print_log(f'wavelength was read to PL')
        else:
            self.print_log(f'No wavelength to read')
        #set double spin box to _wavelength
        
        
    def read_range_PL(self):
        if self.selector_PL == []:
            self.print_log(f'no area to read range')
            return
        xl, xh, yl, yh = self.selector_PL[0].range
        
        if xl is None:
            self.print_log(f'no area to read range')
            return
        
        self.doubleSpinBox_xl.setValue(xl)
        self.doubleSpinBox_xu.setValue(xh)
        self.doubleSpinBox_yl.setValue(yl)
        self.doubleSpinBox_yu.setValue(yh)

        self.print_log(f'PL range updated')
        
    def read_range_PLE(self):
        if self.selector_PLE == []:
            self.print_log(f'no area to read range')
            return
        xl, xh, yl, yh = self.selector_PLE[0].range
        
        if xl is None:
            self.print_log(f'no area to read range')
            return
        
        if self.data_figure_PLE.unit == 'nm':
            self.doubleSpinBox_wl.setValue(xl)
            self.doubleSpinBox_wu.setValue(xh)
        else:
            self.doubleSpinBox_wl.setValue(self.spl/xh)# lower wavelength is higher frequency
            self.doubleSpinBox_wu.setValue(self.spl/xl)

        self.print_log(f'PLE range updated')
        
        
    def read_data_PLE(self):
        for attr in ['exposure_PLE', 'wl', 'wu', 'step_PLE']: # Read from GUI panel
            value = getattr(self, f'doubleSpinBox_{attr}').value()
            setattr(self, attr, value)
            
        for attr in ['center', 'height', 'width']:
            value = self.config_instances.get(attr)
            setattr(self, attr, value)
            
    def read_data_PL(self):
        for attr in ['exposure_PL', 'xl', 'xu', 'yl', 'yu', 'step_PL']:
            value = getattr(self, f'doubleSpinBox_{attr}').value()
            setattr(self, attr, value)
            
    def read_data_Live(self):
        for attr in ['exposure_Live', 'many']:
            value = getattr(self, f'doubleSpinBox_{attr}').value()
            setattr(self, attr, value)
            

    def start_plot_PLE(self):
        self.print_log(f'PLE started')
        self.cur_plot = 'PLE'
        self.stop_plot()
        self.is_running = True
                    
        
        self.read_data_PLE()
        
        data_x = np.arange(self.wl, self.wu, self.step_PLE)
        data_y = np.zeros(len(data_x))

        self.checkBox_is_stabilizer.setChecked(False)
        self.checkBox_is_stabilizer.setDisabled(True)
        self.doubleSpinBox_wavelength.setDisabled(True)
        self.checkBox_is_bind.setChecked(False)
        self.checkBox_is_bind.setDisabled(True)

        self.print_log(f'scanner: (x={self.scanner.x}, y={self.scanner.y})')


        self.data_generator_PLE = PLEAcquire(exposure = self.exposure_PLE, \
                                         data_x=data_x, data_y=data_y, config_instances=self.config_instances)
        self.live_plot_PLE = PLELive(labels=['Wavelength (nm)', f'Counts/{self.exposure_PLE:.2f}s'], 
                                     update_time=1, data_generator=self.data_generator_PLE, data=[data_x, data_y],\
                                    fig=self.canvas_PLE.fig, config_instances=self.config_instances)
        

        self.live_plot_PLE.init_figure_and_data()
        
        
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000*self.live_plot_PLE.update_time)  # Interval in milliseconds
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
            
            
    def start_plot_PL(self):
        self.print_log(f'PL started')
        self.cur_plot = 'PL'
        self.stop_plot()
        self.is_running = True
            
            
        self.read_data_PL()
                
        center = [0,0]
        data_x = np.arange(self.xl, self.xu, self.step_PL) + center[0]
        data_y = np.arange(self.yl, self.yu, self.step_PL) + center[1]
        data_z = np.zeros((len(data_y), len(data_x)))
        # reverse for compensate x,y order of imshow
        if self.checkBox_is_stabilizer.isChecked():
            _wavelength = self.doubleSpinBox_wavelength.value()
        else:
            _wavelength = None

        self.print_log(f'wavelength: {_wavelength}')

        self.checkBox_is_stabilizer.setDisabled(True)
        self.doubleSpinBox_wavelength.setDisabled(True)
        self.checkBox_is_bind.setChecked(False)
        self.checkBox_is_bind.setDisabled(True)

        self.data_generator_PL = PLAcquire(exposure = self.exposure_PL, data_x = data_x, data_y = data_y, \
                               data_z = data_z, config_instances=self.config_instances, wavelength=_wavelength)
            
        self.live_plot_PL = PLGUILive(labels=['X', 'Y', f'Counts/{self.exposure_PL:.2f}s'], \
                        update_time=1, data_generator=self.data_generator_PL, data=[data_x, data_y, data_z],\
                                       fig=self.canvas_PL.fig, config_instances=self.config_instances)
        
        
        self.live_plot_PL.init_figure_and_data()
        
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000*self.live_plot_PL.update_time)  # Interval in milliseconds
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        
            
    def start_plot_Live(self):
        self.print_log(f'Live started')
        self.cur_plot = 'Live'
        self.stop_plot()
        self.is_running = True
        self.read_data_Live()
        
        
        data_x = np.arange(self.many)
        data_y = np.zeros(len(data_x))
        
        _wavelength = None
            
        self.data_generator_Live = LiveAcquire(exposure = self.exposure_Live, data_x=data_x, data_y=data_y, \
                                     config_instances=self.config_instances, wavelength=_wavelength, is_finite=False)
        self.live_plot_Live = PLELive(labels=['Data', f'Counts/{self.exposure_Live:.2f}s'], \
                            update_time=0.05, data_generator=self.data_generator_Live, data=[data_x, data_y],\
                                         fig=self.canvas_Live.fig, config_instances=self.config_instances)
        
        
        self.live_plot_Live.init_figure_and_data()
        
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000*self.live_plot_Live.update_time)  # Interval in milliseconds
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        
    def update_plot(self):
        
        attr = self.cur_plot
        live_plot_handle = getattr(self, f'live_plot_{attr}')

        if live_plot_handle.data_generator.is_alive() and self.is_running:

            live_plot_handle.update_figure()

            if attr == 'PL':
                self.estimate_PL_time()
            elif attr == 'PLE':
                self.estimate_PLE_time()
            # update estimate finish time
        else:

            self.timer.stop()

            live_plot_handle.update_figure()  
            live_plot_handle.line.set_animated(False)                
            live_plot_handle.axes.set_autoscale_on(False)     
            live_plot_handle.choose_selector()
            setattr(self, f'selector_{attr}', live_plot_handle.selector)
            live_plot_handle.stop()
            setattr(self, f'data_figure_{attr}', DataFigure(live_plot_handle))
            self.is_running = False

            if attr == 'PL':
                self.estimate_PL_time()
            elif attr == 'PLE':
                self.estimate_PLE_time()

            self.checkBox_is_stabilizer.setDisabled(False)
            self.doubleSpinBox_wavelength.setDisabled(False)
            self.checkBox_is_bind.setDisabled(False)
            
    
    def stop_plot(self):
        
        self.is_running = False
        self.is_fit = False
        attr = self.cur_plot
        
        for selector in getattr(self, f'selector_{attr}'):
            selector.set_active(False)
        setattr(self, f'selector_{attr}', []) #disable all selector
        setattr(self, f'data_figure_{attr}', None) #disable DataFigure
        
        if hasattr(self, 'timer'):
            self.timer.stop()
            
            
        if hasattr(self, 'live_plot_PLE') and self.live_plot_PLE is not None:
            self.live_plot_PLE.stop()
            
        if hasattr(self, 'live_plot_PL') and self.live_plot_PL is not None:
            self.live_plot_PL.stop()
            
        if hasattr(self, 'live_plot_Live') and self.live_plot_Live is not None:
            self.live_plot_Live.stop()
            
            
            
            
    def stop_and_show(self):
        # stop acquiring data but enable save, fit and selector
        self.print_log(f'Plot stopped')
        self.is_running = False

        #if hasattr(self, 'timer'):
        #    self.timer.stop()

        if hasattr(self, 'live_plot_PLE') and self.live_plot_PLE is not None:
            self.estimate_PLE_time()
            
        if hasattr(self, 'live_plot_PL') and self.live_plot_PL is not None:
            self.estimate_PL_time()



            
    def closeEvent(self, event):
        self.stop_plot()
        plt.close('all') # make sure close all plots which avoids error message

        if self.detached_window is not None:
            self.detached_window.close()

        if self.laser_stabilizer is not None:
            self.laser_stabilizer.stop()
            
        event.accept()
        QtWidgets.QApplication.quit()  # Ensure application exits completely



def GUI(config_instances):
    """
    The function opens pyqt GUI for PLE, PL, live counts, and pulse control.
    Save button will also output data and figure to jupyter notebook.
   
    Examples
    --------
    >>> GUI()

    Read range button reads range from area created by mouse left 

	Read wavelength button reads wavelength from point created by mouse right in PLE

	Read XY button reads x, y coordinates from point created by mouse right in PL 

	Change unit changes PLE unit between 'nm' and 'GHz'

	Move sacnner moves scanner to x, y displayed 
    """

    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        
    
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    w = MainWindow(config_instances)
    app.setStyle('Windows')
    try:
        sys.exit(app.exec_())
    except SystemExit as se:
        if se.code != 0:
            raise se
    # make sure jupyter notebook does not catch error when exit normally
