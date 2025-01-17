import os, sys, time, threading, io
import numpy as np
from decimal import Decimal
from threading import Event

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.widgets import RectangleSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from PIL import Image as PILImage
from IPython import get_ipython
from IPython.display import display, HTML
from IPython.display import Image as IPImage

from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt, QSize
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QRadioButton, QHBoxLayout, QVBoxLayout\
, QPushButton, QGroupBox, QCheckBox, QLineEdit, QComboBox, QLabel, QFileDialog, QDoubleSpinBox, QSizePolicy, QDialog, QMessageBox
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5 import QtCore, QtWidgets, uic

from Confocal_GUI.live_plot import *


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
    def __init__(self, config_instances, measurement_PL, measurement_PLE, measurement_Live, ui='GUI.ui'):
        super().__init__()
        self.config_instances = config_instances
        self.is_running = False
        self.data_figure_PL = None
        self.data_figure_PLE = None
        self.cur_plot = 'PL'
        self.is_fit = False
        self.spl = 299792458
        self.is_save_to_jupyter_flag = True
        self.time_PL = 0
        self.measurement_PL = measurement_PL
        self.measurement_PLE = measurement_PLE
        self.measurement_Live = measurement_Live
        self.ui = ui

        ui_path = os.path.join(os.path.dirname(__file__), self.ui)
        uic.loadUi(ui_path, self)

        self.scale = self.config_instances['display_scale']
        self.init_size = self.size()
        self.setMaximumSize(self.init_size.width()*self.scale, self.init_size.height()*self.scale)
        self.setMinimumSize(self.init_size.width()*self.scale, self.init_size.height()*self.scale)
        self.scale_widgets(self.centralwidget, self.scale, is_recursive=True)
        self.setWindowTitle(f'ConfocalGUI@Wanglab, UOregon' )
        # set size
        
        self.load_figures()
        self.load_default()
        self.connect_buttons()
        self.init_widget()
        self.show()

    def load_figures(self):

        figures = [
            {'child_name': 'widget_figure_PL', 'labels': ['X', 'Y', 'Counts'], 'canvas_name': 'canvas_PL', 'mode': 'PLdis'},
            {'child_name': 'widget_figure_PLE', 'labels': ['Wavelength', 'Counts'], 'canvas_name': 'canvas_PLE', 'mode': 'PLE'},
            {'child_name': 'widget_figure_Live', 'labels': ['Data', 'Counts'], 'canvas_name': 'canvas_Live', 'mode': 'PLE'} 
        ]

        for fig in figures:
            widget = self.findChild(QWidget, fig['child_name'])
            if widget:
                layout = QVBoxLayout(widget)
                layout.setContentsMargins(0, 0, 0, 0)

                canvas = MplCanvas(widget, labels=fig['labels'], mode=fig['mode'], scale=self.scale)
                layout.addWidget(canvas)

                setattr(self, fig['canvas_name'], canvas)



    def connect_buttons(self):

        widget_config = {
            'buttons': [
                {'name': 'pushButton_load', 'func': self.load_file},
                # for load file
                {'name': 'pushButton_start_PLE', 'func': self.start_plot_PLE},
                {'name': 'pushButton_stop_PLE', 'func': self.stop_and_show},
                {'name': 'pushButton_wavelength', 'func': self.read_wavelength},
                {'name': 'pushButton_range_PLE', 'func': self.read_range_PLE},
                {'name': 'pushButton_lorent', 'func': self.fit_func},
                {'name': 'pushButton_save_PLE', 'func': self.save_PLE},
                {'name': 'pushButton_unit', 'func': self.change_unit},
                # for PLE
                {'name': 'pushButton_device_gui', 'func': self.open_device_gui},
                # for load device gui
                {'name': 'pushButton_start_Live', 'func': self.start_plot_Live},
                {'name': 'pushButton_stop_Live', 'func': self.stop_and_show},
                {'name': 'pushButton_detach', 'func': self.detach_page},
                {'name': 'pushButton_reattach', 'func': self.reattach_page},
                # for Live
                {'name': 'pushButton_start_PL', 'func': self.start_plot_PL},
                {'name': 'pushButton_stop_PL', 'func': self.stop_and_show},
                {'name': 'pushButton_XY', 'func': self.read_xy},
                {'name': 'pushButton_range_PL', 'func': self.read_range_PL},
                {'name': 'pushButton_scanner', 'func': self.move_scanner},
                {'name': 'pushButton_save_PL', 'func': self.save_PL},
                # for PL

            ],
            'checkBoxes': [
                {'name': 'checkBox_log', 'func': self.is_save_to_jupyter},
                {'name': 'checkBox_is_stabilizer', 'func': self.is_stabilizer},
                {'name': 'checkBox_is_bind', 'func': self.bind_set},
            ],
            'spinBoxes': [
                {'name': 'doubleSpinBox_exposure_PLE', 'func': self.estimate_PLE_time},
                {'name': 'doubleSpinBox_wl', 'func': self.estimate_PLE_time},
                {'name': 'doubleSpinBox_wu', 'func': self.estimate_PLE_time},
                {'name': 'doubleSpinBox_step_PLE', 'func': [self.estimate_PLE_time, self.step_PLE_in_MHz, self.update_stabilizer_step]},
                {'name': 'doubleSpinBox_wavelength', 'func': self.is_stabilizer},
                # for PLE
                {'name': 'doubleSpinBox_exposure_PL', 'func': self.estimate_PL_time},
                {'name': 'doubleSpinBox_xl', 'func': self.estimate_PL_time},
                {'name': 'doubleSpinBox_xu', 'func': self.estimate_PL_time},
                {'name': 'doubleSpinBox_yl', 'func': self.estimate_PL_time},
                {'name': 'doubleSpinBox_yu', 'func': self.estimate_PL_time},
                {'name': 'doubleSpinBox_step_PL', 'func': self.estimate_PL_time},
                {'name': 'doubleSpinBox_X', 'func': self.bind_set},
                {'name': 'doubleSpinBox_Y', 'func': self.bind_set},
                # for PL
            ]
        }



        for button_ in widget_config.get('buttons', []):
            name = button_['name']
            func = button_['func']
            button = self.findChild(QPushButton, name)
            if button:
                if isinstance(func, list):
                    for f in func:
                        button.clicked.connect(f)
                else:
                    button.clicked.connect(func)

        for checkbox_ in widget_config.get('checkBoxes', []):
            name = checkbox_['name']
            func = checkbox_['func']
            checkbox = self.findChild(QCheckBox, name)
            if checkbox:
                if isinstance(func, list):
                    for f in func:
                        checkbox.toggled.connect(f)
                else:
                    checkbox.toggled.connect(func)

        for spinbox_ in widget_config.get('spinBoxes', []):
            name = spinbox_['name']
            func = spinbox_['func']
            spinbox = self.findChild(QDoubleSpinBox, name)
            if spinbox:
                if isinstance(func, list):
                    for f in func:
                        spinbox.valueChanged.connect(f)
                else:
                    spinbox.valueChanged.connect(func)

        if self.findChild(QLineEdit, 'lineEdit_wavelength') is not None:
            self.timer_wavemeter = QtCore.QTimer()
            self.timer_wavemeter.setInterval(500)  # Interval in milliseconds
            self.timer_wavemeter.timeout.connect(self.read_wavemeter)
            self.timer_wavemeter.start()
            # real time wavemeter read

        if self.findChild(QLineEdit, 'lineEdit_X') is not None:
            self.timer_scanner = QtCore.QTimer()
            self.timer_scanner.setInterval(200)  # Interval in milliseconds
            self.timer_scanner.timeout.connect(self.read_scanner)
            self.timer_scanner.start()
            # real time scanner read

    def _find_widget(self, name, classes):

        for cls in classes:
            widget = self.findChild(cls, name)
            if widget:
                return widget
        return None
        

    def load_default(self):
        # load all params in plot() or default values into GUI slots
        # also load only valid options for counter_mode, data_mode, relim_mode etc.
        # also names in GUI labels


        WIDGET_CLASSES = {
            'setText': [QPushButton, QLabel],
            'setValue': [QDoubleSpinBox],
        }

        widget_config = {
            'setText': [
                {'name': 'pushButton_start_PLE', 'value': lambda: f'Start {self.measurement_PLE.measurement_name}'},
                {'name': 'pushButton_stop_PLE', 'value': lambda: f'Stop {self.measurement_PLE.measurement_name}'},
                {'name': 'pushButton_wavelength', 'value': lambda: f'Read \n {self.measurement_PLE.x_name}'},
                {'name': 'label_wl', 'value': lambda: f'Min ({self.measurement_PLE.x_unit})'},
                {'name': 'label_wu', 'value': lambda: f'Max ({self.measurement_PLE.x_unit})'},
                {'name': 'label_step_PLE', 'value': lambda: f'Step ({self.measurement_PLE.x_unit})'},
                {'name': 'label_device', 'value': lambda: f'{self.measurement_PLE.x_device_name}'},
                {'name': 'pushButton_save_PLE', 'value': lambda: f'Save \n {self.measurement_PLE.measurement_name}'},
                {'name': 'label_relim_PLE', 'value': lambda: f'Relim {self.measurement_PLE.measurement_name}'},
                # PLE
            ],
            'setValue': [
                {'name': 'doubleSpinBox_repeat', 'value': lambda: 1},
                {'name': 'doubleSpinBox_wl', 'value': lambda: self.measurement_PLE.data_x[0]},
                {'name': 'doubleSpinBox_wu', 'value': lambda: self.measurement_PLE.data_x[-1]},
                {'name': 'doubleSpinBox_exposure_PLE', 'value': lambda: self.measurement_PLE.exposure},
                {'name': 'doubleSpinBox_step_PLE', 'value': lambda: abs(self.measurement_PLE.data_x[0] - self.measurement_PLE.data_x[1])},
                {'name': 'doubleSpinBox_wavelength', 'value': lambda: self.measurement_PLE.data_x[0]},
                # PLE
                {'name': 'doubleSpinBox_xl', 'value': lambda: self.measurement_PL.data_x[0][0]},
                {'name': 'doubleSpinBox_xu', 'value': lambda: self.measurement_PL.data_x[0][-1]},
                {'name': 'doubleSpinBox_yl', 'value': lambda: self.measurement_PL.data_x[1][0]},
                {'name': 'doubleSpinBox_yu', 'value': lambda: self.measurement_PL.data_x[1][-1]},
                {'name': 'doubleSpinBox_exposure_PL', 'value': lambda: self.measurement_PL.exposure},
                {'name': 'doubleSpinBox_step_PL', 'value': lambda: abs(self.measurement_PL.data_x[0][0] - self.measurement_PL.data_x[0][1])},
                # PL
            ]
        }


        operation_map = {
            'setText': {'classes': WIDGET_CLASSES['setText'], 'method': 'setText'},
            'setValue': {'classes': WIDGET_CLASSES['setValue'], 'method': 'setValue'},
        }

        for action, widgets in widget_config.items():
            config = operation_map.get(action)
            classes = config['classes']
            method = config['method']

            for widget in widgets:
                name = widget['name']
                value_func = widget['value']
                widget_obj = self._find_widget(name, classes)
                if widget_obj:
                    value = value_func()
                    setter = getattr(widget_obj, method, None)
                    setter(value)


        self.valid_relim_mode = ['normal', 'tight']
        self.valid_fit_func = ['lorent', 'decay', 'rabi']
        if self.findChild(QComboBox, 'comboBox_relim_PL') is not None:
            self.comboBox_relim_PL.addItems(self.valid_relim_mode)
            self.comboBox_relim_PL.setCurrentText(self.measurement_PL.relim_mode)
        if self.findChild(QComboBox, 'comboBox_relim_PLE') is not None:
            self.comboBox_relim_PLE.addItems(self.valid_relim_mode)
            self.comboBox_relim_PLE.setCurrentText(self.measurement_PLE.relim_mode)
            self.comboBox_fit_func.addItems(self.valid_fit_func)
            self.comboBox_fit_func.setCurrentText(self.measurement_PLE.fit_func)
        self.comboBox_relim_Live.addItems(self.valid_relim_mode)
        self.comboBox_relim_Live.setCurrentText('normal')
        self.comboBox_counter_mode.addItems(self.measurement_Live.counter.valid_counter_mode)
        self.comboBox_data_mode.addItems(self.measurement_Live.counter.valid_data_mode)
        if self.findChild(QComboBox, 'comboBox_relim_PLE') is not None:
            self.comboBox_counter_mode.setCurrentText(self.measurement_PLE.counter_mode)
            self.comboBox_data_mode.setCurrentText(self.measurement_PLE.data_mode)
            self.comboBox_device_gui.addItems(list(self.config_instances.keys()))
            # display all available device instances
        else:
            self.comboBox_counter_mode.setCurrentText(self.measurement_PL.counter_mode)
            self.comboBox_data_mode.setCurrentText(self.measurement_PL.data_mode)
            self.comboBox_device_gui.addItems(list(self.config_instances.keys()))

        # set all combo box
        if self.findChild(QPushButton, 'pushButton_reattach') is not None:
            self.pushButton_reattach.setEnabled(False)  # Initially disabled
        # Placeholder for detached window
        self.detached_window = None

        current_time = time.localtime()
        current_date = time.strftime("%Y-%m-%d", current_time)
        time_str = current_date.replace('-', '_')
        self.lineEdit_save.setText(f'{time_str}/')
        self.estimate_PLE_time()
        self.estimate_PL_time()
        self.update_stabilizer_step()

    def load_file(self):
        from Confocal_GUI.live_plot import LoadAcquire
        import glob
        self.stop_plot()
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,'select','','data_figure (*.jpg *.npz)',options=options)

        if fileName == '':
            return
        files_all = glob.glob(fileName[:-4] + '*')# includes .jpg, .npz etc.
        files = []
        for file in files_all:
            if '.npz' in file:
                files.append(file)
        if len(files) > 1:
            print(files)
        if len(files) == 0:
            self.is_error = True
            return
        data_generator = LoadAcquire(address=files[0])
        plot_type = data_generator.info.get('plot_type', 'None')



        if plot_type == 'PLE':
            if self.measurement_PLE is not None:
                data_figure = DataFigure(None, address = files[0][:-4] + '*', fig=self.canvas_PLE.fig)
                figure_title = (files[0][:-4]).split('\\')[-1]
                data_figure.fig.axes[0].set_title(f'{figure_title}')
                data_figure.fig.tight_layout()
                data_figure.fig.canvas.draw()
                self.data_figure_PLE = data_figure
            else:
                self.print_log(f'Cannot load PLE plot_type')
                return
        elif plot_type == 'PL':
            if self.measurement_PL is not None:
                data_figure = DataFigure(None, address = files[0][:-4] + '*', fig=self.canvas_PL.fig)
                figure_title = (files[0][:-4]).split('\\')[-1]
                data_figure.fig.axes[1].set_title(f'{figure_title}')
                data_figure.fig.tight_layout()
                data_figure.fig.canvas.draw()
                self.data_figure_PL = data_figure
            else:
                self.print_log(f'Cannot load PL plot_type')
                return
        else:
            self.print_log(f'Cannot load unknown plot_type')
            return

        self.pushButton_load.setText(f'{figure_title}')

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


    def is_stabilizer(self):

        if not self.checkBox_is_stabilizer.isChecked():
            self.measurement_PLE.to_final_state()
            return
        # else, stabilizer is checked
        self.wavelength = self.doubleSpinBox_wavelength.value()
        self.measurement_PLE.device_to_state(self.wavelength)



    def open_device_gui(self):
        self.stop_and_show()
        device_handle = self.comboBox_device_gui.currentText()
        device_instance = self.config_instances[device_handle]
        if hasattr(device_instance, 'gui'):
            if device_handle == 'pulse':
                self.stop_and_show()
                device_instance.gui(is_in_GUI=True)
                # need different way to open a QWindow instance, and reclose plot to wait a little longer
            else:
                device_instance.gui()
        

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



    def read_wavemeter(self):
        wavelength = self.measurement_PLE.read_x()
        self.lineEdit_wavelength.setText(f'{wavelength:.5f}')





    def init_widget(self):
        current_time = time.localtime()
        current_date = time.strftime("%Y-%m-%d", current_time)
        time_str = current_date.replace('-', '_')

        self.lineEdit_save.setText(f'{time_str}/')

        self.estimate_PL_time()


        
    def print_log(self, text, is_print_jupyter = True):
        self.lineEdit_print.setText(text)
        if is_print_jupyter:
            print(text)
        # out put in jupyter cell
        
                
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
        
        
 

    def read_data_Live(self):
        for attr in ['exposure_Live', 'many']:
            value = getattr(self, f'doubleSpinBox_{attr}').value()
            setattr(self, attr, value)

        for attr in ['relim_Live', 'counter_mode', 'data_mode']: # Read from GUI panel
            value = getattr(self, f'comboBox_{attr}').currentText()
            setattr(self, attr, value)
            
            

  


    def start_plot_Live(self):
        self.print_log(f'Live started')
        self.cur_plot = 'Live'
        self.stop_plot()
        self.is_running = True
        self.read_data_Live()
        
        
        data_x = np.arange(self.many)

        self.measurement_Live.load_params(data_x=data_x, exposure=self.exposure_Live, repeat=self.repeat,\
            counter_mode=self.counter_mode, data_mode=self.data_mode, relim_mode=self.relim_Live)
        data_y = self.measurement_Live.data_y
        self.live_plot_Live = PLELive(labels=['Data', f'Counts/{self.exposure_Live:.2f}s'], \
                            update_time=0.01, data_generator=self.measurement_Live, data=[data_x, data_y],\
                                         fig=self.canvas_Live.fig, config_instances=self.config_instances, relim_mode = self.relim_Live)
        
        
        self.live_plot_Live.init_figure_and_data()
        
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int(1000*self.live_plot_Live.update_time))  # Interval in milliseconds
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
            
            
        
    def update_plot(self):
        
        attr = self.cur_plot
        live_plot_handle = getattr(self, f'live_plot_{attr}')

        if live_plot_handle.data_generator.thread.is_alive() and self.is_running:

            live_plot_handle.update_figure()

            self.estimate_PL_time()
            self.estimate_PLE_time()
            # update estimate finish time
        else:
            self.timer.stop()

            live_plot_handle.update_figure()  
            live_plot_handle.line.set_animated(False)                
            live_plot_handle.axes.set_autoscale_on(False)     
            live_plot_handle.choose_selector()
            live_plot_handle.stop()
            setattr(self, f'data_figure_{attr}', DataFigure(live_plot_handle))
            self.is_running = False


            self.estimate_PL_time()
            self.estimate_PLE_time()
            if self.findChild(QCheckBox, 'checkBox_is_stabilizer') is not None:
                self.checkBox_is_stabilizer.setDisabled(False)
                self.doubleSpinBox_wavelength.setDisabled(False)
            
    
    def stop_plot(self):
        
        self.is_running = False
        self.is_fit = False
        attr = self.cur_plot
        
        setattr(self, f'data_figure_{attr}', None) #disable DataFigure
        
        if hasattr(self, 'timer'):
            self.timer.stop()
            
            
        if hasattr(self, 'live_plot_PL') and self.live_plot_PL is not None:
            self.live_plot_PL.stop()

        if hasattr(self, 'live_plot_Live') and self.live_plot_Live is not None:
            self.live_plot_Live.stop()

        if hasattr(self, 'live_plot_PLE') and self.live_plot_PLE is not None:
            self.live_plot_PLE.stop()
            
            
            
            
            
    def stop_and_show(self):
        # stop acquiring data but enable save, fit and selector
        self.print_log(f'Plot stopped')
        self.is_running = False

        if hasattr(self, 'live_plot_PLE'):
            self.checkBox_is_stabilizer.setDisabled(False)
            self.doubleSpinBox_wavelength.setDisabled(False)
        if hasattr(self, 'live_plot_PL'):
            self.checkBox_is_bind.setChecked(False)
            self.checkBox_is_bind.setDisabled(False)

        if hasattr(self, 'live_plot_PL') and self.live_plot_PL is not None:
            self.estimate_PL_time()

        if hasattr(self, 'live_plot_PLE') and self.live_plot_PLE is not None:
            self.estimate_PLE_time()


    def bind_set(self):
        if self.checkBox_is_bind.isChecked():
            self.move_scanner()


    def read_scanner(self):
        x, y = self.measurement_PL.read_x()

        self.lineEdit_X.setText(f'{x}')
        self.lineEdit_Y.setText(f'{y}')



    def estimate_PL_time(self):
        if self.findChild(QLineEdit, 'lineEdit_time_PL') is None:
            return
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





    def move_scanner(self):
        x = self.doubleSpinBox_X.value()
        y = self.doubleSpinBox_Y.value()

        self.measurement_PL.device_to_state((x, y))

        
    def read_xy(self):
        if self.data_figure_PL.selector == []:
            return
        _xy = self.data_figure_PL.selector[1].xy #cross selector
        
        if _xy is not None:
            self.doubleSpinBox_X.setValue(_xy[0])
            self.doubleSpinBox_Y.setValue(_xy[1])
            self.print_log(f'read x = {_xy[0]}, y = {_xy[1]}')
        else:
            self.print_log(f'read x = None, y = None')
                

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
        
        
    def read_range_PL(self):
        if self.data_figure_PL.selector == []:
            self.print_log(f'no area to read range')
            return
        xl, xh, yl, yh = self.data_figure_PL.selector[0].range
        
        if xl is None:
            self.print_log(f'no area to read range')
            return
        
        self.doubleSpinBox_xl.setValue(xl)
        self.doubleSpinBox_xu.setValue(xh)
        self.doubleSpinBox_yl.setValue(yl)
        self.doubleSpinBox_yu.setValue(yh)

        self.print_log(f'PL range updated')
        
  
            
    def read_data_PL(self):
        for attr in ['exposure_PL', 'xl', 'xu', 'yl', 'yu', 'step_PL', 'repeat']:
            value = getattr(self, f'doubleSpinBox_{attr}').value()
            setattr(self, attr, value)

        for attr in ['relim_PL', 'counter_mode', 'data_mode']: # Read from GUI panel
            value = getattr(self, f'comboBox_{attr}').currentText()
            setattr(self, attr, value)

        self.repeat = int(self.repeat)
            
            
    def start_plot_PL(self):
        self.print_log(f'PL started')
        self.cur_plot = 'PL'
        self.stop_plot()
        self.is_running = True
            
            
        self.read_data_PL()

        data_x = np.array([np.arange(self.xl, self.xu, self.step_PL), np.arange(self.yl, self.yu, self.step_PL)])

        self.measurement_PL.load_params(data_x = data_x, exposure=self.exposure_PL, repeat=self.repeat,\
            counter_mode=self.counter_mode, data_mode=self.data_mode, relim_mode=self.relim_PL)
                

        if self.findChild(QCheckBox, 'checkBox_is_stabilizer') is not None:
            self.checkBox_is_stabilizer.setDisabled(True)
            self.doubleSpinBox_wavelength.setDisabled(True)

        self.checkBox_is_bind.setChecked(False)
        self.checkBox_is_bind.setDisabled(True)


        data_y = self.measurement_PL.data_y
        self.live_plot_PL = PLGUILive(labels=[['X', 'Y'], f'Counts/{self.exposure_PL:.2f}s'], \
                        update_time=1, data_generator=self.measurement_PL, data=[data_x, data_y],\
                                       fig=self.canvas_PL.fig, config_instances=self.config_instances)
        
        
        self.live_plot_PL.init_figure_and_data()
        
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000*self.live_plot_PL.update_time)  # Interval in milliseconds
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()


    def step_PLE_in_MHz(self):
        if self.measurement_PLE.x_unit == 'nm':
            step_in_MHz = 1000*np.abs(self.spl/((self.wl + self.wu)/2) - self.spl/((self.wl + self.wu)/2 + self.step_PLE))
            self.lineEdit_step_PLE.setText(f'{step_in_MHz:.2f}MHz')

    def update_stabilizer_step(self):
        if self.findChild(QDoubleSpinBox, 'doubleSpinBox_wavelength') is None:
            return
        self.read_data_PLE()
        if self.step_PLE != 0:
            self.doubleSpinBox_wavelength.setSingleStep(self.step_PLE)
        else:
            self.doubleSpinBox_wavelength.setSingleStep(self.doubleSpinBox_wavelength.value()/100)


    def estimate_PLE_time(self):
        if self.findChild(QLineEdit, 'lineEdit_time_PLE') is None:
            return
        if self.is_running and self.cur_plot=='PLE':
            points_total = self.live_plot_PLE.points_total
            points_done = self.live_plot_PLE.points_done
            ratio = points_done/points_total
            self.lineEdit_time_PLE.setText(f'{self.measurement_PLE.measurement_name} \
                finishes in {(ratio*self.time_PLE):.2f}s / {self.time_PLE:.2f}s, {ratio*100:.2f}%')

        else:
            self.read_data_PLE()
            time = (self.exposure_PLE + 0.5)* len(np.arange(self.wl, self.wu, self.step_PLE)) 
            # considering the overhead of stabilizing laser frequency
            self.lineEdit_time_PLE.setText(f'new {self.measurement_PLE.measurement_name} finishes in {time:.2f}s')
            self.time_PLE = time


    def read_wavelength(self):
        if self.data_figure_PLE.selector == []:
            self.print_log(f'No {self.measurement_PLE.x_name} to read')
            return
        _wavelength = self.data_figure_PLE.selector[1].wavelength
        
        if _wavelength is not None:
            self.doubleSpinBox_wavelength.setValue(_wavelength)
            self.print_log(f'{self.measurement_PLE.x_name} was read')
        else:
            self.print_log(f'No {self.measurement_PLE.x_name} to read')
        #set double spin box to _wavelength
        
        
    def save_PLE(self):
        if self.data_figure_PLE is None:
            return
        addr = self.lineEdit_save.text()
        self.data_figure_PLE.save(addr = addr) 

        info = self.data_figure_PLE.info
        self.print_log(f'info: {info}')

        if self.is_save_to_jupyter_flag:
            self.save_to_jupyter(self.canvas_PLE.fig)
            
        self.print_log(f'saved {self.measurement_PLE.measurement_name}')
        
        
        
    def fit_func(self):

        self.fit_func = self.comboBox_fit_func.currentText()
        if self.data_figure_PLE is None:
            self.print_log(f'No figure to fit')
            return
        if self.is_fit:
            self.data_figure_PLE.clear()
            self.is_fit = False
            self.print_log(f'fit cleared')
        else:
            eval(f'self.data_figure_PLE.{self.fit_func}()')
            self.is_fit = True
            log_info = self.data_figure_PLE.log_info
            self.print_log(f'curve fitted, {log_info}')

    def change_unit(self):
        if self.data_figure_PLE is None:
            self.print_log(f'No figure to change unit')
            return
        if self.measurement_PLE.x_unit == 'nm':
            if self.data_figure_PLE.unit == 'nm':
                self.data_figure_PLE.to_GHz()
                self.print_log(f'changed unit to GHz')
            else:
                self.data_figure_PLE.to_nm()
                self.print_log(f'changed unit to nm')
        
        
        
    def read_range_PLE(self):
        if self.data_figure_PLE.selector == []:
            self.print_log(f'no area to read range')
            return
        xl, xh, yl, yh = self.data_figure_PLE.selector[0].range
        
        if xl is None:
            self.print_log(f'no area to read range')
            return
        

        self.doubleSpinBox_wl.setValue(xl)
        self.doubleSpinBox_wu.setValue(xh)


        self.print_log(f'{self.measurement_PLE.measurement_name} range updated')
        
        
    def read_data_PLE(self):
        for attr in ['exposure_PLE', 'wl', 'wu', 'step_PLE', 'repeat']: # Read from GUI panel
            value = getattr(self, f'doubleSpinBox_{attr}').value()
            setattr(self, attr, value)

        for attr in ['relim_PLE', 'counter_mode', 'data_mode']: # Read from GUI panel
            value = getattr(self, f'comboBox_{attr}').currentText()
            setattr(self, attr, value)

        self.repeat = int(self.repeat)


    def start_plot_PLE(self):
        self.print_log(f'{self.measurement_PLE.measurement_name} started')
        self.cur_plot = 'PLE'
        self.stop_plot()
        self.is_running = True
        self.pushButton_load.setText(f'Load file:')
                    
        
        self.read_data_PLE()
        
        data_x = np.arange(self.wl, self.wu, self.step_PLE)

        self.checkBox_is_stabilizer.setChecked(False)
        self.checkBox_is_stabilizer.setDisabled(True)
        self.doubleSpinBox_wavelength.setDisabled(True)


        self.measurement_PLE.load_params(data_x=data_x, exposure=self.exposure_PLE, repeat=self.repeat, \
            counter_mode=self.counter_mode, data_mode=self.data_mode, relim_mode=self.relim_PLE)

        data_y = self.measurement_PLE.data_y
        self.live_plot_PLE = PLELive(labels=[f'{self.measurement_PLE.x_name} ({self.measurement_PLE.x_unit})', f'Counts/{self.exposure_PLE:.2f}s'], 
                                     update_time=1, data_generator=self.measurement_PLE, data=[data_x, data_y],\
                                    fig=self.canvas_PLE.fig, config_instances=self.config_instances, relim_mode = self.relim_PLE)
        

        self.live_plot_PLE.init_figure_and_data()
        
        
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000*self.live_plot_PLE.update_time)  # Interval in milliseconds
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        
            

            
    def closeEvent(self, event):
        self.stop_plot()
        plt.close('all') # make sure close all plots which avoids error message

            
        event.accept()
        QtWidgets.QApplication.quit()  # Ensure application exits completely



def GUI_(config_instances, measurement_PLE=None, measurement_PL=None, measurement_Live=None, mode='PL_and_PLE'):
    """
    The function opens pyqt GUI for PLE, PL, live counts, and pulse control.
    Save button will also output data and figure to jupyter notebook.
   
    Examples
    --------
    >>> GUI(config_instances = config_instances)

    Read range button reads range from area created by left mouse event 

	Read {variable_name} button reads {variable} from point created by right mouse event

    Scroll to zoom in or zoom out figure

    """

    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        
    
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    if mode == 'PL_and_PLE':
        w = MainWindow(config_instances, measurement_PLE=measurement_PLE, measurement_PL=measurement_PL\
            , measurement_Live=measurement_Live, ui='GUI.ui')
    elif mode == 'PLE_and_Live':
        w = MainWindow(config_instances, measurement_PLE=measurement_PLE, measurement_PL=measurement_PL\
            , measurement_Live=measurement_Live, ui='PLE.ui')
    elif mode == 'PL_and_Live':
        w = MainWindow(config_instances, measurement_PLE=measurement_PLE, measurement_PL=measurement_PL\
            , measurement_Live=measurement_Live, ui='PL.ui')

    app.setStyle('Windows')
    try:
        sys.exit(app.exec_())
    except SystemExit as se:
        if se.code != 0:
            raise se
    # make sure jupyter notebook does not catch error when exit normally



def GUI_PLE(config_instances, measurement_PLE, measurement_Live):
    """
    GUI for PLE type measurement

    """
    GUI_(config_instances = config_instances, measurement_PLE=measurement_PLE, \
        measurement_PL=None, measurement_Live=measurement_Live, mode='PLE_and_Live')





def GUI_PL(config_instances, measurement_PL, measurement_Live):
    """
    GUI for PL type measurement

    """

    GUI_(config_instances = config_instances, measurement_PL=measurement_PL, \
        measurement_PLE=None, measurement_Live=measurement_Live, mode='PL_and_Live')


class DeviceGUI(QDialog):
    def __init__(self, device, parent=None):
        super().__init__(parent)
        self.device = device
        self.device_name = device.__class__.__name__
        self.setWindowTitle(f'{self.device_name} Control')
        self.init_ui()

    def init_ui(self):
        font = QFont("Arial", 12)
        self.setFont(font)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.controls = {}

        # Iterate over each property to create corresponding controls
        for prop, prop_type in zip(self.device.gui_property, self.device.gui_property_type):
            prop_layout = QHBoxLayout()

            # Label for the property
            label = QLabel(prop.capitalize())
            label.setFont(font)
            label.setFixedWidth(100)
            label.setFixedHeight(30)

            # Current value display (read-only)
            current_value = QLineEdit()
            current_value.setEnabled(False)
            current_value.setMaxLength(10)
            current_value.setFont(font)
            current_value.setFixedWidth(100)
            current_value.setFixedHeight(30)

            # Determine the input control based on property type
            if prop_type == 'float':
                input_control = QDoubleSpinBox()
                input_control.setFont(font)
                input_control.setFixedWidth(100)
                input_control.setFixedHeight(30)

                # Dynamically get the lower and upper bounds from the device
                lb_attr = f"{prop}_lb"
                ub_attr = f"{prop}_ub"
                try:
                    lb = getattr(self.device, lb_attr)
                    ub = getattr(self.device, ub_attr)
                    input_control.setRange(lb, ub)
                except AttributeError:
                    input_control.setRange(-1e9, 1e9)  # Default range if bounds not defined

                # Optionally, set single step and decimals
                input_control.setSingleStep((ub - lb) / 100 if 'lb' in locals() and 'ub' in locals() else 1)
                input_control.setDecimals(3)
            elif prop_type == 'str':
                input_control = QComboBox()
                input_control.setFont(font)
                input_control.setFixedWidth(100)
                input_control.setFixedHeight(30)
                input_control.addItems(['True', 'False'])
            else:
                # For unsupported types, default to QLineEdit
                input_control = QLineEdit()
                input_control.setFont(font)
                input_control.setFixedWidth(100)
                input_control.setFixedHeight(30)

            # Apply button
            apply_button = QPushButton("Apply")
            apply_button.setFont(font)
            apply_button.setFixedWidth(80)
            apply_button.setFixedHeight(30)

            # Connect the apply button to the appropriate handler
            if prop_type == 'float':
                apply_button.clicked.connect(
                    lambda checked, p=prop, i=input_control, c=current_value: self.apply_value(p, i, c)
                )
            elif prop_type == 'str':
                apply_button.clicked.connect(
                    lambda checked, p=prop, i=input_control, c=current_value: self.apply_value(p, i, c)
                )
            else:
                apply_button.clicked.connect(
                    lambda checked, p=prop, i=input_control, c=current_value: self.apply_value(p, i, c)
                )

            # Add widgets to the property layout
            prop_layout.addWidget(label)
            prop_layout.addWidget(QLabel("Current:"))
            prop_layout.addWidget(current_value)
            prop_layout.addWidget(QLabel("Set:"))
            prop_layout.addWidget(input_control)
            prop_layout.addWidget(apply_button)

            # Add the property layout to the main layout
            main_layout.addLayout(prop_layout)

            # Store the controls for later access
            self.controls[prop] = {
                'current_value': current_value,
                'input_control': input_control,
                'apply_button': apply_button,
                'type': prop_type
            }

            # Initialize the current value display
            self.update_property_value(prop)

        # Adjust dialog size based on the number of properties
        height = 150 + 60 * len(self.device.gui_property)
        self.setFixedSize(700, height)

    def update_property_value(self, prop):
        value = getattr(self.device, prop)
        if self.controls[prop]['type'] == 'str':
            value_str = 'True' if value else 'False'
            self.controls[prop]['current_value'].setText(value_str)
        else:
            self.controls[prop]['current_value'].setText(f'{value}')

    def apply_value(self, prop, input_control, current_label: QLineEdit):
        prop_type = self.controls[prop]['type']
        try:
            if prop_type == 'float':
                new_value = float(input_control.value())
            elif prop_type == 'str':
                new_value = input_control.currentText() == 'True'
            else:
                new_value = input_control.text()

            setattr(self.device, prop, new_value)
            self.update_property_value(prop)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to set {prop}: {e}")




def GUI_Device(device_handle):
    """
    GUI for devices,

    device_handle is instance of one of BaseDevice

    """




    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setStyle('Windows')
    dialog = DeviceGUI(device_handle)
    dialog.exec_()


class PulseGUI(QMainWindow):

    def __init__(self, device_handle, parent=None):
        super().__init__()
        font = QFont('Arial', 10) 
        self.setFont(font)
        self.setWindowTitle('PulseGUI@Wanglab, UOregon')
        self.device_handle = device_handle
        self.channel_names_map = [f'Ch{channel}' for channel in range(8)]
        
        self.widget = QWidget()
        self.layout = QVBoxLayout(self.widget)
        self.widget_button = QWidget()
        self.layout.addWidget(self.widget_button)
        self.widget_dataset = QWidget()
        self.layout.addWidget(self.widget_dataset)
        self.widget_saveload = QWidget()
        self.layout.addWidget(self.widget_saveload)
        self.layout_button = QHBoxLayout(self.widget_button)
        self.layout_dataset = QHBoxLayout(self.widget_dataset)
        self.layout_saveload = QHBoxLayout(self.widget_saveload)
        
        self.btn1 = QPushButton('Off Pulse')
        self.btn1.setFixedSize(150,100)
        self.btn1.clicked.connect(self.off_pulse)
        self.layout_button.addWidget(self.btn1)
        
        self.btn1 = QPushButton('On Pulse')
        self.btn1.setFixedSize(150,100)
        self.btn1.clicked.connect(self.on_pulse)
        self.layout_button.addWidget(self.btn1)
        
        self.btn2 = QPushButton('Remove Column')
        self.btn2.setFixedSize(150,100)
        self.btn2.clicked.connect(self.remove_column)
        self.layout_button.addWidget(self.btn2)
        
        self.btn3 = QPushButton('Add Column')
        self.btn3.setFixedSize(150,100)
        self.btn3.clicked.connect(self.add_column)
        self.layout_button.addWidget(self.btn3)

        self.btn4 = QPushButton('Save Pulse')
        self.btn4.setFixedSize(150,100)
        self.btn4.clicked.connect(self.save_data)
        self.layout_button.addWidget(self.btn4)
        
        self.setCentralWidget(self.widget)

        self.load_data()

        self.btn3 = QPushButton('Save to file')
        self.btn3.setFixedSize(150,100)
        self.btn3.clicked.connect(self.save_to_file)
        self.layout_saveload.addWidget(self.btn3)

        self.btn4 = QPushButton('Load from file')
        self.btn4.setFixedSize(150,100)
        self.btn4.clicked.connect(self.load_from_file)
        self.layout_saveload.addWidget(self.btn4)

        self.show()


    def off_pulse(self):


        # Create a sequence object
        self.device_handle.off_pulse()
        # need to repeat 8 times because pulse streamer will pad sequence to multiple of 8ns, otherwise unexpected changes of pulse


    def on_pulse(self):

        self.device_handle.data_matrix = self.read_data()
        self.device_handle.delay_array = self.read_delay()
        self.device_handle.channel_names = self.read_channel_names()
        self.device_handle.on_pulse()

    def handle_text_change(self, text, combo_box):
        
        if text.isdigit() or (len(text)>=2 and (text[0] in ['-', '+']) and text[1:].isdigit()):
            combo_box.setEnabled(True)
            if combo_box.currentText() == 'str (ns)':
                combo_box.setCurrentText('ns')
        else:
            combo_box.setCurrentText('str (ns)')
            combo_box.setEnabled(False)


    def load_data(self, is_read_tmp=True):
        """
        load self.device_handle.delay_array, self.device_handle.data_matrix back to GUI
        to recover GUI state after reopen GUI window
        """

        while self.layout_dataset.count() > 0:
            item = self.layout_dataset.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        # remove all column is exists

        if is_read_tmp:
            self.delay_array = self.device_handle.delay_array_tmp
            self.data_matrix = self.device_handle.data_matrix_tmp
            self.channel_names = self.device_handle.channel_names_tmp
        else:
            self.delay_array = self.device_handle.delay_array
            self.data_matrix = self.device_handle.data_matrix
            self.channel_names = self.device_handle.channel_names


        self.add_channel_names()
        name_widget = self.layout_dataset.itemAt(0).widget()
        name_layout = name_widget.layout()

        for j, name_value in enumerate(self.channel_names):
            sublayout = name_layout.itemAt(j)
            line_edit = sublayout.itemAt(1).widget() 
            line_edit.setText(str(name_value))


        self.add_delay()
        delay_widget = self.layout_dataset.itemAt(1).widget()
        delay_layout = delay_widget.layout()

        for j, delay_value in enumerate(self.delay_array):
            sublayout = delay_layout.itemAt(j)

            line_edit = sublayout.itemAt(1).widget()
            combo_box = sublayout.itemAt(2).widget()
            
            if isinstance(delay_value, str):
                combo_box.setCurrentText('str (ns)')
            else:
                combo_box.setCurrentText('ns')  
            line_edit.setText(str(delay_value))

        for i, row_data in enumerate(self.data_matrix):
            self.add_column()
            row_widget = self.layout_dataset.itemAt(i + 2).widget()
            row_layout = row_widget.layout()

            sublayout = row_layout.itemAt(0)  
            line_edit = sublayout.itemAt(1).widget() 
            combo_box = sublayout.itemAt(2).widget()  
                
            if isinstance(row_data[0], str):
                combo_box.setCurrentText('str (ns)')
            else:
                combo_box.setCurrentText('ns')
            line_edit.setText(str(row_data[0]))


            for j in range(1, 9):  
                checkbox = row_layout.itemAt(j).widget()
                checkbox.setChecked(bool(row_data[j]))


    def save_data(self):
        """
        save GUI state to self.device_handle.delay_array, self.device_handle.data_matrix 
        """
        self.device_handle.data_matrix = self.read_data()
        self.device_handle.delay_array = self.read_delay()
        self.device_handle.channel_names = self.read_channel_names()

        
    def print_index(self):
        count = self.layout_dataset.count()
        for i in range(count):
            item = self.layout_dataset.itemAt(i)
            widget = item.widget()
            layout = widget.layout()
            for j in range(1):
                item_sub = layout.itemAt(j)
                layout_sub = item_sub.layout()
                duration_num = layout_sub.itemAt(1).widget().text()
                duration_unit = layout_sub.itemAt(2).widget().currentText()
                print(duration_num, duration_unit)
            for j in range(1,5):
                item = layout.itemAt(j)
                widget = item.widget()
                if(widget.isChecked()):
                    print((i,j))
                    
    def read_data(self):
        count = self.layout_dataset.count()-2  #number of pulses 
        data_matrix = [[0]*9 for _ in range(count)] #skip first delay layout

        for i in range(count):
            item = self.layout_dataset.itemAt(i+2)#first is delay
            widget = item.widget()
            layout = widget.layout()
            for j in range(1):
                item_sub = layout.itemAt(j)
                layout_sub = item_sub.layout()
                duration_unit = layout_sub.itemAt(2).widget().currentText()
                if(duration_unit == 'str (ns)'):
                    duration_num = layout_sub.itemAt(1).widget().text()
                elif(duration_unit == 'ns'):
                    duration_num = int(layout_sub.itemAt(1).widget().text())
                    duration_num *= 1
                elif(duration_unit == 'us'):
                    duration_num = int(layout_sub.itemAt(1).widget().text())
                    duration_num *= 1000
                elif(duration_unit == 'ms'):
                    duration_num = int(layout_sub.itemAt(1).widget().text())
                    duration_num *= 1000000
                    
                data_matrix[i][j] = duration_num
                
            for j in range(1,9):
                item = layout.itemAt(j)
                widget = item.widget()
                if(widget.isChecked()):
                    data_matrix[i][j] = 1
        
        return data_matrix

    def read_delay(self):

        item = self.layout_dataset.itemAt(1)#second is delay
        widget = item.widget()
        layout = widget.layout()
        delay_array = [0, 0, 0, 0, 0, 0, 0, 0]
        for j in range(8):
            item_sub = layout.itemAt(j)
            layout_sub = item_sub.layout()
            duration_unit = layout_sub.itemAt(2).widget().currentText()
            if(duration_unit == 'str (ns)'):
                duration_num = layout_sub.itemAt(1).widget().text()
            elif(duration_unit == 'ns'):
                duration_num = int(layout_sub.itemAt(1).widget().text())
                duration_num *= 1
            elif(duration_unit == 'us'):
                duration_num = int(layout_sub.itemAt(1).widget().text())
                duration_num *= 1000
            elif(duration_unit == 'ms'):
                duration_num = int(layout_sub.itemAt(1).widget().text())
                duration_num *= 1000000
                
            delay_array[j] = duration_num

        return delay_array

    def read_channel_names(self):

        item = self.layout_dataset.itemAt(0)#first is name
        widget = item.widget()
        layout = widget.layout()
        channel_names = ['', '', '', '', '', '', '', '']
        for j in range(8):
            item_sub = layout.itemAt(j)
            layout_sub = item_sub.layout()
                
            channel_names[j] = layout_sub.itemAt(1).widget().text()

        return channel_names
    
        
    def remove_column(self):
        count = self.layout_dataset.count()
        #print(count)
        if(count>=5):
            item = self.layout_dataset.itemAt(count-1)
            widget = item.widget()
            widget.deleteLater()
            
    def add_column(self):
        count = self.layout_dataset.count()
        row = QGroupBox('Pulse%d'%(count-2))
        self.layout_dataset.addWidget(row)
        layout_data = QVBoxLayout(row)
        
        sublayout = QVBoxLayout()
        layout_data.addLayout(sublayout)
        btn = QLabel('Duration:')
        btn.setFixedSize(70,20)
        sublayout.addWidget(btn)
        btn = QLineEdit('10')
        btn.setFixedSize(70,20)
        sublayout.addWidget(btn)
        btn2 = QComboBox()
        btn2.addItems(['ns','us' ,'ms', 'str (ns)'])
        btn2.setFixedSize(70,20)
        sublayout.addWidget(btn2)
        btn.textChanged.connect(lambda text, cb=btn2: self.handle_text_change(text, cb))
        
        for index in range(1, 9):
            btn = QCheckBox()
            btn.setText(self.channel_names_map[index-1])
            btn.setCheckable(True)
            layout_data.addWidget(btn)
        
        
    def add_delay(self):
        count = self.layout_dataset.count()
        row = QGroupBox('Delay')
        self.layout_dataset.addWidget(row)
        layout_data = QVBoxLayout(row)
        
        for index in range(1, 9):
            sublayout = QHBoxLayout()
            layout_data.addLayout(sublayout)
            btn = QLabel(f'{self.channel_names_map[index-1]} delay:')
            btn.setFixedSize(70,20)
            sublayout.addWidget(btn)
            btn = QLineEdit('0')
            btn.setFixedSize(70,20)
            sublayout.addWidget(btn)
            btn2 = QComboBox()
            btn2.addItems(['ns','us' ,'ms', 'str (ns)'])
            btn2.setFixedSize(70,20)
            sublayout.addWidget(btn2)
            btn.textChanged.connect(lambda text, cb=btn2: self.handle_text_change(text, cb))

    def add_channel_names(self):
        count = self.layout_dataset.count()
        row = QGroupBox('Channel Names')
        self.layout_dataset.addWidget(row)
        layout_data = QVBoxLayout(row)
        
        for index in range(1, 9):
            sublayout = QHBoxLayout()
            layout_data.addLayout(sublayout)
            btn = QLabel(f'Ch{index-1} name:')
            btn.setFixedSize(70,20)
            sublayout.addWidget(btn)
            btn = QLineEdit('')
            btn.setFixedSize(70,20)
            btn.textChanged.connect(lambda text, channel=(index-1): self.replace_channel_names(text, channel))
            sublayout.addWidget(btn)

    def save_to_file(self):

        self.device_handle.data_matrix = self.read_data()
        self.device_handle.delay_array = self.read_delay()
        self.device_handle.channel_names = self.read_channel_names()

        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self,'select','','data_figure (*.npz)',options=options)

        if fileName == '':
            return

        if '.npz' in fileName:
            fileName = fileName[:-4]

        if '_pulse' in fileName:
            fileName = fileName[:-6]

        self.device_handle.save_to_file(addr = fileName)

    def load_from_file(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,'select','','data_figure (*.npz)',options=options)

        if fileName == '':
            return

        self.device_handle.load_from_file(addr = fileName[:-4]+'*')

        self.load_data(is_read_tmp=False)

    def replace_channel_names(self, text, channel):
        if text == '':
            self.channel_names_map[channel] = f'Ch{channel}'
        else:
            self.channel_names_map[channel] = text

        # make sure load works where channel_names loaded before all other

        item = self.layout_dataset.itemAt(1)#second is delay
        if item is not None:
            widget = item.widget()
            layout = widget.layout()
            item_sub = layout.itemAt(channel)
            layout_sub = item_sub.layout()
            layout_sub.itemAt(0).widget().setText(f'{self.channel_names_map[channel]} delay')
        # replace delay row


        count = self.layout_dataset.count()-2  #number of pulses 
        for i in range(count):
            item = self.layout_dataset.itemAt(i+2)#first is delay
            if item is not None:
                widget = item.widget()
                layout = widget.layout()
                item = layout.itemAt(channel+1)
                item.widget().setText(f'{self.channel_names_map[channel]}')

        # replace pulse row




    def closeEvent(self, event):
        self.device_handle.data_matrix_tmp = self.read_data()
        self.device_handle.delay_array_tmp = self.read_delay() 
        self.device_handle.channel_names_tmp = self.read_channel_names() 

def GUI_Pulse(device_handle, is_in_GUI=False):
    """
    GUI for pulse control

    device_handle is a BasePulse instance

    """

    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    w = PulseGUI(device_handle)
    app.setStyle('Windows')
    if is_in_GUI:
        w.show()
        # otherwise won't opne a QWindow in another QWindow
    else:
        try:
            sys.exit(app.exec_())
        except SystemExit as se:
            if se.code != 0:
                raise se