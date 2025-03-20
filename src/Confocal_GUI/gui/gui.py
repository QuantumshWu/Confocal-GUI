import os, sys, time, threading, io
import numpy as np
from decimal import Decimal
from threading import Event
import numbers

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

from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt, QSize, QMutex, QMutexLocker, QMimeData, QRegExp
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QRadioButton, QHBoxLayout, QVBoxLayout\
, QPushButton, QGroupBox, QCheckBox, QLineEdit, QComboBox, QLabel, QFileDialog, QDoubleSpinBox, QSizePolicy, QDialog, QMessageBox, QFrame
from PyQt5.QtGui import QPalette, QColor, QFont, QDrag, QPixmap, QRegExpValidator
from PyQt5 import QtCore, QtWidgets, uic

import Confocal_GUI.live_plot as live_plot_dir
from Confocal_GUI.live_plot import *

PULSES_INIT_DIR = os.path.dirname(os.path.dirname(live_plot_dir.__file__)) + '/pulses/'
FIGS_INIT_DIR = os.getcwd()


class MplCanvas(FigureCanvasQTAgg):
    """
    labels = [xlabel, ylabel]
    """
    def __init__(self, parent=None, labels=None, mode=None):

        change_to_inline()

        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(111)
        if labels is not None:
            self.axes.set_xlabel(labels[0])
            self.axes.set_ylabel(labels[1])
            
        if mode=='PL':
            
            line = self.axes.imshow(0*np.random.random((100, 100)), animated=True, cmap='inferno')
            cbar = self.fig.colorbar(line)
            cbar.set_label(labels[2])
        elif mode=='PLdis':
            
            divider = make_axes_locatable(self.axes)
            axdis = divider.append_axes("top", size="20%", pad=0.25)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            
            line = self.axes.imshow(0*np.random.random((100, 100)), animated=True, cmap='inferno')
            cbar = self.fig.colorbar(line, cax = cax)
            cbar.set_label(labels[2])
        elif mode=='PLE':
            self.axes.set_ylim(0, 1000)
            self.axes.set_xlim(0, 100)
        elif mode=='Live':
            divider = make_axes_locatable(self.axes)
            axdis = divider.append_axes("right", size="20%", pad=0.1, sharey=self.axes)
            axdis.tick_params(axis='y', labelleft=False)
            axdis.tick_params(axis='both', which='both',bottom=False,top=False)

            self.axes.set_ylim(0, 1000)
            self.axes.set_xlim(0, 100)

            
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

        self.init_size = self.size()
        self.setMaximumSize(self.init_size.width(), self.init_size.height())
        self.setMinimumSize(self.init_size.width(), self.init_size.height())

        self.widget.show()



    def closeEvent(self, event):
        self.parent_window.reattach_page()
        event.accept()




class MainWindow(QMainWindow):
    def __init__(self, measurement_PL, measurement_PLE, measurement_Live, ui='GUI.ui'):
        super().__init__()
        from Confocal_GUI.device import config_instances
        self.config_instances = config_instances
        self.data_figure_PL = None
        self.data_figure_PLE = None
        self.data_figure_Live = None
        self.cur_plot = 'PL'
        self.cur_live_plot = None
        self.spl = 299792458
        self.is_save_to_jupyter_flag = True
        self.time_PL = 0
        self.measurement_PL = measurement_PL
        self.measurement_PLE = measurement_PLE
        self.measurement_Live = measurement_Live
        self.pulse_PL = None
        self.pulse_PLE = None
        self.pulse_Live = None
        self.ui = ui
        self.overhead_PL = 0.01 # estimated 10ms overhead for single data point
        self.overhead_PLE = 0.5 # estimated 500ms overhead for single data point PLE
        ui_path = os.path.join(os.path.dirname(__file__), self.ui)
        uic.loadUi(ui_path, self)

        self.init_size = self.size()
        self.setMaximumSize(self.init_size.width(), self.init_size.height())
        self.setMinimumSize(self.init_size.width(), self.init_size.height())
        self.setWindowTitle(f'ConfocalGUI@Wanglab, UOregon' )
        # set size

        plt.close('all')
        # close all previous figures
        self.load_figures()
        self.load_default()
        self.connect_buttons()
        self.init_widget()
        self.show()

    def expand_settings_PL(self):
        if self.contentWidget_PL.isVisible():
            self.contentWidget_PL.setVisible(False)
            self.contentWidget_PL.lower()
        else:
            self.contentWidget_PL.raise_()
            self.contentWidget_PL.setVisible(True)
    def expand_settings_PLE(self):
        if self.contentWidget_PLE.isVisible():
            self.contentWidget_PLE.setVisible(False)
            self.contentWidget_PLE.lower()
        else:
            self.contentWidget_PLE.raise_()
            self.contentWidget_PLE.setVisible(True)
    def expand_settings_Live(self):
        if self.contentWidget_Live.isVisible():
            self.contentWidget_Live.setVisible(False)
            self.contentWidget_Live.lower()
        else:
            self.contentWidget_Live.raise_()
            self.contentWidget_Live.setVisible(True)



    def load_figures(self):
        PLE_x_label = '' if self.measurement_PLE is None else f'{self.measurement_PLE.x_name} ({self.measurement_PLE.x_unit})'
        PLE_y_label = '' if self.measurement_PLE is None else f'Counts/{self.measurement_PLE.exposure}s'
        PL_y_label = '' if self.measurement_PL is None else f'Counts/{self.measurement_PL.exposure}s'
        figures = [
            {'child_name': 'widget_figure_PL', 'labels': ['X', 'Y', PL_y_label], 'canvas_name': 'canvas_PL', 'mode': 'PLdis'},
            {'child_name': 'widget_figure_PLE', 'labels': [PLE_x_label, PLE_y_label], 'canvas_name': 'canvas_PLE', 'mode': 'PLE'},
            {'child_name': 'widget_figure_Live', 'labels': ['Data (1)', 'Counts'], 'canvas_name': 'canvas_Live', 'mode': 'Live'} 
        ]

        for fig in figures:
            widget = self.findChild(QWidget, fig['child_name'])
            if widget:
                layout = QVBoxLayout(widget)
                layout.setContentsMargins(0, 0, 0, 0)

                canvas = MplCanvas(widget, labels=fig['labels'], mode=fig['mode'])
                layout.addWidget(canvas)

                setattr(self, fig['canvas_name'], canvas)

        if self.findChild(QComboBox, 'comboBox_relim_PL') is not None:
            self.contentWidget_PL.setVisible(False)
            self.collapseButton_PL.clicked.connect(self.expand_settings_PL)
        if self.findChild(QComboBox, 'comboBox_relim_PLE') is not None:
            self.contentWidget_PLE.setVisible(False)
            self.collapseButton_PLE.clicked.connect(self.expand_settings_PLE)
        if self.findChild(QComboBox, 'comboBox_relim_Live') is not None:
            self.contentWidget_Live.setVisible(False)
            self.collapseButton_Live.clicked.connect(self.expand_settings_Live)


    def connect_buttons(self):

        widget_config = {
            'buttons': [
                {'name': 'pushButton_load', 'func': self.load_file},
                # for load file
                {'name': 'pushButton_start_PLE', 'func': self.start_plot_PLE},
                {'name': 'pushButton_stop_PLE', 'func': self.stop_plot},
                {'name': 'pushButton_lorent', 'func': self.fit_func_PLE},
                {'name': 'pushButton_save_PLE', 'func': self.save_PLE},
                {'name': 'pushButton_unit', 'func': self.change_unit},
                {'name': 'pushButton_load_pulse_PLE', 'func': self.choose_pulse},
                # for PLE
                {'name': 'pushButton_device_gui', 'func': self.open_device_gui},
                # for load device gui
                {'name': 'pushButton_start_Live', 'func': self.start_plot_Live},
                {'name': 'pushButton_stop_Live', 'func': self.stop_plot},
                {'name': 'pushButton_detach', 'func': self.detach_page},
                {'name': 'pushButton_reattach', 'func': self.reattach_page},
                {'name': 'pushButton_load_pulse_Live', 'func': self.choose_pulse},
                # for Live
                {'name': 'pushButton_lorent_PL', 'func': self.fit_func_PL},
                {'name': 'pushButton_start_PL', 'func': self.start_plot_PL},
                {'name': 'pushButton_stop_PL', 'func': self.stop_plot},
                {'name': 'pushButton_save_PL', 'func': self.save_PL},
                {'name': 'pushButton_load_pulse_PL', 'func': self.choose_pulse},
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
                {'name': 'doubleSpinBox_wavelength', 'func': lambda _: self.is_stabilizer()},
                # only pass param to is_stabilizer when toggled
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
                {'name': 'pushButton_start_PLE', 'value': lambda: f'Start'},
                {'name': 'pushButton_stop_PLE', 'value': lambda: f'Stop'},
                {'name': 'pushButton_wavelength', 'value': lambda: f'Read \n {self.measurement_PLE.x_name}'},
                {'name': 'label_wl', 'value': lambda: f'Min ({self.measurement_PLE.x_unit})'},
                {'name': 'label_wu', 'value': lambda: f'Max ({self.measurement_PLE.x_unit})'},
                {'name': 'label_step_PLE', 'value': lambda: f'Step ({self.measurement_PLE.x_unit})'},
                {'name': 'label_device', 'value': lambda: f'{self.measurement_PLE.x_device_name}'},
                {'name': 'pushButton_save_PLE', 'value': lambda: f'Save'},
                {'name': 'label_relim_PLE', 'value': lambda: f'Relim'},
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
                {'name': 'doubleSpinBox_xl', 'value': lambda: self.measurement_PL.x_array[0]},
                {'name': 'doubleSpinBox_xu', 'value': lambda: self.measurement_PL.x_array[-1]},
                {'name': 'doubleSpinBox_yl', 'value': lambda: self.measurement_PL.y_array[0]},
                {'name': 'doubleSpinBox_yu', 'value': lambda: self.measurement_PL.y_array[-1]},
                {'name': 'doubleSpinBox_exposure_PL', 'value': lambda: self.measurement_PL.exposure},
                {'name': 'doubleSpinBox_step_PL', 'value': lambda: abs(self.measurement_PL.x_array[0] - self.measurement_PL.x_array[1])},
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
        self.valid_fit_func_PLE = ['lorent', 'decay', 'rabi', 'lorent_zeeman']
        self.valid_fit_func_PL = ['center']

        if self.findChild(QComboBox, 'comboBox_relim_PLE') is not None:
            self.comboBox_counter_mode_PLE.addItems(self.measurement_PLE.counter.valid_counter_mode)
            self.comboBox_data_mode_PLE.addItems(self.measurement_PLE.counter.valid_data_mode)
            self.comboBox_relim_PLE.addItems(self.valid_relim_mode)
            self.comboBox_fit_func.addItems(self.valid_fit_func_PLE)
            self.comboBox_update_mode_PLE.addItems(self.measurement_PLE.valid_update_mode)

            self.comboBox_relim_PLE.setCurrentText(self.measurement_PLE.relim_mode)
            self.comboBox_fit_func.setCurrentText(self.measurement_PLE.fit_func)
            self.comboBox_counter_mode_PLE.setCurrentText(self.measurement_PLE.counter_mode)
            self.comboBox_data_mode_PLE.setCurrentText(self.measurement_PLE.data_mode)
            self.comboBox_update_mode_PLE.setCurrentText(self.measurement_PLE.update_mode)
        if self.findChild(QComboBox, 'comboBox_relim_PL') is not None:
            self.comboBox_relim_PL.addItems(self.valid_relim_mode)
            self.comboBox_counter_mode_PL.addItems(self.measurement_PL.counter.valid_counter_mode)
            self.comboBox_data_mode_PL.addItems(self.measurement_PL.counter.valid_data_mode)
            self.comboBox_update_mode_PL.addItems(self.measurement_PL.valid_update_mode)
            self.comboBox_fit_func_PL.addItems(self.valid_fit_func_PL)

            self.comboBox_counter_mode_PL.setCurrentText(self.measurement_PL.counter_mode)
            self.comboBox_data_mode_PL.setCurrentText(self.measurement_PL.data_mode)
            self.comboBox_relim_PL.setCurrentText(self.measurement_PL.relim_mode)
            self.comboBox_update_mode_PL.setCurrentText(self.measurement_PL.update_mode)
        if self.findChild(QComboBox, 'comboBox_relim_Live') is not None:
            self.comboBox_counter_mode_Live.addItems(self.measurement_Live.counter.valid_counter_mode)
            self.comboBox_data_mode_Live.addItems(self.measurement_Live.counter.valid_data_mode)
            self.comboBox_relim_Live.addItems(self.valid_relim_mode)
            self.comboBox_update_mode_Live.addItems(self.measurement_Live.valid_update_mode)

            self.comboBox_counter_mode_Live.setCurrentText(self.measurement_Live.counter_mode)
            self.comboBox_data_mode_Live.setCurrentText(self.measurement_Live.data_mode)
            self.comboBox_relim_Live.setCurrentText(self.measurement_Live.relim_mode)
            self.comboBox_update_mode_Live.setCurrentText(self.measurement_Live.update_mode)
        if self.findChild(QComboBox, 'comboBox_device_gui') is not None:
            self.comboBox_device_gui.addItems(list(self.config_instances.keys()))


        if 'pulse' in list(self.config_instances.keys()):
            self.comboBox_device_gui.setCurrentText('pulse')

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
        self.step_PLE_in_MHz()
        self.estimate_PL_time()
        self.update_stabilizer_step()

    def choose_pulse(self):
        global PULSES_INIT_DIR
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,'select',PULSES_INIT_DIR,'pulse_file (*_pulse.npz)',options=options)

        obj_name = self.sender().objectName()
        attr = 'pulse_' + obj_name.split('_')[-1]
        value = fileName if fileName!='' else None
        if value is not None:
            PULSES_INIT_DIR = os.path.dirname(fileName)
        fileName_display = fileName.split('/')[-1][-28:-10] if fileName!='' else 'None' 
        setattr(self, attr, value)
        self.sender().setText(fileName_display)


    def load_file(self):
        global FIGS_INIT_DIR
        from Confocal_GUI.live_plot import LoadAcquire
        import glob
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, 'select', FIGS_INIT_DIR, 'data_figure (*.jpg *.npz)', options=options)

        if fileName == '':
            return
        FIGS_INIT_DIR = os.path.dirname(fileName)
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
        plot_type = data_generator.plot_type



        if plot_type == '1D':
            if self.measurement_PLE is not None:
                self.stop_plot(select_stop='PLE')
                # make sure no residual selector which may cause thread problem
                if self.data_figure_PLE is not None:
                        self.data_figure_PLE.close_selector()

                data_figure = DataFigure(address = files[0][:-4] + '*', fig=self.canvas_PLE.fig)
                figure_title = (files[0][:-4]).split('\\')[-1]
                data_figure.fig.axes[0].set_title(f'{figure_title}')
                data_figure.fig.tight_layout()
                data_figure.fig.canvas.draw()
                self.data_figure_PLE = data_figure

                self.checkBox_is_auto.setChecked(False)
                # load callback to selector
                from functools import partial
                self.cur_plot = 'PLE'
                params = 'is_print=False, is_in_callback=True'
                eval(f'self.data_figure_{self.cur_plot}.register_selector_callback(0, partial(self.read_range_{self.cur_plot}, {params}))')
                eval(f'self.data_figure_{self.cur_plot}.register_selector_callback(2, partial(self.read_range_{self.cur_plot}, {params}))')
                eval(f'self.data_figure_{self.cur_plot}.register_selector_callback(1, partial(self.read_xy_{self.cur_plot}, {params}))')
            else:
                self.print_log(f'Cannot load 1D plot_type')
                return
        elif plot_type == '2D':
            if self.measurement_PL is not None:
                self.stop_plot(select_stop='PL')
                # make sure no residual selector which may cause thread problem
                if self.data_figure_PL is not None:
                        self.data_figure_PL.close_selector()

                data_figure = DataFigure(address = files[0][:-4] + '*', fig=self.canvas_PL.fig)
                figure_title = (files[0][:-4]).split('\\')[-1]
                data_figure.fig.axes[1].set_title(f'{figure_title}')
                data_figure.fig.tight_layout()
                data_figure.fig.canvas.draw()
                self.data_figure_PL = data_figure

                self.checkBox_is_auto.setChecked(False)
                # load callback to selector
                from functools import partial
                self.cur_plot = 'PL'
                params = 'is_print=False, is_in_callback=True'
                eval(f'self.data_figure_{self.cur_plot}.register_selector_callback(0, partial(self.read_range_{self.cur_plot}, {params}))')
                eval(f'self.data_figure_{self.cur_plot}.register_selector_callback(2, partial(self.read_range_{self.cur_plot}, {params}))')
                eval(f'self.data_figure_{self.cur_plot}.register_selector_callback(1, partial(self.read_xy_{self.cur_plot}, {params}))')
            else:
                self.print_log(f'Cannot load 2D plot_type')
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


    def is_stabilizer(self, is_toggle_changed=False):

        if not self.checkBox_is_stabilizer.isChecked():
            self.measurement_PLE.to_final_state()
            return
        # else, stabilizer is checked
        if is_toggle_changed is True:
            self.wavelength = self.doubleSpinBox_wavelength.value()
            self.measurement_PLE.to_initial_state()
            self.measurement_PLE.device_to_state(self.wavelength)
        else:
            # only value is changed
            self.wavelength = self.doubleSpinBox_wavelength.value()
            self.measurement_PLE.device_to_state(self.wavelength)



    def open_device_gui(self):
        device_handle = self.comboBox_device_gui.currentText()
        device_instance = self.config_instances[device_handle]
        if hasattr(device_instance, 'gui'):
            if device_handle == 'pulse':
                self.pulse_gui_handle = device_instance.gui(is_in_GUI=True)
                self.pulse_gui_handle.show()
            else:
                device_instance.gui()
        

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
        for attr in ['exposure_Live', 'many', 'repeat_Live']:
            value = getattr(self, f'doubleSpinBox_{attr}').value()
            setattr(self, attr, value)
        self.repeat_Live = int(self.repeat_Live)

        for attr in ['relim_Live', 'counter_mode_Live', 'data_mode_Live', 'update_mode_Live']: # Read from GUI panel
            value = getattr(self, f'comboBox_{attr}').currentText()
            setattr(self, attr, value)

        for attr in ['is_finite']:
            value = getattr(self, f'checkBox_{attr}').isChecked()
            setattr(self, attr, value)
            
            

  


    def start_plot_Live(self):
        self.stop_plot()

        self.print_log(f'Live started')
        self.cur_plot = 'Live'
        self.read_data_Live()
        
        
        data_x = np.arange(self.many)

        self.measurement_Live.load_params(data_x=data_x, exposure=self.exposure_Live, repeat=self.repeat_Live,
            is_finite=self.is_finite, update_mode=self.update_mode_Live, pulse_file=self.pulse_Live,
            counter_mode=self.counter_mode_Live, data_mode=self.data_mode_Live, relim_mode=self.relim_Live)
        data_y = self.measurement_Live.data_y
        self.live_plot_Live = LiveAndDisLive(labels=['Data', f'Counts/{self.exposure_Live:.2f}s'],
                            update_time=0.02, data_generator=self.measurement_Live, data=[data_x, data_y],
                                         fig=self.canvas_Live.fig, relim_mode = self.relim_Live)
        
        
        self.cur_live_plot = self.live_plot_Live
        self.live_plot_Live.init_figure_and_data()

        # make sure no residual selector which may cause thread problem
        if self.data_figure_Live is not None:
                self.data_figure_Live.close_selector()
                self.data_figure_Live = None
                
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int(1000*self.live_plot_Live.update_time))  # Interval in milliseconds
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
            
            
        
    def update_plot(self):
        # basically copy of live_plot.plot() method, to handle the update in pyqt thread
        if not self.cur_live_plot.data_generator.is_done:
            if (self.cur_live_plot.data_generator.points_done == self.cur_live_plot.points_done):
                # if no new data then no update
                return

            self.cur_live_plot.update_figure()
            self.estimate_PL_time()
            self.estimate_PLE_time()
            # update estimate finish time
        else:
            if not self.timer.isActive():
                return
            else:
                self.timer.stop()
                self.cur_live_plot.update_figure()
                self.estimate_PL_time()
                self.estimate_PLE_time()
                self.cur_live_plot.data_generator.stop()  
                self.cur_live_plot.after_plot()
                setattr(self, f'data_figure_{self.cur_plot}', DataFigure(live_plot=self.cur_live_plot))

                if self.findChild(QComboBox, 'comboBox_relim_PLE') is not None:
                    self.checkBox_is_stabilizer.setDisabled(False)
                    self.doubleSpinBox_wavelength.setDisabled(False)
                if self.findChild(QComboBox, 'comboBox_relim_PL') is not None:
                    self.checkBox_is_bind.setChecked(False)
                    self.checkBox_is_bind.setDisabled(False)

                if self.cur_plot not in ['PL', 'PLE']:
                    return
                else:
                    # load callback to selector
                    from functools import partial
                    params = 'is_print=False, is_in_callback=True'
                    eval(f'self.data_figure_{self.cur_plot}.register_selector_callback(0, partial(self.read_range_{self.cur_plot}, {params}))')
                    eval(f'self.data_figure_{self.cur_plot}.register_selector_callback(2, partial(self.read_range_{self.cur_plot}, {params}))')
                    eval(f'self.data_figure_{self.cur_plot}.register_selector_callback(1, partial(self.read_xy_{self.cur_plot}, {params}))')
    
    def stop_plot(self, clicked=True, select_stop=None):
        self.print_log(f'Plot stopped')
        if select_stop is None:
            # must be second params, cause button click event will pass a bool variable to function connected
            if self.cur_live_plot is not None:
                self.cur_live_plot.stop()
                self.update_plot()
                # block excuetion until plot is done
        else:
            if hasattr(self, f'live_plot_{select_stop}'):
                plot_handle = getattr(self, f'live_plot_{select_stop}')
                plot_handle.stop()
                self.update_plot()

        if self.findChild(QComboBox, 'comboBox_relim_PLE') is not None:
            self.checkBox_is_stabilizer.setDisabled(False)
            self.doubleSpinBox_wavelength.setDisabled(False)
            self.estimate_PLE_time()
        if self.findChild(QComboBox, 'comboBox_relim_PL') is not None:
            self.checkBox_is_bind.setChecked(False)
            self.checkBox_is_bind.setDisabled(False)
            self.estimate_PL_time()



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
        if hasattr(self, 'live_plot_PL') and self.live_plot_PL.data_generator.thread.is_alive():
            points_total = self.live_plot_PL.points_total
            points_done = self.live_plot_PL.points_done
            if points_done==0:
                return
            ratio = points_done/points_total
            time_done = time.time()-self.time_PL_start
            self.overhead_PL = time_done/points_done - self.exposure_PL
            self.lineEdit_time_PL.setText(f'PL finishes in {time_done:.2f}s / {time_done/ratio:.2f}s'
                f', {ratio*100:.2f}%, {self.overhead_PL+self.exposure_PL:.2f}s/point')

        else:
            self.read_data_PL()
            self.PL_points = (len(np.arange(self.xl, self.xu+self.step_PL, self.step_PL)) * 
                len(np.arange(self.yl, self.yu+self.step_PL, self.step_PL)))
            time_est = (self.exposure_PL+self.overhead_PL) * self.PL_points
            self.lineEdit_time_PL.setText(f'new PL finishes in {time_est:.2f}s')
            self.time_PL_start = time.time()





    def move_scanner(self):
        x = self.doubleSpinBox_X.value()
        y = self.doubleSpinBox_Y.value()

        self.measurement_PL.device_to_state((x, y))

        
    def read_xy_PL(self, is_print=True, is_in_callback=False):
        if is_in_callback and not self.checkBox_is_auto.isChecked():
            return
        if self.data_figure_PL is None:
            return
        if self.data_figure_PL.selector == []:
            return
        _xy = self.data_figure_PL.selector[1].xy #cross selector
        
        if _xy is not None:
            self.doubleSpinBox_X.setValue(_xy[0])
            self.doubleSpinBox_Y.setValue(_xy[1])
            if is_print:
                self.print_log(f'read x = {_xy[0]}, y = {_xy[1]}')
        else:
            if is_print:
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
        
        
    def read_range_PL(self, is_print=True, is_in_callback=False):
        if is_in_callback and not self.checkBox_is_auto.isChecked():
            return
        if self.data_figure_PL is None:
            return
        if self.data_figure_PL.selector[0].range[0] is None:
            xlim = self.data_figure_PL.fig.axes[0].get_xlim()
            ylim = self.data_figure_PL.fig.axes[0].get_ylim()
            xl, xh, yl, yh = np.min(xlim), np.max(xlim), np.min(ylim), np.max(ylim)
        else:
            xl, xh, yl, yh = self.data_figure_PL.selector[0].range

        self.doubleSpinBox_xl.setValue(self.data_figure_PL._align_to_grid(xl, type='x'))
        self.doubleSpinBox_xu.setValue(self.data_figure_PL._align_to_grid(xh, type='x'))
        self.doubleSpinBox_yl.setValue(self.data_figure_PL._align_to_grid(yl, type='y'))
        self.doubleSpinBox_yu.setValue(self.data_figure_PL._align_to_grid(yh, type='y'))

        if is_print:
            self.print_log(f'PL range updated')
        
  
            
    def read_data_PL(self):
        for attr in ['exposure_PL', 'xl', 'xu', 'yl', 'yu', 'step_PL', 'repeat_PL']:
            value = getattr(self, f'doubleSpinBox_{attr}').value()
            setattr(self, attr, value)

        for attr in ['relim_PL', 'counter_mode_PL', 'data_mode_PL', 'update_mode_PL']: # Read from GUI panel
            value = getattr(self, f'comboBox_{attr}').currentText()
            setattr(self, attr, value)

        self.repeat_PL = int(self.repeat_PL)
            
            
    def start_plot_PL(self):
        self.stop_plot()

        self.print_log(f'PL started')
        self.cur_plot = 'PL'  
        self.read_data_PL()

        x_array = np.arange(self.xl, self.xu+self.step_PL, self.step_PL)
        y_array = np.arange(self.yl, self.yu+self.step_PL, self.step_PL)
        # self.step_PL to include the end point if possible

        self.measurement_PL.load_params(x_array=x_array, y_array=y_array, exposure=self.exposure_PL, repeat=self.repeat_PL,\
            counter_mode=self.counter_mode_PL, update_mode=self.update_mode_PL, pulse_file=self.pulse_PL,
            data_mode=self.data_mode_PL, relim_mode=self.relim_PL)
                

        if self.findChild(QCheckBox, 'checkBox_is_stabilizer') is not None:
            self.checkBox_is_stabilizer.setDisabled(True)
            self.doubleSpinBox_wavelength.setDisabled(True)

        self.checkBox_is_bind.setChecked(False)
        self.checkBox_is_bind.setDisabled(True)


        data_y = self.measurement_PL.data_y
        data_x = self.measurement_PL.data_x
        update_time = float(np.max([1, self.exposure_PL*len(data_x)/1000]))
        self.live_plot_PL = PLDisLive(labels=[['X', 'Y'], f'Counts/{self.exposure_PL:.2f}s'],
                        update_time=update_time, data_generator=self.measurement_PL, data=[data_x, data_y],
                                       fig=self.canvas_PL.fig, relim_mode = self.relim_PL)
        
        self.cur_live_plot = self.live_plot_PL
        self.live_plot_PL.init_figure_and_data()

        # make sure no residual selector which may cause thread problem
        if self.data_figure_PL is not None:
                self.data_figure_PL.close_selector()
                self.data_figure_PL = None
        
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int(1000*self.live_plot_PL.update_time))  # Interval in milliseconds
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()


    def step_PLE_in_MHz(self):
        if (not hasattr(self, 'measurement_PLE')) or (self.measurement_PLE is None):
            return
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
        if hasattr(self, 'live_plot_PLE') and self.live_plot_PLE.data_generator.thread.is_alive():
            points_total = self.live_plot_PLE.points_total
            points_done = self.live_plot_PLE.points_done
            if points_done==0:
                return
            ratio = points_done/points_total
            time_done = time.time() - self.time_PLE_start
            self.overhead_PLE = time_done/points_done - self.exposure_PLE
            self.lineEdit_time_PLE.setText(f'{self.measurement_PLE.measurement_name}'\
                f'finishes in {time_done:.2f}s / {time_done/ratio:.2f}s'
                f', {ratio*100:.2f}%, {self.overhead_PLE+self.exposure_PLE:.2f}s/point')

        else:
            self.read_data_PLE()
            self.PLE_points = len(np.arange(self.wl, self.wu+self.step_PLE, self.step_PLE)) 
            time_est = (self.exposure_PLE + self.overhead_PLE)* self.PLE_points
            # considering the overhead of stabilizing laser frequency
            self.lineEdit_time_PLE.setText(f'new {self.measurement_PLE.measurement_name} finishes in {time_est:.2f}s')
            self.time_PLE_start = time.time()


    def read_xy_PLE(self, is_print=True, is_in_callback=False):
        if is_in_callback and not self.checkBox_is_auto.isChecked():
            return
        if self.data_figure_PLE is None:
            return
        if self.data_figure_PLE.selector == []:
            if is_print:
                self.print_log(f'No {self.measurement_PLE.x_name} to read')
            return
        _wavelength = self.data_figure_PLE.selector[1].wavelength
        
        if _wavelength is not None:
            self.doubleSpinBox_wavelength.setValue(self.data_figure_PLE.transform_back(_wavelength))
            # transform back to original unit
            if is_print:
                self.print_log(f'{self.measurement_PLE.x_name} was read')
        else:
            if is_print:
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
        
        
        
    def fit_func_PLE(self):

        fit_func_PLE = self.comboBox_fit_func.currentText()
        if self.data_figure_PLE is None:
            self.print_log(f'No figure to fit')
            return
        if self.data_figure_PLE.fit is not None:
            self.data_figure_PLE.clear()
            self.print_log(f'fit cleared')
        else:
            eval(f'self.data_figure_PLE.{fit_func_PLE}()')
            log_info = self.data_figure_PLE.log_info
            self.print_log(f'curve fitted, {log_info}')

    def fit_func_PL(self):

        fit_func_PL = self.comboBox_fit_func_PL.currentText()
        if self.data_figure_PL is None:
            self.print_log(f'No figure to fit')
            return
        if self.data_figure_PL.fit is not None:
            self.data_figure_PL.clear()
            self.print_log(f'fit cleared')
        else:
            eval(f'self.data_figure_PL.{fit_func_PL}()')
            log_info = self.data_figure_PL.log_info
            self.print_log(f'curve fitted, {log_info}')

    def change_unit(self):
        if self.data_figure_PLE is None:
            self.print_log(f'No figure to change unit')
            return
        self.data_figure_PLE.change_unit()
        
        
        
    def read_range_PLE(self, is_print=True, is_in_callback=False):
        if is_in_callback and not self.checkBox_is_auto.isChecked():
            return
        if self.data_figure_PLE is None:
            return
        if self.data_figure_PLE.selector[0].range[0] is None:
            xlim = self.data_figure_PLE.fig.axes[0].get_xlim()
            ylim = self.data_figure_PLE.fig.axes[0].get_ylim()
            xl, xh, yl, yh = np.min(xlim), np.max(xlim), np.min(ylim), np.max(ylim)
        else:
            xl, xh, yl, yh = self.data_figure_PLE.selector[0].range
        
        
        new_xl, new_xh = np.sort([self.data_figure_PLE.transform_back(xl), self.data_figure_PLE.transform_back(xh)])

        self.doubleSpinBox_wl.setValue(new_xl)
        self.doubleSpinBox_wu.setValue(new_xh)

        if is_print:
            self.print_log(f'{self.measurement_PLE.measurement_name} range updated')
        
        
    def read_data_PLE(self):
        for attr in ['exposure_PLE', 'wl', 'wu', 'step_PLE', 'repeat_PLE']: # Read from GUI panel
            value = getattr(self, f'doubleSpinBox_{attr}').value()
            setattr(self, attr, value)

        for attr in ['relim_PLE', 'counter_mode_PLE', 'data_mode_PLE', 'update_mode_PLE']: # Read from GUI panel
            value = getattr(self, f'comboBox_{attr}').currentText()
            setattr(self, attr, value)

        self.repeat_PLE = int(self.repeat_PLE)


    def start_plot_PLE(self):
        self.stop_plot()

        self.print_log(f'{self.measurement_PLE.measurement_name} started')
        self.cur_plot = 'PLE'
        self.pushButton_load.setText(f'Load file:')
                    
        
        self.read_data_PLE()
        
        data_x = np.arange(self.wl, self.wu+self.step_PLE, self.step_PLE)
        # +step to include the end point is possible

        self.checkBox_is_stabilizer.setChecked(False)
        self.checkBox_is_stabilizer.setDisabled(True)
        self.doubleSpinBox_wavelength.setDisabled(True)


        self.measurement_PLE.load_params(data_x=data_x, exposure=self.exposure_PLE, repeat=self.repeat_PLE,
            update_mode=self.update_mode_PLE, pulse_file=self.pulse_PLE,
            counter_mode=self.counter_mode_PLE, data_mode=self.data_mode_PLE, relim_mode=self.relim_PLE)

        data_y = self.measurement_PLE.data_y
        update_time = float(np.max([1, self.exposure_PLE*len(data_x)/1000]))
        self.live_plot_PLE = PLELive(labels=[f'{self.measurement_PLE.x_name} ({self.measurement_PLE.x_unit})', f'Counts/{self.exposure_PLE:.2f}s'], 
                                     update_time=update_time, data_generator=self.measurement_PLE, data=[data_x, data_y],
                                    fig=self.canvas_PLE.fig, relim_mode = self.relim_PLE)
        
        self.cur_live_plot = self.live_plot_PLE
        self.live_plot_PLE.init_figure_and_data()

        # make sure no residual selector which may cause thread problem
        if self.data_figure_PLE is not None:
                self.data_figure_PLE.close_selector()
                self.data_figure_PLE = None
        
        
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int(1000*self.live_plot_PLE.update_time))  # Interval in milliseconds
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

        
            

            
    def closeEvent(self, event):
        if self.detached_window is not None:
            self.reattach_page()

        if self.cur_live_plot is not None:
            self.cur_live_plot.stop()

        if self.findChild(QCheckBox, 'checkBox_is_stabilizer') is not None:
            self.checkBox_is_stabilizer.setChecked(False)

        if hasattr(self, 'pulse_gui_handle') and (self.pulse_gui_handle is not None):
            self.pulse_gui_handle.close()

        plt.close('all') # make sure close all plots which avoids error message
        event.accept()



def GUI_(measurement_PLE=None, measurement_PL=None, measurement_Live=None, mode='PL_and_PLE'):
    """
    The function opens pyqt GUI for PLE, PL, live counts, and pulse control.
    Save button will also output data and figure to jupyter notebook.
   
    Examples
    --------
    >>> GUI()

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
        w = MainWindow(measurement_PLE=measurement_PLE, measurement_PL=measurement_PL
            , measurement_Live=measurement_Live, ui='GUI.ui')
    elif mode == 'PLE_and_Live':
        w = MainWindow(measurement_PLE=measurement_PLE, measurement_PL=measurement_PL
            , measurement_Live=measurement_Live, ui='PLE.ui')
    elif mode == 'PL_and_Live':
        w = MainWindow(measurement_PLE=measurement_PLE, measurement_PL=measurement_PL
            , measurement_Live=measurement_Live, ui='PL.ui')
    elif mode == 'Live':
        w = MainWindow(measurement_PLE=measurement_PLE, measurement_PL=measurement_PL
            , measurement_Live=measurement_Live, ui='Live.ui')

    app.setStyle('Windows')
    try:
        sys.exit(app.exec_())
    except SystemExit as se:
        if se.code != 0:
            raise se
    # make sure jupyter notebook does not catch error when exit normally



def GUI_PLE(measurement_PLE, measurement_Live):
    """
    GUI for PLE type measurement

    """
    GUI_(measurement_PLE=measurement_PLE,
        measurement_PL=None, measurement_Live=measurement_Live, mode='PLE_and_Live')





def GUI_PL(measurement_PL, measurement_Live):
    """
    GUI for PL type measurement

    """

    GUI_(measurement_PL=measurement_PL,
        measurement_PLE=None, measurement_Live=measurement_Live, mode='PL_and_Live')

def GUI_Live(measurement_Live):
    """
    GUI for Live type measurement

    """

    GUI_(measurement_PL=None,
        measurement_PLE=None, measurement_Live=measurement_Live, mode='Live')


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
            current_value.setMaxLength(20)  # **Increased to accommodate scientific notation**
            current_value.setFont(font)
            current_value.setFixedWidth(150)
            current_value.setFixedHeight(30)

            # Determine the input control based on property type
            if prop_type == 'float':
                # **Use QLineEdit instead of QDoubleSpinBox to support scientific notation input**
                input_control = QLineEdit()
                input_control.setFont(font)
                input_control.setFixedWidth(150)
                input_control.setFixedHeight(30)

                # **Set a regular expression validator to allow scientific notation format**
                regex = QRegExp(r'^[-+]?\d*\.?\d+([eE][-+]?\d+)?$')
                validator = QRegExpValidator(regex)
                input_control.setValidator(validator)

                # **Set placeholder text to guide user input format**
                input_control.setPlaceholderText("e.g. 2e9 or 1.2")

                # Dynamically get the lower and upper bounds from the device
                lb_attr = f"{prop}_lb"
                ub_attr = f"{prop}_ub"
                try:
                    lb = getattr(self.device, lb_attr)
                    ub = getattr(self.device, ub_attr)
                except AttributeError:
                    lb = -1e9
                    ub = 1e9  # **Use default range if bounds are not defined**

                # **Store lower and upper bounds for validation during apply**
                input_control.lb = lb
                input_control.ub = ub

            elif prop_type == 'str':
                input_control = QComboBox()
                input_control.setFont(font)
                input_control.setFixedWidth(150)
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

            # Connect the apply button to the handler
            apply_button.clicked.connect(
                lambda checked, p=prop, i=input_control, c=current_value: self.apply_value(p, i, c)
            )

            # Add widgets to the property layout
            prop_layout.addWidget(label)
            prop_layout.addWidget(current_value)
            prop_layout.addStretch(1)
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

    def format_float(self, value):
        """
        Formats the float value based on its magnitude:
        - If the absolute value is between 1e-3 and 1e3, display in standard decimal format without trailing zeros.
        - Otherwise, display in scientific notation without trailing zeros.
        """
        if value is None:
            return 'None'

        abs_value = abs(value)
        if abs_value==0:
            return '0'
        elif 1e-3 <= abs_value < 1e3:
            # **Standard decimal format, displaying raw data
            formatted = f"{value}"
            return formatted
        else:
            # **Scientific notation format, displaying up to the last non-zero digit**
            for i, digit in enumerate((f"{value}")[::-1]):
                if digit!='0' and digit!='.':
                    if '.' in ((f"{value}")[::-1])[i:]:
                        non_zero_length = len(f"{value}")-i-1
                    else:
                        non_zero_length = len(f"{value}")-i
                    break
                non_zero_length = len(f"{value}")
            formatted = f"{value:.{non_zero_length-1}e}"
            return formatted

    def update_property_value(self, prop):
        value = getattr(self.device, prop)
        if self.controls[prop]['type'] == 'str':
            value_str = 'True' if value else 'False'
            self.controls[prop]['current_value'].setText(value_str)
        elif self.controls[prop]['type'] == 'float':
            # **Choose display format based on the value range**
            formatted_value = self.format_float(value)
            self.controls[prop]['current_value'].setText(formatted_value)
        else:
            self.controls[prop]['current_value'].setText(f'{value}')

    def apply_value(self, prop, input_control, current_label: QLineEdit):
        prop_type = self.controls[prop]['type']
        try:
            if prop_type == 'float':
                input_text = input_control.text()
                if not input_text:
                    raise ValueError("Input cannot be empty.")
                new_value = float(input_text)

                # **Enforce bounds if they exist**
                lb = getattr(input_control, 'lb', -1e9)
                ub = getattr(input_control, 'ub', 1e9)
                if not (lb <= new_value <= ub):
                    raise ValueError(f"Value must be between {self.format_float(lb)} and {self.format_float(ub)}.")

            elif prop_type == 'str':
                new_value = input_control.currentText() == 'True'
            else:
                new_value = input_control.text()

            setattr(self.device, prop, new_value)
            self.update_property_value(prop)
        except ValueError as ve:
            QMessageBox.warning(self, "Input Error", f"Invalid input for {prop}: {ve}")
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


class DraggableItem:
    """
    Save properties
    """
    def __init__(self, widget, item_type):
        self.widget = widget
        self.item_type = item_type
        if self.item_type != 'pulse':
            self.widget.setStyleSheet("""QGroupBox {border: 2px solid grey;}""")


class DragContainer(QWidget):
    """
    Drag container for dragable bracket in pulse control gui
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.layout_main = QHBoxLayout(self)
        self.layout_main.setSpacing(20)
        self.items = []

        self.dragStartPos = None
        self.draggingIndex = None

        # maintain a indicator between containers but hide 
        self.insert_indicator = QFrame()
        self.insert_indicator.setFrameShape(QFrame.VLine)
        self.insert_indicator.setFrameShadow(QFrame.Raised)
        self.insert_indicator.setStyleSheet("QFrame { background-color: red; width: 2px; }")
        self.insert_indicator.hide()

    def add_item(self, widget, item_type):
        it = DraggableItem(widget, item_type)
        self.items.append(it)
        self.layout_main.addWidget(widget)

    def insert_item(self, index, widget, item_type):
        it = DraggableItem(widget, item_type)
        self.items.insert(index, it)
        self.refresh_layout()

    def refresh_layout(self):
        # remove all widget
        while self.layout_main.count() > 0:
            c = self.layout_main.takeAt(0)
            w = c.widget()
            if w:
                w.setParent(None)
        # reinsert follows order
        for it in self.items:
            self.layout_main.addWidget(it.widget)
        # add end indicator
        self.layout_main.addWidget(self.insert_indicator)
        self.insert_indicator.hide()

    def index_of_widget(self, w):
        for i, it in enumerate(self.items):
            if it.widget == w:
                return i
        return -1

    # ========== Drag & Drop ==========
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragStartPos = event.pos()
            clicked_index = self.find_child_index_by_pos(event.pos())
            if clicked_index != -1:
                self.draggingIndex = clicked_index
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.dragStartPos is not None:
            distance = (event.pos() - self.dragStartPos).manhattanLength()
            if distance > QApplication.startDragDistance():
                if self.draggingIndex is not None:
                    self.startDrag(self.draggingIndex)
                    self.draggingIndex = None
                    self.dragStartPos = None
        super().mouseMoveEvent(event)

    def startDrag(self, index):
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData("application/x-drag-bracket", str(index).encode("utf-8"))
        drag.setMimeData(mime)

        # highlight selected drag container
        self.index = index
        self.items[self.index].widget.setStyleSheet("QGroupBox { border: 2px solid red; }")

        # screentshot selected container
        pixmap = QPixmap(self.items[self.index].widget.size())
        self.items[self.index].widget.render(pixmap)
        drag.setPixmap(pixmap)

        dropAction = drag.exec_(Qt.MoveAction)

        # end of drag
        if self.items[self.index].item_type == 'pulse':
            self.items[self.index].widget.setStyleSheet("")
            self.update_pulse_index()
        else:
            self.items[self.index].widget.setStyleSheet("QGroupBox { border: 2px solid grey; }")

    def update_pulse_index(self):
        pulse_list = [item for item in self.items if item.item_type=='pulse']
        for ii, item in enumerate(pulse_list):
            widget = item.widget
            widget.setTitle(f'Pulse{ii}')

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-drag-bracket"):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-drag-bracket"):
            event.acceptProposedAction()
            # pos of drag
            insert_pos = self.get_item_at_pos(event.pos())
            self.show_insert_indicator(insert_pos)
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasFormat("application/x-drag-bracket"):
            # 1) get item of drag container
            raw_data = event.mimeData().data("application/x-drag-bracket")
            old_bytes = bytes(raw_data)
            old_index = int(old_bytes.decode("utf-8"))
            dragged_item = self.items[old_index]

            # 2) new pos
            insert_pos = self.get_item_at_pos(event.pos())
            # 3) new item array
            new_items = self.items[:]             
            new_items.remove(dragged_item)
            if insert_pos > old_index:
                insert_pos -= 1
            new_items.insert(insert_pos, dragged_item)

            # 4) check constarin
            if not self.check_bracket_constraints(new_items):
                event.ignore()
                self.insert_indicator.hide()
                return
            else:
                event.setDropAction(Qt.MoveAction)
                event.accept()
                self.index = insert_pos
                self.items = new_items
                self.refresh_layout()
                self.insert_indicator.hide()

        else:
            super().dropEvent(event)


    def dragLeaveEvent(self, event):
        self.insert_indicator.hide()
        super().dragLeaveEvent(event)

    # ========== sub functions ==========
    def find_child_index_by_pos(self, pos):
        for i, it in enumerate(self.items):
            if it.widget.geometry().contains(pos):
                return i
        return -1

    def get_item_at_pos(self, pos):
        x = pos.x()
        for i, it in enumerate(self.items):
            w = it.widget
            geo = w.geometry()
            mid = geo.x() + geo.width() // 2
            if x < mid:
                return i
        return len(self.items)

    def show_insert_indicator(self, index):
        self.layout_main.removeWidget(self.insert_indicator)
        self.layout_main.insertWidget(index, self.insert_indicator)
        self.insert_indicator.show()

    def check_bracket_constraints(self, item_list=None):
        if item_list is None:
            item_list = self.items  

        start_idx = None
        end_idx = None
        for i, it in enumerate(item_list):
            if it.item_type == "bracket_start":
                start_idx = i
            elif it.item_type == "bracket_end":
                end_idx = i

        if start_idx is not None and end_idx is not None:
            # 1) end must larger than start
            if end_idx <= start_idx:
                return False
            # 2) at least two pulses away
            if end_idx < start_idx + 3:
                return False
        return True

class PulseGUI(QDialog):
    """
    GUI:

        Off Pulse/On Pulse:
            button to off and on pulse, using pulse sequence currently in the GUI.
            On pulse will also save current pulse sequence.

        Remove/Add Column:
            button to remove or add one more pulse column

        Save to/Load from file:
            button to save pulse to file(*pulse.npz) or load from saved pulse(*pulse.npz)
            
        Save Pulses:
            button to apply current pulse sequence (potentially used by measurement) but not on pulse.

        Add/Delete Bracket:
            button to add a bracket which defines which part of pulse sequence should be repeated

        Ref settings:
            if checkbox is checked, will automatically repeat current sequence another time for reference,
            where the second sequence disables signal and replace DAQ gate with DAQ_ref gate while clock will
            not be repeated for the ref pulse

        x:
            number in Channel delay or pulse duration can be a number or expression, if expression, then corresponding
            time will be replaced by pulse.x property when pulse.read_data() is called
            e.g.
                first run
                pulse.x = 100
                pulse1 duration: 1000-x -> 1000-100=900  

                second run
                pulse.x = 200
                pulse1 duration: 1000-x -> 1000-200=800

            which enables configure pulse sequence before measurement, and gives measurement an option to change pulse
            on demand by only changing pulse.x or config_instances['pulse'].x 

        Pulse:
            you can drag and insert all pulses into any new position
    """

    def __init__(self, device_handle, parent=None):
        super().__init__()
        font = QFont('Arial', 10) 
        self.setFont(font)
        self.setWindowTitle('PulseGUI@Wanglab, UOregon')
        self.device_handle = device_handle
        self.channel_names_map = [f'Ch{channel}' for channel in range(8)]
        self.drag_container = None

        global PULSES_INIT_DIR
        
        self.layout = QVBoxLayout(self)
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
        

        self.btn3 = QPushButton('Save to file')
        self.btn3.setFixedSize(150,100)
        self.btn3.clicked.connect(self.save_to_file)
        self.layout_saveload.addWidget(self.btn3)

        self.btn4 = QPushButton('Load from file')
        self.btn4.setFixedSize(150,100)
        self.btn4.clicked.connect(self.load_from_file)
        self.layout_saveload.addWidget(self.btn4)

        self.btn4 = QPushButton('Save Pulse')
        self.btn4.setFixedSize(150,100)
        self.btn4.clicked.connect(self.save_data)
        self.layout_saveload.addWidget(self.btn4)

        self.btn_add_bracket = QPushButton("Add Bracket")
        self.btn_add_bracket.clicked.connect(self.on_add_bracket)
        self.btn_add_bracket.setFixedSize(150,100)
        self.layout_saveload.addWidget(self.btn_add_bracket)

        self.btn_ref_info = self.add_ref_info()
        self.btn_ref_info.setFixedSize(150,150)
        self.layout_saveload.addWidget(self.btn_ref_info)

        self.load_data()


        self.show()


    def off_pulse(self):


        # Create a sequence object
        self.device_handle.off_pulse()
        # need to repeat 8 times because pulse streamer will pad sequence to multiple of 8ns, otherwise unexpected changes of pulse


    def on_pulse(self):

        self.save_data()
        self.device_handle.on_pulse()

    def add_ref_info(self):
        group_box = QGroupBox('Ref settings')
        group_layout = QVBoxLayout()

        self.checkbox = QCheckBox('Auto append ref')
        group_layout.addWidget(self.checkbox)

        self.comboboxes = {}
        combobox_labels = ['Signal', 'DAQ', 'DAQ_ref', 'Clock']
        for label_text in combobox_labels:
            h_layout = QHBoxLayout()
            label = QLabel(label_text)
            combobox = QComboBox()
            combobox.addItem('None')
            combobox.addItems([f'Ch{channel}' for channel in range(8)])
            self.comboboxes[label_text] = combobox

            h_layout.addWidget(label)
            h_layout.addWidget(combobox)
            group_layout.addLayout(h_layout)

        group_box.setLayout(group_layout)
        return group_box


    def handle_text_change(self, text, combo_box):
        
        if text.isdigit() or (len(text)>=2 and (text[0] in ['-', '+']) and text[1:].isdigit()):
            combo_box.setEnabled(True)
            if combo_box.currentText() == 'str (ns)':
                combo_box.setCurrentText('ns')
        else:
            combo_box.setCurrentText('str (ns)')
            combo_box.setEnabled(False)


    def on_add_bracket(self, start_index=0, end_index=-1):
        """
        If brackets do not exist yet:
          1) Find the first pulse index and the last pulse index.
          2) Insert the start bracket right before the first pulse.
          3) Insert the end bracket right after the last pulse.
        If brackets already exist, remove them.
        """
        if not self.bracket_exists:
            # 1) Identify the indices of the first and last pulse
            first_pulse_index = 0
            last_pulse_index = len(self.drag_container.items)
            end_index = last_pulse_index if (end_index==-1) else (end_index+1)
            # 2) Create the start bracket widget
            start_box = QGroupBox("Start")
            start_box.setFont(QFont('Arial', 10))
            start_box.setFixedWidth(100)
            vb1 = QVBoxLayout(start_box)
            sublayout = QVBoxLayout()
            vb1.addLayout(sublayout)
            btn = QLabel("Start\nof\nrepeat")
            btn.setFont(QFont('Arial', 10))
            sublayout.addWidget(btn)


            # 3) Create the end bracket widget
            end_box = QGroupBox("End")
            end_box.setFont(QFont('Arial', 10))
            end_box.setFixedWidth(100)
            vb2 = QVBoxLayout(end_box)
            sublayout = QVBoxLayout()
            vb2.addLayout(sublayout)
            btn = QLabel("End\nof\nrepeat")
            btn.setFont(QFont('Arial', 10))
            sublayout.addWidget(btn)
            sp = QDoubleSpinBox()
            sp.setDecimals(0)
            sp.setRange(1, 999)
            sp.setValue(self.repeat_info[2])
            sp.setFont(QFont('Arial', 10))
            sublayout.addWidget(sp)

            # -- Inserting the brackets --
            # NOTE: Insert the end bracket first so the index of the first pulse won't shift.
            #       We want the end bracket after the last pulse, so its index is last_pulse_index + 1.
            #self.drag_container.insert_item(last_pulse_index + 1, end_box, "bracket_end")

            # Now insert the start bracket right before the first pulse (no shift occurs yet).
            self.drag_container.insert_item(start_index, start_box, "bracket_start")
            self.drag_container.insert_item(end_index+1, end_box, "bracket_end")
            #self.drag_container.refresh_layout()

            # Toggle flag and button text
            self.bracket_exists = True
            self.btn_add_bracket.setText("Delete Bracket")

        else:
            # Remove the bracket items only, leaving pulses intact
            self.delete_bracket_only()
            self.bracket_exists = False
            self.btn_add_bracket.setText("Add Bracket")


    def delete_bracket_only(self):
        """
        Remove items of type 'bracket_start' and 'bracket_end', keeping only pulses.
        """
        new_list = []
        for it in self.drag_container.items:
            if it.item_type not in ("bracket_start", "bracket_end"):
                new_list.append(it)
        self.drag_container.items = new_list
        self.drag_container.refresh_layout()


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
            self.repeat_info = self.device_handle.repeat_info_tmp
            self.ref_info = self.device_handle.ref_info_tmp
        else:
            self.delay_array = self.device_handle.delay_array
            self.data_matrix = self.device_handle.data_matrix
            self.channel_names = self.device_handle.channel_names
            self.repeat_info = self.device_handle.repeat_info
            self.ref_info = self.device_handle.ref_info

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
                line_edit.setText(str(delay_value))
            else:
                duration = int(delay_value)
                if duration==0:
                    combo_box.setCurrentText('ns')
                    line_edit.setText(str(duration))
                elif duration%1000000==0:
                    combo_box.setCurrentText('ms')
                    line_edit.setText(str(duration//1000000))
                elif duration%1000==0:
                    combo_box.setCurrentText('us')
                    line_edit.setText(str(duration//1000))
                else:
                    combo_box.setCurrentText('ns')
                    line_edit.setText(str(duration))

        self.bracket_exists = False
        self.drag_container = DragContainer()
        self.layout_dataset.addWidget(self.drag_container)
        for i, row_data in enumerate(self.data_matrix):
            self.add_column()
            row_widget = self.drag_container.items[i].widget
            row_layout = row_widget.layout()

            sublayout = row_layout.itemAt(0)  
            line_edit = sublayout.itemAt(1).widget() 
            combo_box = sublayout.itemAt(2).widget()  
                
            if isinstance(row_data[0], str):
                combo_box.setCurrentText('str (ns)')
                line_edit.setText(str(row_data[0]))
            else:
                duration = int(row_data[0])
                if duration%1000000==0:
                    combo_box.setCurrentText('ms')
                    line_edit.setText(str(duration//1000000))
                elif duration%1000==0:
                    combo_box.setCurrentText('us')
                    line_edit.setText(str(duration//1000))
                else:
                    combo_box.setCurrentText('ns')
                    line_edit.setText(str(duration))


            for j in range(1, 9):  
                checkbox = row_layout.itemAt(j).widget()
                checkbox.setChecked(bool(row_data[j]))

        if self.repeat_info[1]!=-1:
            # otherwise just default repeat_info setting and no bracket
            self.on_add_bracket(self.repeat_info[0], self.repeat_info[1])

        # load ref_info
        checkbox = self.btn_ref_info.layout().itemAt(0).widget()
        checkbox.setChecked(self.ref_info.get('is_ref', False))
        for ii, type in enumerate(['signal', 'DAQ', 'DAQ_ref', 'clock']):
            combobox = self.btn_ref_info.layout().itemAt(ii+1).layout().itemAt(1).widget()
            ch = self.ref_info.get(type, None)
            if ch is not None:
                combobox.setCurrentText(f'Ch{ch}')
            else:
                combobox.setCurrentText(f'None')

    def save_data(self):
        """
        save GUI state to self.device_handle.delay_array, self.device_handle.data_matrix 
        """
        self.device_handle.data_matrix = self.read_data()
        self.device_handle.delay_array = self.read_delay()
        self.device_handle.channel_names = self.read_channel_names()
        self.device_handle.repeat_info = self.read_repeat_info()
        self.device_handle.ref_info = self.read_ref_info()

                    
    def read_data(self):
        pulse_list = [item for item in self.drag_container.items if item.item_type=='pulse']
        count = len(pulse_list) #number of pulses 
        data_matrix = [[0]*9 for _ in range(count)] 

        for i in range(count):
            item = pulse_list[i]
            widget = item.widget
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

    def read_repeat_info(self):
        if not self.bracket_exists:
            return [0, -1, 1]
            # default repeat_info
        else:
            bracket_index_list = [ii for ii, item in enumerate(self.drag_container.items) if item.item_type!='pulse']
            start_index = bracket_index_list[0]
            end_index = bracket_index_list[1] - 2 # [start_index, end_index] pulses area inside bracket, include end_index

            widget = self.drag_container.items[bracket_index_list[1]].widget
            layout = widget.layout()
            item_sub = layout.itemAt(0)
            layout_sub = item_sub.layout()
            repeat = layout_sub.itemAt(1).widget().value()

            return [start_index, end_index, repeat]

    def read_ref_info(self):
        new_ref_info = {}
        checkbox = self.btn_ref_info.layout().itemAt(0).widget()
        new_ref_info['is_ref'] = checkbox.isChecked()
        combobox = self.btn_ref_info.layout().itemAt(1).layout().itemAt(1).widget()
        new_ref_info['signal'] = None if combobox.currentText()=='None' else int(combobox.currentText()[-1])
        combobox = self.btn_ref_info.layout().itemAt(2).layout().itemAt(1).widget()
        new_ref_info['DAQ'] = None if combobox.currentText()=='None' else int(combobox.currentText()[-1])
        combobox = self.btn_ref_info.layout().itemAt(3).layout().itemAt(1).widget()
        new_ref_info['DAQ_ref'] = None if combobox.currentText()=='None' else int(combobox.currentText()[-1])
        combobox = self.btn_ref_info.layout().itemAt(4).layout().itemAt(1).widget()
        new_ref_info['clock'] = None if combobox.currentText()=='None' else int(combobox.currentText()[-1])
        return new_ref_info
    
        
    def remove_column(self):
        pulse_list = [ii for ii, item in enumerate(self.drag_container.items) if item.item_type=='pulse']
        count = len(pulse_list)
        #print(count)
        if(count>=3):
            widget = self.drag_container.items[pulse_list[-1]].widget
            widget.deleteLater()
            self.drag_container.items = [item for ii, item in enumerate(self.drag_container.items) if ii!=pulse_list[-1]]
            self.drag_container.refresh_layout()
            
    def add_column(self):
        pulse_list = [item for item in self.drag_container.items if item.item_type=='pulse']
        count = len(pulse_list)

        row = QGroupBox('Pulse%d'%(count))
        row.setFont(QFont('Arial', 10))
        self.drag_container.add_item(row, "pulse")
        layout_data = QVBoxLayout(row)
        
        sublayout = QVBoxLayout()
        layout_data.addLayout(sublayout)
        btn = QLabel('Duration:')
        btn.setFixedSize(70,20)
        btn.setFont(QFont('Arial', 10))
        sublayout.addWidget(btn)
        btn = QLineEdit('10')
        btn.setFixedSize(70,20)
        btn.setFont(QFont('Arial', 10))
        sublayout.addWidget(btn)
        btn2 = QComboBox()
        btn2.addItems(['ns','us' ,'ms', 'str (ns)'])
        btn2.setFixedSize(70,20)
        btn2.setFont(QFont('Arial', 10))
        sublayout.addWidget(btn2)
        btn.textChanged.connect(lambda text, cb=btn2: self.handle_text_change(text, cb))
        
        for index in range(1, 9):
            btn = QCheckBox()
            btn.setText(self.channel_names_map[index-1])
            btn.setCheckable(True)
            btn.setFont(QFont('Arial', 10))
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
        global PULSES_INIT_DIR
        self.save_data()

        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self,'select',PULSES_INIT_DIR,'data_figure (*.npz)',options=options)

        if fileName == '':
            return

        if '.npz' in fileName:
            fileName = fileName[:-4]

        if '_pulse' in fileName:
            fileName = fileName[:-6]

        PULSES_INIT_DIR = os.path.dirname(fileName)
        self.device_handle.save_to_file(addr = fileName)

    def load_from_file(self):
        global PULSES_INIT_DIR
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,'select',PULSES_INIT_DIR,'data_figure (*.npz)',options=options)

        if fileName == '':
            return

        PULSES_INIT_DIR = os.path.dirname(fileName)
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

        if self.drag_container is None:
            return
        else:
            pulse_list = [item for item in self.drag_container.items if item.item_type=='pulse']
            for item in pulse_list:
                if item is not None:
                    widget = item.widget
                    layout = widget.layout()
                    item = layout.itemAt(channel+1)
                    item.widget().setText(f'{self.channel_names_map[channel]}')

            # replace pulse row




    def closeEvent(self, event):
        self.device_handle.data_matrix_tmp = self.read_data()
        self.device_handle.delay_array_tmp = self.read_delay() 
        self.device_handle.channel_names_tmp = self.read_channel_names() 
        self.device_handle.repeat_info_tmp = self.read_repeat_info()
        self.device_handle.ref_info_tmp = self.read_ref_info()


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
    app.setStyle('Windows')
    dialog = PulseGUI(device_handle)
    if is_in_GUI is True:
        return dialog
    else:
        dialog.exec_()

GUI_Pulse.__doc__ = PulseGUI.__doc__



def GUI_Load():
    global FIGS_INIT_DIR
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    app.setStyle('Windows')

    fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, "Select Data File", FIGS_INIT_DIR, "Data Files (*.npz *.jpg);;All Files (*)"
    )
    if fileName is not '':
        FIGS_INIT_DIR = os.path.dirname(fileName)
    
    return fileName