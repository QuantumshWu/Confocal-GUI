import numpy as np
from .__init__ import *
# share all function names with __init__
from Confocal_GUI.gui import *


def GUI(measurement_PL='pl', measurement_PLE='ple', measurement_Live='live'):
    """
    GUI

    args:
    measurement_PLE can be any measurement of 1D plot_type, such as 'odmr', 'rabi', 'spinecho' etc.

    example:
    GUI() or GUI(measurement_PLE='odmr')

    GUI notes:

        Load file: 
        button to load any 1D or 2D saved measurements (*.npz) into GUI

        Read X,Y range/Read range: 
        button to read range from figure when there is a area selector enabled by left mouse

        Read X,Y/Read {measurement_PLE.x_name}: 
        button to read data from figure when there is a cross selector enabled by right mouse click

        Start/Stop {measurement_PLE/PL.measurement_name}: 
        button to start or stop live plot

        Set device: 
        button to select device from combobox on the right and open a gui for device for editting parameters

        Fit: 
        button to fit 1D plot using the fit function from the combo box on the right

        Apply/Bind set to scanner:
        Activate the number in Set X/Set Y or in {measurement_PLE.device_name}

    
    """

    measurement_Live_handle = (globals().get(measurement_Live))(**{'is_plot':False})
    measurement_PL_handle = (globals().get(measurement_PL))(**{'is_plot':False})
    measurement_PLE_handle = (globals().get(measurement_PLE))(**{'is_plot':False})

    return GUI_(measurement_PLE=measurement_PLE_handle, measurement_PL=measurement_PL_handle\
        , measurement_Live=measurement_Live_handle, mode='PL_and_PLE')