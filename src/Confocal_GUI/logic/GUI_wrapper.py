import numpy as np
from .logic import *
from Confocal_GUI.gui import *


def GUI(config_instances, measurement_PL='pl', measurement_PLE='ple', measurement_Live='live'):
    """
    
    """

    measurement_Live_handle = (globals().get(measurement_Live))(**{'config_instances':config_instances, 'is_plot':False})
    measurement_PL_handle = (globals().get(measurement_PL))(**{'config_instances':config_instances, 'is_plot':False})
    measurement_PLE_handle = (globals().get(measurement_PLE))(**{'config_instances':config_instances, 'is_plot':False})

    return GUI_(config_instances, measurement_PLE=measurement_PLE_handle, measurement_PL=measurement_PL_handle\
        , measurement_Live=measurement_Live_handle, mode='PL_and_PLE')