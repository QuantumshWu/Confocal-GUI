import numpy as np
import sys
import time
import threading
from decimal import Decimal
from threading import Event
from scipy.optimize import curve_fit
from abc import ABC, abstractmethod
import numbers

from Confocal_GUI.live_plot import *
from Confocal_GUI.gui import *
from .base import register_measurement, BaseMeasurement, live


class RFWithPulseMeasurement(BaseMeasurement):

    def device_to_state(self, duration):
        # move device state to x from data_x, defaul frequency in GHz
        self.rf.on = True
        self.rf.frequency = self.frequency
        self.pulse.x = duration
        self.pulse.on_pulse()


    def to_initial_state(self):
        # move device/data state to initial state before measurement
        self.rf.on = True
        self.rf.frequency = self.frequency


    def to_final_state(self):
        # move device/data state to final state after measurement
        self.rf.on = False
        self.pulse.off_pulse()

    def read_x(self):
        return self.pulse.x

    def assign_names(self):

        self.x_name = 'RF duration'
        self.x_unit = 'ns'
        self.measurement_name = None
        self.x_device_name = 'Pulse'
        self.plot_type = '1D'
        self.fit_func = None
        self.loaded_params = False

        # need to be override by specific measurement such as rabi, spinecho

        self.counter = self.config_instances.get('counter', None)
        self.rf = self.config_instances.get('rf', None)
        self.pulse = self.config_instances.get('pulse', None)
        self.scanner = self.config_instances.get('scanner', None)
        # init assignment
        if (self.counter is None) or (self.rf is None) or (self.pulse is None):
            raise KeyError('Missing devices in config_instances')


    def _load_params(self, data_x=None, exposure=0.1, power=None, frequency=None, pulse_file=None, config_instances=None, \
        repeat=1, is_GUI=False, \
        counter_mode='apd', data_mode='ref_sub', relim_mode='tight', update_mode='normal', is_plot=True):
        """
        rabi func doc str
        """
        self.loaded_params = True
        if data_x is None:
            data_x = np.arange(10, 1000, 10)
        self.data_x = data_x
       
        self.exposure = exposure
        self.repeat = repeat
        self.is_GUI = is_GUI
        self.counter_mode = counter_mode
        self.data_mode = data_mode
        self.relim_mode = relim_mode
        self.update_mode = update_mode

        self.power = self.rf.power if power is None else power
        self.frequency = self.rf.frequency if frequency is None else frequency*1e9

        if pulse_file is not None:
            self.pulse.load_from_file(pulse_file)
            # may load from 'rabi_pulse*'
        self.is_plot = is_plot
        self.info = {'measurement_name':self.measurement_name, 'plot_type':self.plot_type, 'exposure':self.exposure\
                    , 'repeat':self.repeat, 'power':self.power, 'frequency':self.frequency, \
                    'scanner':(None if self.scanner is None else (self.scanner.x, self.scanner.y)), \
                    'pulse':{'data_matrix': self.pulse.data_matrix, \
                              'delay_array': self.pulse.delay_array, \
                               'repeat_info': self.pulse.repeat_info, \
                               'channel_names': self.pulse.channel_names}}

    def plot(self, **kwargs):
        self.load_params(**kwargs)
        if not self.is_plot:
            return self

        if self.is_GUI:
            self.measurement_Live = live(config_instances=self.config_instances, is_plot=False)
            GUI_PLE(config_instances = self.config_instances, measurement_PLE=self, measurement_Live=self.measurement_Live)
            return None, None
        else:
            data_x = self.data_x
            data_y = self.data_y
            data_generator = self
            liveplot = PLELive(labels=[f'{self.x_name} ({self.x_unit})', f'Counts/{self.exposure}s'], \
                                update_time=0.1, data_generator=data_generator, data=[data_x, data_y], \
                                config_instances = self.config_instances, relim_mode=self.relim_mode)
            fig, selector = liveplot.plot()
            data_figure = DataFigure(liveplot)
            return fig, data_figure


def rabi(data_x=None, exposure=0.1, power=-10, frequency=2.88, pulse_file=None, config_instances=None, \
        repeat=1, is_GUI=False, \
        counter_mode='apd', data_mode='ref_sub', relim_mode='tight', is_plot=True):
    """
    rabi

    args:
    (data_x=None, exposure=0.1, power=-10, frequency=2.88, pulse_file=None, config_instances=None, 
        repeat=1, is_GUI=False, 
        counter_mode='apd', data_mode='ref_sub', relim_mode='tight'):

    example:
    fig, data_figure = rabi(data_x=np.arange(20, 2000, 10), exposure=0.1, power=-10, frequency=2.88, 
                            pulse_file=None,
                            config_instances=config_instances, repeat=1, is_GUI=False,
                            counter_mode='apd', data_mode='ref_sub', relim_mode='tight')

    notes:
    dafault pulse file at '../src/Confocal_GUI/device/Rabi_pulse*', edit using pulse.gui()

    pulse:
    Init -> RF(x) -> Readout

    """
    
    if pulse_file is None:
        pulse_file = '../src/Confocal_GUI/device/Rabi_pulse*'
    if data_x is None:
        data_x = np.arange(20, 2000, 10)

    measurement = RFWithPulseMeasurement(config_instances=config_instances)
    measurement.x_name = 'RF duration'
    measurement.x_unit = 'ns'
    measurement.measurement_name = 'Rabi'
    measurement.x_device_name = 'Pulse'
    measurement.plot_type = '1D'
    measurement.fit_func = 'rabi'

    return measurement.plot(data_x=data_x, exposure=exposure, power=power, frequency=frequency, pulse_file=pulse_file,\
                    config_instances=config_instances, repeat=repeat, is_GUI=is_GUI, counter_mode=counter_mode,\
                    data_mode=data_mode, relim_mode=relim_mode, is_plot=is_plot)

def ramsey(data_x=None, exposure=0.1, power=-10, frequency=2.88, pulse_file=None, config_instances=None, \
        repeat=1, is_GUI=False, \
        counter_mode='apd', data_mode='ref_sub', relim_mode='tight', is_plot=True):
    """
    ramsey

    args:
    (data_x=None, exposure=0.1, power=-10, frequency=2.88, pulse_file=None, config_instances=None, 
        repeat=1, is_GUI=False, 
        counter_mode='apd', data_mode='ref_sub', relim_mode='tight'):

    example:
    fig, data_figure = ramsey(data_x=np.arange(20, 2000, 10), exposure=0.1, power=-10, frequency=2.88, 
                            pulse_file=None,
                            config_instances=config_instances, repeat=1, is_GUI=False,
                            counter_mode='apd', data_mode='ref_sub', relim_mode='tight')

    notes:
    dafault pulse file at '../src/Confocal_GUI/device/Ramsey_pulse*', edit using pulse.gui()

    pulse:
    Init -> pi/2 -> x -> pi/2 -> Readout

    """

    if pulse_file is None:
        pulse_file = '../src/Confocal_GUI/device/Ramsey_pulse*'
    if data_x is None:
        data_x = np.arange(20, 2000, 10)

    measurement = RFWithPulseMeasurement(config_instances=config_instances)
    measurement.x_name = 'Gap'
    measurement.x_unit = 'ns'
    measurement.measurement_name = 'Ramsey'
    measurement.x_device_name = 'Pulse'
    measurement.plot_type = '1D'
    measurement.fit_func = 'rabi'

    return measurement.plot(data_x=data_x, exposure=exposure, power=power, frequency=frequency, pulse_file=pulse_file,\
                    config_instances=config_instances, repeat=repeat, is_GUI=is_GUI, counter_mode=counter_mode,\
                    data_mode=data_mode, relim_mode=relim_mode, is_plot=is_plot)

def spinecho(data_x=None, exposure=0.1, power=-10, frequency=2.88, pulse_file=None, config_instances=None, \
        repeat=1, is_GUI=False, \
        counter_mode='apd', data_mode='ref_sub', relim_mode='tight', is_plot=True):
    """
    spinecho

    args:
    (data_x=None, exposure=0.1, power=-10, frequency=2.88, pulse_file=None, config_instances=None, 
        repeat=1, is_GUI=False, 
        counter_mode='apd', data_mode='ref_sub', relim_mode='tight'):

    example:
    fig, data_figure = spinecho(data_x=np.arange(20, 2000, 10), exposure=0.1, power=-10, frequency=2.88, 
                            pulse_file=None,
                            config_instances=config_instances, repeat=1, is_GUI=False,
                            counter_mode='apd', data_mode='ref_sub', relim_mode='tight')

    notes:
    dafault pulse file at ../src/Confocal_GUI/device/Spinecho_pulse*', edit using pulse.gui()

    pulse:
    Init -> pi/2 -> x -> pi -> x -> pi/2 -> Readout

    """

    if pulse_file is None:
        pulse_file = '../src/Confocal_GUI/device/Spinecho_pulse*'
    if data_x is None:
        data_x = np.arange(20, 2000, 10)

    measurement = RFWithPulseMeasurement(config_instances=config_instances)
    measurement.x_name = 'half-Gap'
    measurement.x_unit = 'ns'
    measurement.measurement_name = 'Spinecho'
    measurement.x_device_name = 'Pulse'
    measurement.plot_type = '1D'
    measurement.fit_func = 'decay'

    return measurement.plot(data_x=data_x, exposure=exposure, power=power, frequency=frequency, pulse_file=pulse_file,\
                    config_instances=config_instances, repeat=repeat, is_GUI=is_GUI, counter_mode=counter_mode,\
                    data_mode=data_mode, relim_mode=relim_mode, is_plot=is_plot)


def roduration(data_x=None, exposure=0.1, power=-10, frequency=2.88, pulse_file=None, \
    config_instances=None, repeat=1, is_GUI=False, \
        counter_mode='apd', data_mode='ref_sub', relim_mode='tight', is_plot=True):
    """
    roduration

    args:
    (data_x=None, exposure=0.1, power=-10, frequency=2.88, pulse_file=None, config_instances=None, 
        repeat=1, is_GUI=False, 
        counter_mode='apd', data_mode='ref_sub', relim_mode='tight'):

    example:
    fig, data_figure = roduration(data_x=np.arange(-1000, 10000, 100), exposure=0.1, power=-10, frequency=2.88, 
                            pulse_file=None,
                            config_instances=config_instances, repeat=1, is_GUI=False,
                            counter_mode='apd', data_mode='ref_sub', relim_mode='tight')

    notes:
    dafault pulse file at ../src/Confocal_GUI/device/ROduration_pulse*', edit using pulse.gui()

    pulse:
    Init -> RF -> Readout
    delay(Green, 0)
    delay(DAQ, x)

    """

    if pulse_file is None:
        pulse_file = '../src/Confocal_GUI/device/ROduration_pulse*'
    if data_x is None:
        data_x = np.arange(-1000, 10000, 100)

    measurement = RFWithPulseMeasurement(config_instances=config_instances)
    measurement.x_name = 'DAQ read at'
    measurement.x_unit = 'ns'
    measurement.measurement_name = 'ROduration'
    measurement.x_device_name = 'Pulse'
    measurement.plot_type = '1D'
    measurement.fit_func = 'decay'

    return measurement.plot(data_x=data_x, exposure=exposure, power=power, frequency=frequency, pulse_file=pulse_file,\
                    config_instances=config_instances, repeat=repeat, is_GUI=is_GUI, counter_mode=counter_mode,\
                    data_mode=data_mode, relim_mode=relim_mode, is_plot=is_plot)


def t1(data_x=None, exposure=0.1, power=-10, frequency=2.88, pulse_file=None, config_instances=None, \
        repeat=1, is_GUI=False, \
        counter_mode='apd', data_mode='ref_sub', relim_mode='tight', is_plot=True):
    """
    t1

    args:
    (data_x=None, exposure=0.1, power=-10, frequency=2.88, pulse_file=None, config_instances=None, 
        repeat=1, is_GUI=False, 
        counter_mode='apd', data_mode='ref_sub', relim_mode='tight'):

    example:
    fig, data_figure = t1(data_x=np.arange(10, 300000, 1000), exposure=0.1, power=-10, frequency=2.88, 
                            pulse_file=None,
                            config_instances=config_instances, repeat=1, is_GUI=False,
                            counter_mode='apd', data_mode='ref_sub', relim_mode='tight')

    notes:
    dafault pulse file at ../src/Confocal_GUI/device/T1_pulse*', edit using pulse.gui()

    pulse:
    Init -> RF -> x -> Readout

    """

    if pulse_file is None:
        pulse_file = '../src/Confocal_GUI/device/T1_pulse*'
    if data_x is None:
        data_x = np.arange(10, 300000, 1000)

    measurement = RFWithPulseMeasurement(config_instances=config_instances)
    measurement.x_name = 'Gap'
    measurement.x_unit = 'ns'
    measurement.measurement_name = 'T1'
    measurement.x_device_name = 'Pulse'
    measurement.plot_type = '1D'
    measurement.fit_func = 'decay'

    return measurement.plot(data_x=data_x, exposure=exposure, power=power, frequency=frequency, pulse_file=pulse_file,\
                    config_instances=config_instances, repeat=repeat, is_GUI=is_GUI, counter_mode=counter_mode,\
                    data_mode=data_mode, relim_mode=relim_mode, is_plot=is_plot)
    


