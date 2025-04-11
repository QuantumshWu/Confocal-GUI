from abc import ABC, abstractmethod
import time
import sys, os
import numpy as np
import threading
from .base import *
from Confocal_GUI.gui import *
                

class SGCounter(BaseCounter):
    """
    Software gated counter, using time.sleep, therefore
    duration stability may not be sufficienct for some cases
    """

    def __init__(self):
        pass

    @property
    def valid_counter_mode(self):
        return ['apd']

    @property
    def valid_data_mode(self):
        return ['single']

    def read_counts(exposure, counter_mode='apd', data_mode='single',**kwargs):
        """
        software gated counter for USB-6211, and reset pulse every time
        """
        with nidaqmx.Task() as task:
            task.ci_channels.add_ci_count_edges_chan("Dev3/ctr0")
            task.triggers.pause_trigger.dig_lvl_src = '/Dev3/PFI1'
            task.triggers.pause_trigger.trig_type = nidaqmx.constants.TriggerType.DIGITAL_LEVEL
            task.triggers.pause_trigger.dig_lvl_when = nidaqmx.constants.Level.LOW
            task.start()
            time.sleep(exposure)
            data_counts = task.read()
            task.stop()
        return data_counts


class TimeTaggerCounter(BaseCounter):
    """
    Uses timetagger as daq to count counts (edge counting) from APDs
    """

    def __init__(self, click_channel, begin_channel, end_channel, n_values=int(1e6)):
        from TimeTagger import createTimeTagger, Histogram, CountBetweenMarkers
        tagger = createTimeTagger('1809000LGG')
        tagger.reset()

        self.counter_handle = CountBetweenMarkers(tagger=tagger, click_channel=click_channel, begin_channel=begin_channel, \
            end_channel=end_channel, n_values=n_values)

    @property
    def valid_counter_mode(self):
        return ['apd']

    @property
    def valid_data_mode(self):
        return ['single']

    def read_counts(self, exposure, counter_mode='apd', data_mode='single', **kwargs):
        self.counter_handle.startFor(int(exposure*1e12))
        time.sleep(exposure)
        counts = np.sum(self.counter_handle.getData())
        self.counter_handle.stop()
        return counts



class USB6346(BaseCounterNI, BaseScanner, metaclass=SingletonAndCloseMeta):
    """
    Class for NI DAQ USB-6346
    will be used for scanner: ao0, ao1 for X and Y of Galvo
    and for counter, 
    CTR1 uses PFI3, PFI4 for src and gate, 
    CTR2 (ref) uses PFI3, PFI5 for src and gate
    exit_handler method defines how to close task when exit
    """

    def __init__(self, port_config=None):

        if port_config is None:
            port_config = {'analog_signal':'/Dev1/ai0', 'analog_gate':'/Dev1/ai1', 'analog_gate_ref':'/Dev1/ai2',\
                           'apd_signal':'/Dev1/PFI3', 'apd_gate':'/Dev1/PFI4', 'apd_gate_ref':'/Dev1/PFI5'}
        super().__init__(port_config=port_config)
        self.valid_counter_mode = ['analog', 'apd', 'apd_pg']
        self.valid_data_mode = ['single', 'ref_div', 'ref_sub', 'dual']
    

        self.task = self.nidaqmx.Task()
        self.task.ao_channels.add_ao_voltage_chan('Dev1/ao0', min_val=-5, max_val=5)
        self.task.ao_channels.add_ao_voltage_chan('Dev1/ao1', min_val=-5, max_val=5)
        self.task.start()

        self._x = 0
        self._y = 0
        self.x_lb = -5000
        self.x_ub = 5000
        self.y_lb = -5000
        self.y_ub = 5000

    def gui(self):
        """
        Use self.gui_property and self.gui_property_type to determine how to display configurable parameters
        """
        self.gui_property = ['x', 'y']
        self.gui_property_type = ['float', 'float']
        GUI_Device(self)
        
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = int(value) # in mV 
        self.task.write([self._x/1000, self._y/1000], auto_start=True) # in V
        
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        self._y = int(value) # in mV 
        self.task.write([self._x/1000, self._y/1000], auto_start=True) # in V


class DSG836(BaseRF, metaclass=SingletonAndCloseMeta):
    """
    Class for RF generator DSG836
    
    power in dbm
    
    frequency for frequency
    
    on for if output is on
    """
    
    def __init__(self, visa_str='USB0::0x1AB1::0x099C::DSG8M223900103::INSTR'):
        import pyvisa
        rm = pyvisa.ResourceManager()
        self.handle = rm.open_resource(visa_str)
        self._power = eval(self.handle.query('SOURce:Power?')[:-1])
        self._frequency = eval(self.handle.query('SOURce:FREQuency?')[:-1])
        self._on = False # if output is on
        self.power_ub = 10
        self.frequency_lb = 9e3
        self.frequency_ub = 3.6e9
        self.lock = threading.Lock() 

    def gui(self):
        """
        Use self.gui_property and self.gui_property_type to determine how to display configurable parameters
        """
        self.gui_property = ['on', 'power', 'frequency']
        self.gui_property_type = ['str', 'float', 'float']
        GUI_Device(self)
        
    @property
    def power(self):
        with self.lock:
            self._power = eval(self.handle.query('SOURce:Power?')[:-1])
            return self._power
    
    @power.setter
    def power(self, value):
        with self.lock:
            if value > self.power_ub:
                value = self.power_ub
                print(f'can not exceed RF power {self.power_ub}dbm')
            self._power = value
            self.handle.write(f'SOURce:Power {self._power}')
    
    @property
    def frequency(self):
        with self.lock:
            self._frequency = eval(self.handle.query('SOURce:Frequency?')[:-1])
            return self._frequency
    
    @frequency.setter
    def frequency(self, value):
        with self.lock:        
            self._frequency = value
            self.handle.write(f'SOURce:Frequency {self._frequency}')
        
        
    @property
    def on(self):
        with self.lock:
            return self._on
    
    @on.setter
    def on(self, value):
        with self.lock:
            self._on = value
            if value is True:
                # from False to True
                self.handle.write('OUTPut:STATe ON')
            else:
                # from True to False
                self.handle.write('OUTPut:STATe OFF')

    def close(self):
        pass
            
class Pulse(BasePulse):
    """
    Pulse class to pulse streamer control,
    call pulse.gui() to access to all functions and pulse sequence editing

    notes:
    pulse duration, delay can be a int in ns or str contains 'x' which will be later replaced by self.x,
    therefore enabling fast pulse control by self.x = x and self.on_pulse() without reconfiguring self.data_matrix
    and sel.delay_array

    Save Pulse: save pulse sequence to self.data_matrix, self.delay_array but not in file
    Save to file: save pulse sequence to a .npz file


    """

    def __init__(self, ip=None):
        super().__init__()
        from pulsestreamer import PulseStreamer, Sequence 
        self.PulseStreamer = PulseStreamer
        self.Sequence = Sequence
        if ip is None:
            self.ip = '169.254.8.2'
            # default ip address of pulse streamer
        else:
            self.ip = ip
        self.ps = PulseStreamer(self.ip)

    def off_pulse(self):


        # Create a sequence object
        sequence = self.ps.createSequence()
        pattern_off = [(1e3, 0), (1e3, 0)]
        for channel in range(0, 8):
            sequence.setDigital(channel, pattern_off)
        # Stream the sequence and repeat it indefinitely
        n_runs = self.PulseStreamer.REPEAT_INFINITELY
        self.ps.stream(self.Sequence.repeat(sequence, 8), n_runs)
        # need to repeat 8 times because pulse streamer will pad sequence to multiple of 8ns, otherwise unexpected changes of pulse
        
    def on_pulse(self):
        
        self.off_pulse()
        
        def check_chs(array): 
            # return a bool(0, 1) list for channels
            # defines the truth table of channels at a given period of pulse
            return array[1:]
        
        time_slices = self.read_data()
        sequence = self.ps.createSequence()

        for channel in range(0, 8):
            time_slice = time_slices[channel]
            count = len(time_slice)
            pattern = []
            # pattern is [(duration in ns, 1 for on or 0 for off), ...]
            pattern.append((time_slice[0][0], time_slice[0][1]))
            for i in range(count-2):
                pattern.append((time_slice[i+1][0], time_slice[i+1][1]))
            pattern.append((time_slice[-1][0], time_slice[-1][1]))

            sequence.setDigital(channel, pattern)


        # Stream the sequence and repeat it indefinitely
        n_runs = self.PulseStreamer.REPEAT_INFINITELY
        self.ps.stream(self.Sequence.repeat(sequence, 8), n_runs)

        time.sleep(0.1 + self.total_duration/1e9)
        # make sure pulse is stable and ready for measurement
        return time_slices





    




