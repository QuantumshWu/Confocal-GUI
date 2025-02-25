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



class USB6346(BaseCounter, BaseScanner, metaclass=SingletonAndCloseMeta):
    """
    Class for NI DAQ USB-6346
    will be used for scanner: ao0, ao1 for X and Y of Galvo
    and for counter, 
    CTR1 uses PFI3, PFI4 for src and gate, 
    CTR2 (ref) uses PFI3, PFI5 for src and gate
    exit_handler method defines how to close task when exit
    """
    
    def __init__(self, exposure=1, port_config=None):

        import nidaqmx
        from nidaqmx.constants import AcquisitionType
        from nidaqmx.constants import TerminalConfiguration
        from nidaqmx.stream_readers import AnalogMultiChannelReader
        import warnings 

        if port_config is None:
            port_config = {'analog_signal':'ai0', 'analog_gate':'ai1', 'analog_gate_ref':'ai2',\
                           'apd_signal':'PFI3', 'apd_gate':'PFI4', 'apd_gate_ref':'PFI5'}
        self.port_config = port_config

        self.nidaqmx = nidaqmx
        self.task = nidaqmx.Task()
        self.task.ao_channels.add_ao_voltage_chan('Dev1/ao0', min_val=-5, max_val=5)
        self.task.ao_channels.add_ao_voltage_chan('Dev1/ao1', min_val=-5, max_val=5)
        self.task.start()

        self.counter_mode = None
        # analog, apd
        self.data_mode = None
        # single, ref_div, ref_sub, dual

        self.exposure = None
        self.tasks_to_close = [] # tasks need to be closed after swicthing counter mode  
        self.data_buffer = None
        self.reader = None
        # data_buffer for faster read

        self._x = 0
        self._y = 0
        self.x_lb = -5000
        self.x_ub = 5000
        self.y_lb = -5000
        self.y_ub = 5000

        self.clock = 1e4 # sampling rate for edge counting, defines when transfer counts from DAQ to PC, should be not too large to accomdate long exposure
        self.buffer_size = int(1e6)

    def gui(self):
        """
        Use self.gui_property and self.gui_property_type to determine how to display configurable parameters
        """
        self.gui_property = ['x', 'y']
        self.gui_property_type = ['float', 'float']
        GUI_Device(self)

    @property
    def valid_counter_mode(self):
        return ['analog', 'apd']

    @property
    def valid_data_mode(self):
        return ['single', 'ref_div', 'ref_sub', 'dual']

        
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



    def set_timing(self, exposure):

        # change match case to if elif to fit python before 3.10

        if self.counter_mode == 'apd':
            self.sample_num = int(round(self.clock*exposure))
            self.task_counter_ctr.timing.cfg_samp_clk_timing(self.clock, source = '/Dev1/Ctr3InternalOutput', \
                sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=self.buffer_size)
            self.task_counter_ctr_ref.timing.cfg_samp_clk_timing(self.clock, source = '/Dev1/Ctr3InternalOutput', \
                sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=self.buffer_size)

            self.exposure = exposure

            self.counts_main_array = np.zeros(self.sample_num+1, dtype=np.uint32)
            self.counts_ref_array = np.zeros(self.sample_num+1, dtype=np.uint32)
            self.reader_ctr = self.nidaqmx.stream_readers.CounterReader(self.task_counter_ctr.in_stream)
            self.reader_ctr_ref = self.nidaqmx.stream_readers.CounterReader(self.task_counter_ctr_ref.in_stream)

            self.task_counter_ctr.start()
            self.task_counter_ctr_ref.start()
            # start clock after counter tasks
            self.task_counter_clock.start()

        elif self.counter_mode == 'analog':
            self.clock = 500e3 # sampling rate for analog input, should be fast enough to capture gate signal for postprocessing
            self.sample_num = int(np.ceil(self.clock*exposure))
            self.task_counter_ai.stop()
            self.task_counter_ai.timing.cfg_samp_clk_timing(self.clock, sample_mode=self.nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=self.sample_num)
            self.exposure = exposure
            self.data_buffer = np.zeros((3, self.sample_num), dtype=np.float64)

        else:
            print(f'can only be one of the {self.valid_counter_mode}')

    def close_old_tasks(self):
        for task in self.tasks_to_close:
            task.stop()
            task.close()
        self.tasks_to_close = []

    def close(self):
        self.close_old_tasks()
        self.task.stop()
        self.task.close()

    def set_counter(self, counter_mode = 'apd'):
        if counter_mode == 'apd':
            self.close_old_tasks()
            self.clock = 1e4
            self.task_counter_ctr = self.nidaqmx.Task()
            self.task_counter_ctr.ci_channels.add_ci_count_edges_chan("Dev1/ctr1")
            # ctr1 source PFI3, gate PFI4
            self.task_counter_ctr.triggers.pause_trigger.dig_lvl_src = '/Dev1/'+self.port_config['apd_gate']
            self.task_counter_ctr.ci_channels.all.ci_count_edges_term = '/Dev1/'+self.port_config['apd_signal']
            self.task_counter_ctr.triggers.pause_trigger.trig_type = self.nidaqmx.constants.TriggerType.DIGITAL_LEVEL
            self.task_counter_ctr.triggers.pause_trigger.dig_lvl_when = self.nidaqmx.constants.Level.LOW

            self.task_counter_ctr_ref = self.nidaqmx.Task()
            self.task_counter_ctr_ref.ci_channels.add_ci_count_edges_chan("Dev1/ctr2")
            # ctr1 source PFI3, gate PFI5
            self.task_counter_ctr_ref.triggers.pause_trigger.dig_lvl_src = '/Dev1/'+self.port_config['apd_gate_ref']
            self.task_counter_ctr_ref.ci_channels.all.ci_count_edges_term = '/Dev1/'+self.port_config['apd_signal']
            self.task_counter_ctr_ref.triggers.pause_trigger.trig_type = self.nidaqmx.constants.TriggerType.DIGITAL_LEVEL
            self.task_counter_ctr_ref.triggers.pause_trigger.dig_lvl_when = self.nidaqmx.constants.Level.LOW

            self.task_counter_ctr.in_stream.relative_to = self.nidaqmx.constants.ReadRelativeTo.FIRST_SAMPLE
            self.task_counter_ctr_ref.in_stream.relative_to = self.nidaqmx.constants.ReadRelativeTo.FIRST_SAMPLE
            # relative to beginning of buffer, change offset instead

            self.task_counter_clock = self.nidaqmx.Task()
            self.task_counter_clock.co_channels.add_co_pulse_chan_freq(counter="Dev1/ctr3", freq=self.clock, duty_cycle=0.5)
            # ctr3 clock for buffered edge counting ctr1 and ctr2
            self.task_counter_clock.timing.cfg_implicit_timing(sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS)

            self.counter_mode = counter_mode
            self.tasks_to_close = [self.task_counter_ctr, self.task_counter_ctr_ref, self.task_counter_clock]

        elif counter_mode == 'analog':

            self.close_old_tasks()

            self.task_counter_ai = self.nidaqmx.Task()
            self.task_counter_ai.ai_channels.add_ai_voltage_chan('Dev1/'+self.port_config['analog_signal'])
            self.task_counter_ai.ai_channels.add_ai_voltage_chan('Dev1/'+self.port_config['analog_gate'])
            self.task_counter_ai.ai_channels.add_ai_voltage_chan('Dev1/'+self.port_config['analog_gate_ref'])
            # for analog counter
            self.task_counter_ai.start()
            self.counter_mode = counter_mode
            self.tasks_to_close = [self.task_counter_ai]
            self.reader = self.nidaqmx.stream_readers.AnalogMultiChannelReader(self.task_counter_ai.in_stream)

        else:
            print(f'can only be one of the {self.valid_counter_mode}')


    def read_counts(self, exposure, counter_mode = 'apd', data_mode='single',**kwargs):

        if exposure<1/self.clock:
            print('Exposure too short, change clock rate accordingly.')
            exposure = 1/self.clock

        if exposure>0.5*(self.buffer_size/self.clock) and self.counter_mode == 'apd':
            print('Exposure too long, change buffer size accordingly.')
            exposure = 0.5*(self.buffer_size/self.clock)

        if exposure<0.03:
            print('Exposure too short, this PC is slow, exposure can only larger than 0.03s.')
            exposure = 0.03

        self.data_mode = data_mode
        if (counter_mode != self.counter_mode) or (exposure != self.exposure):
            self.set_counter(counter_mode)
            self.set_timing(exposure)


        if self.counter_mode == 'analog':

            self.task_counter_ai.stop()
            self.task_counter_ai.start()
            time.sleep(exposure)
            self.reader.read_many_sample(self.data_buffer, number_of_samples_per_channel = self.sample_num)

            data = self.data_buffer[0, :]
            gate1 = self.data_buffer[1, :]
            gate2 = self.data_buffer[2, :]
            threshold = 2.7

            gate1_index = np.where(gate1 > threshold)[0]
            gate2_index = np.where(gate2 > threshold)[0]

            data_main = float(np.mean(data[gate1_index])) if len(gate1_index)!=0 else 0
            data_ref = float(np.mean(data[gate2_index])) if len(gate2_index)!=0 else 0

            # seems better than np.sum()/np.sum(), don't know why?
            # may due to finite sampling rate than they have different array length


        elif self.counter_mode == 'apd':
            total_sample = self.task_counter_ctr.in_stream.total_samp_per_chan_acquired
            self.task_counter_ctr.in_stream.offset = total_sample
            self.task_counter_ctr_ref.in_stream.offset = total_sample
            # update read pos accrodingly to keep reading most recent self.sample_num+1 samples
            self.reader_ctr.read_many_sample_uint32(self.counts_main_array\
                , number_of_samples_per_channel = (self.sample_num+1), timeout=self.nidaqmx.constants.WAIT_INFINITELY)
            self.reader_ctr_ref.read_many_sample_uint32(self.counts_ref_array\
                , number_of_samples_per_channel = (self.sample_num+1), timeout=self.nidaqmx.constants.WAIT_INFINITELY)

            data_main = float(self.counts_main_array[-1] - self.counts_main_array[-self.sample_num-1])
            data_ref = float(self.counts_ref_array[-1] - self.counts_ref_array[-self.sample_num-1])
        else:
            print(f'can only be one of the {self.valid_counter_mode}')


        if self.data_mode == 'single':

            return [data_main,]

        elif self.data_mode == 'ref_div':

            if data_main==0 or data_ref==0:
                return [0,]
            else:
                return [data_main/data_ref,]

        elif self.data_mode == 'ref_sub':
            return [(data_main - data_ref),]

        elif self.data_mode == 'dual':
            return [data_main, data_ref]

        else:
            print(f'can only be one of the {self.valid_data_mode}')


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

    def gui(self, is_in_GUI=False):
        GUI_Pulse(self, is_in_GUI)
    gui.__doc__ = GUI_Pulse.__doc__

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





    




