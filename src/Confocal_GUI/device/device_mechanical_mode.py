from abc import ABC, abstractmethod
import time
import sys, os
current_directory = os.path.dirname(os.path.abspath(__file__))
# location of new focus laser driver file
sys.path.append(current_directory)
import numpy as np
import functools
import threading
from .base import *
from Confocal_GUI.gui import *



class DLCpro(BaseLaser, metaclass=SingletonAndCloseMeta):
    """
    laser = DLCpro()
    
    >>> laser1.wavelength
    >>> 737.11
    >>> laser1.wavelength = 737.12
    >>> 737.12
    # return or set wavelngth
    
    >>> laser1.piezo
    >>> 0
    >>> laser1.piezo = 10
    >>> 10
    # return or set piezo voltage
        
    """     
    def __init__(self, ip=None):

        from toptica.lasersdk.dlcpro.v2_2_0 import DLCpro, NetworkConnection        
        if ip is None:
            ip = '128.223.23.108'
        self.ip = ip
        self.DLCpro = DLCpro
        self.NetworkConnection = NetworkConnection
        self.dlc = self.DLCpro(self.NetworkConnection(self.ip))
        self.dlc.__enter__()

        self._wavelength = None
        self._piezo = self.dlc.laser1.dl.pc.voltage_set.get()
        self.piezo_min = 68-25
        self.piezo_max = 68+25
        # piezo range where DLCpro is mode-hop free

    def gui(self):
        """
        Use self.gui_property and self.gui_property_type to determine how to display configurable parameters
        """
        self.gui_property = ['piezo',]
        self.gui_property_type = ['float',]
        GUI_Device(self)
        
    @property
    def wavelength(self):
        print('Cannot set wavelength, must manually set motor')
        return
    
    @wavelength.setter
    def wavelength(self, value):
        print('Cannot set wavelength, must manually set motor')
        return
        
    @property
    def piezo(self):
        self._piezo = self.dlc.laser1.dl.pc.voltage_set.get()
        return self._piezo
    
    @piezo.setter
    def piezo(self, value):
        if self.piezo_min<=value<=self.piezo_max:
            piezo = value
        else:
            print(f'Piezo {value} out of range, should be between {self.piezo_min} and {self.piezo_max}')
            return

        #with self.DLCpro(self.NetworkConnection(self.ip)) as dlc:
        self.dlc.laser1.dl.pc.voltage_set.set(piezo)
        self._piezo = value

    def close(self):
        self.dlc.__exit__()



class LaserStabilizerDLCpro(BaseLaserStabilizer, metaclass=SingletonAndCloseMeta):
    """
    core logic for stabilizer,
    
    .run() will read wavemeter and change laser piezo
    .is_ready=True when wavelength_desired = wavelength_actual
    .wavelength is the wavelength desired
    .ratio = -0.85GHz/V defines the ratio for feedback
    """
    
    def __init__(self, config_instances):
        super().__init__(config_instances=config_instances)
        self.ratio = 0.54 # +1V piezo -> +0.54GHz freq
        self.laser = config_instances.get('laser')
        self.spl = 299792458
        self.v_mid = 0.5*(self.laser.piezo_max + self.laser.piezo_min)
        self.v_min = self.laser.piezo_min + 0.05*(self.laser.piezo_max - self.laser.piezo_min)
        self.v_max = self.laser.piezo_min + 0.95*(self.laser.piezo_max - self.laser.piezo_min)
        self.freq_thre = 0.025 #25MHz threshold defines when to return is_ready
        self.P = 0.8 #scaling factor of PID control
        # leaves about 10% extra space
        self.wavelength_lb = 737.08
        self.wavelength_ub = 737.125

    def gui(self):
        """
        Use self.gui_property and self.gui_property_type to determine how to display configurable parameters
        """
        self.gui_property = ['on', 'wavelength']
        self.gui_property_type = ['str', 'float']
        GUI_Device(self)
        
    def _stabilizer_core(self):
        
        freq_desired = self.spl/self.wavelength
        wave_cache = self.wavemeter.wavelength
        freq_diff_guess = freq_desired - self.spl/wave_cache#freq_desired - self.freq_recent
        v_diff = self.P*freq_diff_guess/self.ratio 
        v_0 = self.laser.piezo
        #print(f'read wave {wave_cache}')
        if (v_0+v_diff)<self.v_min or (v_0+v_diff)>self.v_max:
            
            pass
        else:
            self.laser.piezo = v_0+v_diff

        time.sleep(0.2)# wait for piezo stable and measurement converge
        freq_actual = self.spl/self.wavemeter.wavelength #wait
        freq_diff = freq_desired - freq_actual
        if np.abs(freq_diff) <= self.freq_thre:
            self.is_ready = True
        else:
            self.is_ready = False
        return







class PulseSpinCore(BasePulse):
    """
    Pulse class to spincore control,
    call pulse.gui() to access to all functions and pulse sequence editing

    notes:
    pulse duration, delay can be a int in ns or str contains 'x' which will be later replaced by self.x,
    therefore enabling fast pulse control by self.x = x and self.on_pulse() without reconfiguring self.data_matrix
    and sel.delay_array

    Save Pulse: save pulse sequence to self.data_matrix, self.delay_array but not in file
    Save to file: save pulse sequence to a .npz file


    all pulse duration, delay array are round to mutiple time of 2ns


    """

    def __init__(self, ):
        super().__init__(t_resolution=(10, 2)) #10ns minimum width
        import spinapi 
        self.spinapi = spinapi

    def gui(self, is_in_GUI=False):
        GUI_Pulse(self, is_in_GUI)

    def _init(self):
        from spinapi import pb_set_debug, pb_get_version, pb_count_boards, pb_get_error, pb_core_clock, pb_init
        pb_set_debug(0)

        if pb_init() != 0:
            print("Error initializing board: %s" % pb_get_error())
            input("Please press a key to continue.")
            exit(-1)

        # Configure the core clock
        pb_core_clock(500)


    def off_pulse(self):
        from spinapi import pb_stop,pb_close
        try:
            self._init()
        except:
            pass
        pb_stop()
        pb_close()
        
    def on_pulse(self):
        from spinapi import pb_start_programming, pb_inst_pbonly, CONTINUE, BRANCH, pb_reset, pb_start\
        , pb_stop_programming, PULSE_PROGRAM

        ch1 = 0b000000000000000000000001
        ch2 = 0b000000000000000000000010
        ch3 = 0b000000000000000000000100
        ch4 = 0b000000000000000000001000
        ch5 = 0b000000000000000000010000
        ch6 = 0b000000000000000000100000
        ch7 = 0b000000000000000001000000
        ch8 = 0b000000000000000010000000

        channels = (ch1,
                  ch2,
                  ch3,
                  ch4,
                  ch5,
                  ch6,
                  ch7,
                  ch8)
        all_ch = 0b0
        for i in range(len(channels)):
            all_ch += channels[i]
        disable = 0b0#0b111000000000000000000000
        
        try:
            self._init()
        except:
            pass
        
        def check_chs(array):
            chs = 0b0
            for ii, i in enumerate(array[1:]):
                chs += channels[ii]*int(i)
            return chs
        
        data_matrix = self.read_data(type='data_matrix')
        count = len(data_matrix)
        # Program the pulse program

        pb_start_programming(PULSE_PROGRAM)


        start = pb_inst_pbonly(check_chs(data_matrix[0])+disable, CONTINUE, 0, data_matrix[0][0])
        for i in range(count-2):
            pb_inst_pbonly(check_chs(data_matrix[i+1])+disable, CONTINUE, 0, data_matrix[i+1][0])
        pb_inst_pbonly(check_chs(data_matrix[-1])+disable, BRANCH, start, data_matrix[-1][0])

        pb_stop_programming()

        # Trigger the board
        pb_reset() 
        pb_start()

        time.sleep(0.01 + self.total_duration/1e9)
        # make sure pulse is stable and ready for measurement




class USB2120(BaseCounter, metaclass=SingletonAndCloseMeta):
    """
    Class for NI DAQ USB-2120
    """
    
    def __init__(self, exposure=1, port_config=None):

        import nidaqmx
        from nidaqmx.constants import AcquisitionType
        from nidaqmx.constants import TerminalConfiguration
        from nidaqmx.stream_readers import AnalogMultiChannelReader
        import warnings 

        if port_config is None:
            port_config = {'apd_signal':'/Dev2/PFI3', 'apd_gate':'/Dev2/PFI4', 'apd_gate_ref':'/Dev2/PFI1'}
        self.port_config = port_config
        self.nidaqmx = nidaqmx

        self.task = nidaqmx.Task()
        self.nidaqmx = nidaqmx

        self.counter_mode = None
        # analog, apd
        self.data_mode = None
        # single, ref_div, ref_sub, dual

        self.exposure = None
        self.clock = 1e3 # sampling rate for edge counting, defines when transfer counts from DAQ to PC, should be not too large to accomdate long exposure
        self.read_n = 0
        self.read_start = False
        self.data_ready_event = threading.Event()
        self.callback_func = None
        self.tasks_to_close = [] # tasks need to be closed after swicthing counter mode  



    @property
    def valid_counter_mode(self):
        return ['apd',]

    @property
    def valid_data_mode(self):
        return ['single', 'ref_div', 'ref_sub', 'dual']

    def _callback_read(self, task, task_handle, event_type, number_of_samples, callback_data):
        if self.read_start:
            if self.read_n >= 10:
                self.counts_main_array = self.task_counter_ctr.read(number_of_samples_per_channel = -1)
                self.counts_ref_array = self.task_counter_ctr_ref.read(number_of_samples_per_channel = -1)
                self.read_n = 0
                self.read_start = False
                self.data_ready_event.set()
            else:
                self.read_n += 1
        else:
            _ = self.task_counter_ctr.read(number_of_samples_per_channel = -1)
            _ = self.task_counter_ctr_ref.read(number_of_samples_per_channel = -1)

        return 0


    def set_timing(self, exposure):

        # change match case to if elif to fit python before 3.10

        if self.counter_mode == 'apd':
            self.sample_num_div_10 = int(round(self.clock*exposure/10))
            self.sample_num = 10*self.sample_num_div_10
            self.task_counter_clock.stop()
            self.task_counter_ctr.stop()
            self.task_counter_ctr_ref.stop()
            self.task_counter_ctr.timing.cfg_samp_clk_timing(self.clock, source = '/Dev2/Ctr0InternalOutput', \
                sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=self.sample_num)
            self.task_counter_ctr_ref.timing.cfg_samp_clk_timing(self.clock, source = '/Dev2/Ctr0InternalOutput', \
                sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=self.sample_num)

            self.exposure = exposure

            if self.callback_func is not None:
                self.task_counter_ctr.unregister_every_n_samples_acquired_into_buffer_event(self.sample_num_div_10, \
                    None)
            self.callback_func = functools.partial(self._callback_read, self.task_counter_ctr)
            self.task_counter_ctr.register_every_n_samples_acquired_into_buffer_event(self.sample_num_div_10, \
                self.callback_func)
            # register call back for one of counter, only one

            self.task_counter_clock.start()
            self.task_counter_ctr.start()
            self.task_counter_ctr_ref.start()

        else:
            print(f'can only be one of the {self.valid_counter_mode}')



    def close_old_tasks(self):
        for task in self.tasks_to_close:
            task.stop()
            task.close()

        self.tasks_to_close = []

    def close(self):
        self.close_old_tasks()

    def set_counter(self, counter_mode = 'apd'):
        if counter_mode == 'apd':
            self.close_old_tasks()

            self.task_counter_ctr = self.nidaqmx.Task()
            self.task_counter_ctr.ci_channels.add_ci_count_edges_chan('/Dev2/ctr1')
            # ctr1 source PFI3, gate PFI4
            self.task_counter_ctr.triggers.pause_trigger.dig_lvl_src = self.port_config['apd_gate']
            self.task_counter_ctr.ci_channels.all.ci_count_edges_term = self.port_config['apd_signal']
            self.task_counter_ctr.triggers.pause_trigger.trig_type = self.nidaqmx.constants.TriggerType.DIGITAL_LEVEL
            self.task_counter_ctr.triggers.pause_trigger.dig_lvl_when = self.nidaqmx.constants.Level.LOW

            self.task_counter_ctr_ref = self.nidaqmx.Task()
            self.task_counter_ctr_ref.ci_channels.add_ci_count_edges_chan('/Dev2/ctr2')
            # ctr2 source PFI3, gate PFI1
            self.task_counter_ctr_ref.triggers.pause_trigger.dig_lvl_src = self.port_config['apd_gate_ref']
            self.task_counter_ctr_ref.ci_channels.all.ci_count_edges_term = self.port_config['apd_signal']
            self.task_counter_ctr_ref.triggers.pause_trigger.trig_type = self.nidaqmx.constants.TriggerType.DIGITAL_LEVEL
            self.task_counter_ctr_ref.triggers.pause_trigger.dig_lvl_when = self.nidaqmx.constants.Level.LOW


            self.task_counter_clock = self.nidaqmx.Task()
            self.task_counter_clock.co_channels.add_co_pulse_chan_freq(counter="Dev2/ctr0", freq=1e3, duty_cycle=0.5)
            # ctr3 clock for buffered edge counting ctr1 and ctr2
            self.task_counter_clock.timing.cfg_implicit_timing(sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS)

            self.counter_mode = counter_mode
            self.tasks_to_close = [self.task_counter_ctr, self.task_counter_ctr_ref, self.task_counter_clock]

        else:
            print(f'can only be one of the {self.valid_counter_mode}')


    def read_counts(self, exposure, counter_mode = 'apd', data_mode='single',**kwargs):
        if exposure<10/self.clock:
            print('Exposure too short, change clock rate accordingly.')
            exposure = 10/self.clock

        self.data_mode = data_mode
        if (counter_mode != self.counter_mode):
            self.set_counter(counter_mode)
            self.set_timing(exposure)
        elif (exposure != self.exposure):
            self.set_timing(exposure)


        if self.counter_mode == 'apd':
            self.read_n = 0
            self.data_ready_event.clear()
            self.read_start = True

            if self.data_ready_event.wait(timeout=self.exposure*10):
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


class AFG31152(BaseScanner):
    """
    class for scanner AFG31152
    
    example:
    >>> afg3052c = AFG3052C()
    >>> afg3052c.x
    10
    >>> afg3052c.x = 20
    
    """
    def __init__(self, visa_str = 'GPIB0::1::INSTR'):    
        import pyvisa   
        rm = pyvisa.ResourceManager()
        self.handle = rm.open_resource(visa_str)
        # default at x, y = 1182mV
        #
        self.lock = threading.Lock()

    def gui(self):
        """
        Use self.gui_property and self.gui_property_type to determine how to display configurable parameters
        """
        self.gui_property = ['x', 'y']
        self.gui_property_type = ['float', 'float']
        GUI_Device(self)
    
    @property
    def x(self):
        with self.lock:
            result_str = self.handle.query('SOURce1:VOLTage:LEVel:IMMediate:OFFSet?')
            self._x = int(1000*eval(result_str[:-1]))
            return self._x
    
    @x.setter
    def x(self, value):
        with self.lock:
            self.handle.write(f'SOURce1:VOLTage:LEVel:IMMediate:OFFSet {value}mV')
            result_str = self.handle.query('SOURce1:VOLTage:LEVel:IMMediate:OFFSet?')
            self._x = int(1000*eval(result_str[:-1]))
        
    @property
    def y(self):
        with self.lock:
            result_str = self.handle.query('SOURce2:VOLTage:LEVel:IMMediate:OFFSet?')
            self._y = int(1000*eval(result_str[:-1]))
            return self._y
    
    @y.setter
    def y(self, value):
        with self.lock:
            self.handle.write(f'SOURce2:VOLTage:LEVel:IMMediate:OFFSet {value}mV')
            result_str = self.handle.query('SOURce2:VOLTage:LEVel:IMMediate:OFFSet?')
            self._y = int(1000*eval(result_str[:-1]))

