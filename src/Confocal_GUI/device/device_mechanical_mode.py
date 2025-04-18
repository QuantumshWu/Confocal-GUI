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
    def __init__(self, ip=None, piezo_min=None, piezo_max=None):

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
        self.piezo_min = 60-25 if piezo_min is None else piezo_min
        self.piezo_max = 60+25 if piezo_max is None else piezo_max
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
    
    def __init__(self, wavelength_lb=None, wavelength_ub=None):
        super().__init__()
        self.ratio = 0.54 # +1V piezo -> +0.54GHz freq
        self.laser = config_instances.get('laser')
        self.spl = 299792458
        self.v_mid = 0.5*(self.laser.piezo_max + self.laser.piezo_min)
        self.v_min = self.laser.piezo_min + 0.05*(self.laser.piezo_max - self.laser.piezo_min)
        self.v_max = self.laser.piezo_min + 0.95*(self.laser.piezo_max - self.laser.piezo_min)
        self.freq_thre = 0.025 #25MHz threshold defines when to return is_ready
        self.P = 0.8 #scaling factor of PID control
        # leaves about 10% extra space
        self.wavelength_lb = 737.10 if wavelength_lb is None else wavelength_lb
        self.wavelength_ub = 737.125 if wavelength_ub is None else wavelength_ub

    def gui(self):
        """
        Use self.gui_property and self.gui_property_type to determine how to display configurable parameters
        """
        self.gui_property = ['on', 'wavelength']
        self.gui_property_type = ['str', 'float']
        GUI_Device(self)
        
    def _stabilizer_core(self):
        
        freq_desired = self.spl/self.wavelength
        while(1):
            wave_cache = self.wavemeter.wavelength #wait
            if wave_cache != 0:
                break
            else:
                time.sleep(0.1)
        freq_diff = freq_desired - self.spl/wave_cache#freq_desired - self.freq_recent
        if np.abs(freq_diff) <= self.freq_thre:
            self.is_ready = True
        else:
            self.is_ready = False
        v_diff = self.P*freq_diff/self.ratio 
        v_0 = self.laser.piezo
        if (v_0+v_diff)<self.v_min or (v_0+v_diff)>self.v_max:
            
            pass
        else:
            self.laser.piezo = v_0+v_diff

        time.sleep(0.2)# wait for piezo stable and measurement converge
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




class USB2120(BaseCounterNI, metaclass=SingletonAndCloseMeta):
    """
    Class for NI DAQ USB-2120
    """
    
    def __init__(self, port_config=None):

        if port_config is None:
            port_config = {'apd_signal':'/Dev2/PFI3', 'apd_gate':'/Dev2/PFI4', 'apd_gate_ref':'/Dev2/PFI1', 'apd_clock':'/Dev2/PFI12'}
        super().__init__(port_config=port_config)
        self.valid_counter_mode = ['apd', 'apd_pg']
        self.valid_data_mode = ['single', 'ref_div', 'ref_sub', 'dual']


class USB6009(BaseScanner, metaclass=SingletonAndCloseMeta):
    """
    Class for NI DAQ USB-6009
    """
    
    def __init__(self):

        import nidaqmx
        import warnings 

        self.nidaqmx = nidaqmx

        self.task = nidaqmx.Task()
        self.nidaqmx = nidaqmx
        self.task.ao_channels.add_ao_voltage_chan('Dev1/ao0', min_val=0, max_val=5)
        self.task.ao_channels.add_ao_voltage_chan('Dev1/ao1', min_val=0, max_val=5)
        self.task.start()

        self.tasks_to_close = [self.task,] # tasks need to be closed after swicthing counter mode  


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

    def close_old_tasks(self):
        for task in self.tasks_to_close:
            task.stop()
            task.close()
        self.tasks_to_close = []

    def close(self):
        self.close_old_tasks()

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

