from abc import ABC, abstractmethod
import time
import sys, os
current_directory = os.path.dirname(os.path.abspath(__file__))
# location of new focus laser driver file
sys.path.append(current_directory)
import numpy as np
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
        self.piezo_min = 68-20
        self.piezo_max = 68+20
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
        self.freq_thre = 0.05 #50MHz threshold defines when to return is_ready
        self.P = 0.8 #scaling factor of PID control
        # leaves about 10% extra space
        self.wavelength_lb = 737.09
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
