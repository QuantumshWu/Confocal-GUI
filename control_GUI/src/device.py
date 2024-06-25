from abc import ABC, abstractmethod
import telnetlib, time
import sys
import clr
from System.Text import StringBuilder
from System import Int32
from System.Reflection import Assembly
clr.AddReference(r'mscorlib')
sys.path.append('C:\\Program Files\\New Focus\\New Focus Tunable Laser Application\\')
# location of new focus laser driver file
clr.AddReference('UsbDllWrap')
import Newport
import pyvisa
import nidaqmx
from nidaqmx.constants import AcquisitionType
from nidaqmx.constants import TerminalConfiguration
from nidaqmx.stream_readers import AnalogMultiChannelReader
import numpy as np
        

class Laser(ABC):
    """
    class for all lasers, only few methods to use
    """
    def __init__(self, model: str):
        self.model = model
        self._wavelength = None
        self._piezo = None
        
    @property
    @abstractmethod
    def wavelength(self):
        pass
        
    @wavelength.setter
    @abstractmethod
    def wavelength(self, value):
        pass
        
    @property
    @abstractmethod
    def piezo(self):
        pass
    
    @piezo.setter
    @abstractmethod
    def piezo(self, value):
        pass
    
    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def disconnect(self):
        pass
        
    
class TLB6700(Laser):
    """
    laser1 = TLB6700()
    
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
    
    def __tlb_open(self):
        self.tlb.OpenDevices(self.ProductID, True)

    def __tlb_close(self):
        self.tlb.CloseDevices()

    def __tlb_query(self, msg):
        self.answer.Clear()
        self.tlb.Query(self.DeviceKey, msg, self.answer)
        return self.answer.ToString()    
    
    def __init__(self):

        self.tlb = Newport.USBComm.USB()
        self.answer = StringBuilder(64)

        self.ProductID = 4106
        self.DeviceKey = '6700 SN37711'
        
        super().__init__('TLB-6700')
        self.connect()
        
        self.piezo_min = 0
        self.piezo_max = 100
        
    def connect(self):
        self.__tlb_open()
        
    def disconnect(self):
        self.__tlb_close()
        
    @property
    def wavelength(self):
        self._wavelength = float(self.__tlb_query('SOURCE:WAVELENGTH?'))
        return self._wavelength
    
    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value
        self.__tlb_query(f'SOURce:WAVElength {self._wavelength:.2f}')
        self.__tlb_query('OUTPut:TRACK 1')
        
    @property
    def piezo(self):
        self._piezo = float(self.__tlb_query('SOURce:VOLTage:PIEZO ?'))
        return self._piezo
    
    @piezo.setter
    def piezo(self, value):
        self._piezo = value
        self.__tlb_query(f'SOURce:VOLTage:PIEZO {self._piezo:.2f}')
        
class WaveMeter671():
    """
    Control code for 671 Wavelength Meter
    
    wavemeter671 = WaveMeter671()
    
    >>> wavemeter671.wavelength
    >>> 737.105033
    # read wavelength from wavemeter
    
    """
    

    def __init__(self):
        self.HOST = '10.199.199.1'
        self.tn = telnetlib.Telnet(self.HOST, timeout=1)
        self.tn.write(b'*IDN?\r\n')
        time.sleep(0.5)
        self.tn.read_very_eager()
        
    
    @property
    def wavelength(self):
        self.tn.write(b':READ:WAV?\r\n')
        self._wavelength = float(self.tn.expect([b'\r\n'])[-1])
        return self._wavelength
    
    def connect(self):
        self.tn = telnetlib.Telnet(self.HOST, timeout=1)
        self.tn.write(b'*IDN?\r\n')
        time.sleep(0.5)
        self.tn.read_very_eager()
    
    def disconnect(self):
        self.tn.close()

        
def read_counts(duration):
    """
    software gated counter for USB-6211, and reset pulse every time but maybe good enough
    """
    with nidaqmx.Task() as task:
        task.ci_channels.add_ci_count_edges_chan("Dev3/ctr0")
        task.triggers.pause_trigger.dig_lvl_src = '/Dev3/PFI1'
        task.triggers.pause_trigger.trig_type = nidaqmx.constants.TriggerType.DIGITAL_LEVEL
        task.triggers.pause_trigger.dig_lvl_when = nidaqmx.constants.Level.LOW
        task.start()
        time.sleep(duration)
        data_counts = task.read()
        task.stop()
    return data_counts


class AFG3052C():
    """
    class for scanner AFG3052C
    
    example:
    >>> afg3052c = AFG3052C()
    >>> afg3052c.x
    10
    >>> afg3052c.x = 20
    
    """
    def __init__(self):       
        rm = pyvisa.ResourceManager()
        self.handle = rm.open_resource('GPIB2::11::INSTR')
    
    @property
    def x(self):
        result_str = self.handle.query('SOURce1:VOLTage:LEVel:IMMediate:OFFSet?')
        self._x = int(1000*eval(result_str[:-1]))
        return self._x
    
    @x.setter
    def x(self, x_in):
        self.handle.write(f'SOURce1:VOLTage:LEVel:IMMediate:OFFSet {x_in}mV')
        result_str = self.handle.query('SOURce1:VOLTage:LEVel:IMMediate:OFFSet?')
        self._x = int(1000*eval(result_str[:-1]))
        
    @property
    def y(self):
        result_str = self.handle.query('SOURce2:VOLTage:LEVel:IMMediate:OFFSet?')
        self._y = int(1000*eval(result_str[:-1]))
        return self._y
    
    @y.setter
    def y(self, y_in):
        self.handle.write(f'SOURce2:VOLTage:LEVel:IMMediate:OFFSet {y_in}mV')
        result_str = self.handle.query('SOURce2:VOLTage:LEVel:IMMediate:OFFSet?')
        self._y = int(1000*eval(result_str[:-1]))
    


        
class USB6211():
    """
    class for counter NI USB-6211
    
    example
    
    >>> with USB6211(duration=1, clock_rate=1000, samples=1000) as counter:
    
    within block
    >>> counter.read_counts
    >>> 100
    >>> counter.read_bins
    >>> [1, 0, 1..., 1]
    
    
    can also call
    >>> counter.connect
    # start task
    >>> counter.disconnect
    # end task
    
    
    """
    
    def __init__(self, duration, clock_rate=None):
        if clock_rate==None:
            clock_rate = int((1/duration)*1000)

        self.duration = duration
        self.clock_rate = clock_rate
        self.samples = int(duration*clock_rate)
        
        
    def connect(self):
        co_task = nidaqmx.Task()
        co_task.co_channels.add_co_pulse_chan_freq('Dev3/ctr1', freq=self.clock_rate)
        co_task.timing.cfg_implicit_timing(sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        co_task.start()
        # set task for clock
        
        task = nidaqmx.Task()
        task.ci_channels.add_ci_count_edges_chan("Dev3/ctr0")
        task.timing.cfg_samp_clk_timing(rate=self.clock_rate, source='/Dev3/ctr1InternalOutput', \
                                        sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        
        task.start()
        # set task for counter
        
        self.co_task = co_task
        self.task = task
        
    def disconnect(self):
        self.task.stop()
        self.task.close()
        self.co_task.stop()
        self.co_task.close()
    
    
    def __enter__(self):
        self.connect()
        
    def __exit__(self):
        self.disconnect()
    
    def read_counts(self):
        self.task.in_stream.offset = 0
        self.task.in_stream.relative_to = nidaqmx.constants.ReadRelativeTo.MOST_RECENT_SAMPLE
        self.task.read(number_of_samples_per_channel=1)    
        self.task.in_stream.relative_to = nidaqmx.constants.ReadRelativeTo.MOST_RECENT_SAMPLE
        data = self.task.read(number_of_samples_per_channel=(self.samples+1))
        # read to enforce read the most recent information 'MOST_RECENT_SAMPLE', 
        # refer to https://github.com/ni/nidaqmx-python/issues/49
        # number_of_samples_per_channel=(self.samples+1) to make sure safely skip the first one
        return np.sum(np.diff(data, prepend=data[0]))
            
    def read_bins(self):        
        self.task.in_stream.offset = 0
        self.task.in_stream.relative_to = nidaqmx.constants.ReadRelativeTo.MOST_RECENT_SAMPLE
        self.task.read(number_of_samples_per_channel=1)    
        self.task.in_stream.relative_to = nidaqmx.constants.ReadRelativeTo.MOST_RECENT_SAMPLE
        data = self.task.read(number_of_samples_per_channel=(self.samples+1))
        # read to enforce read the most recent information 'MOST_RECENT_SAMPLE', 
        # refer to https://github.com/ni/nidaqmx-python/issues/49
        # number_of_samples_per_channel=(self.samples+1) to make sure safely skip the first one
        return np.diff(data, prepend=data[0])[1:]

