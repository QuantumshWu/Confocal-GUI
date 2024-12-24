from abc import ABC, abstractmethod
import time
import sys, os
# location of new focus laser driver file
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)
import pyvisa
import numpy as np


def initialize_classes(config, lookup_dict):
    """
    config: dict 
        {
            "counter": {
                "type": "VirtualScanner.virtual_read_counts",
            },
            "scanner": {
                "type": "VirtualScanner",
            },
            ...
        }
    """
    instances = {}

    def get_callable_from_type(type_name, init_params):

        if "." in type_name:
            class_part, method_part = type_name.rsplit(".", 1)
            cls_or_obj = lookup_dict[class_part]
            if hasattr(cls_or_obj, "__bases__"):  
                obj = cls_or_obj(**init_params)
                return getattr(obj, method_part)
            else:
                raise ValueError(f"{class_part} is not a class, cannot get method {method_part}.")
        else:
            class_or_func = lookup_dict[type_name]
            if hasattr(class_or_func, "__bases__"):

                return class_or_func(**init_params)
            else:
                if init_params:
                    raise ValueError(
                        f"'{type_name}' is a function, but you provided extra params {init_params}."
                    )
                return class_or_func


    for key, raw_params in config.items():
        params = dict(raw_params)  
        if "config_instances" not in params:
            type_name = params.pop("type")
            result = get_callable_from_type(type_name, params)
            instances[key] = result


    for key, raw_params in config.items():
        params = dict(raw_params)
        if "config_instances" in params:
            type_name = params.pop("type")
            params.pop("config_instances")  
            class_or_func = lookup_dict[type_name]
            if hasattr(class_or_func, "__bases__"):

                obj = class_or_func(instances, **params)
                instances[key] = obj
            else:

                func = class_or_func
                instances[key] = func(instances, **params)


    for k, v in instances.items():
        print(f"{k} => {v}")
    return instances



class SingletonMeta(type):
    # make sure all devices only have one instance
    _instance_map = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance_map:
            cls._instance_map[cls] = super().__call__(*args, **kwargs)
        return cls._instance_map[cls]
        

class Laser(ABC):
    """
    class for all lasers, only few methods to use
    """


    def __init__(self, model: str):
        import clr
        from System.Text import StringBuilder
        from System import Int32
        from System.Reflection import Assembly
        import Newport
        clr.AddReference(r'mscorlib')
        sys.path.append('C:\\Program Files\\New Focus\\New Focus Tunable Laser Application\\')
        # location of new focus laser driver file
        clr.AddReference('UsbDllWrap')

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
        import telnetlib
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

        
def read_counts(duration, parent):
    """
    software gated counter for USB-6211, and reset pulse every time, but maybe good enough
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


class HighFiness():
    """
    class for HighFiness wavemeter
    
    wavelength returns wavelength in GHz
    
    need to unselect auto expsosure before start stabilizer
    """

    def __init__(self):

        import wlmData
        import wlmConst


        self.spl = 299792458
    
        DLL_PATH = "wlmData.dll"
        try:
            wlmData.LoadDLL(DLL_PATH)
        except:
            sys.exit("Error: Couldn't find DLL on path %s. Please check the DLL_PATH variable!" % DLL_PATH)

        # Checks the number of WLM server instance(s)
        if wlmData.dll.GetWLMCount(0) == 0:
            print("There is no running wlmServer instance(s).")
        else:
            # Read Type, Version, Revision and Build number
            Version_type = wlmData.dll.GetWLMVersion(0)
            Version_ver = wlmData.dll.GetWLMVersion(1)
            Version_rev = wlmData.dll.GetWLMVersion(2)
            Version_build = wlmData.dll.GetWLMVersion(3)
            print("WLM Version: [%s.%s.%s.%s]" % (Version_type, Version_ver, Version_rev, Version_build))
            
    @property
    def wavelength(self):
        # Read frequency
        Frequency = wlmData.dll.GetFrequency(0.0)
        if Frequency == wlmConst.ErrWlmMissing:
            StatusString = "WLM inactive"
        elif Frequency == wlmConst.ErrNoSignal:
            StatusString = 'No Signal'
        elif Frequency == wlmConst.ErrBadSignal:
            StatusString = 'Bad Signal'
        elif Frequency == wlmConst.ErrLowSignal:
            StatusString = 'Low Signal'
        elif Frequency == wlmConst.ErrBigSignal:
            StatusString = 'High Signal'
        else:
            StatusString = 'WLM is running'
            
        if Frequency == wlmConst.ErrOutOfRange:
            print("Ch1 Error: Out of Range")
        elif Frequency <= 0:
            print("Ch1 Error code: %d" % Frequency)
        else:
            pass#print("Ch1 Frequency: %.3f GHz" % Frequency)
            
        return self.spl/(float(Frequency)*1000)
    

class Keysight():
    """
    class for Keysight RF generator
    
    amp in V-pp
    frequency for frequency of sine wave
    """
    
    def __init__(self):
        rm = pyvisa.ResourceManager()
        self.handle = rm.open_resource('TCPIP0::localhost::hislip0::INSTR')
        self._power = 1
        self._frequency = 1.1e9
        self.handle.write(':ABOR')
        self._on = False
        self.addr = os.path.join(os.getcwd(), 'tmp.txt') # addr for txt file
        self.frequency = 1.1e9
        
    @property
    def on(self):
        return self._on
    
    @on.setter
    def on(self, on_in):
        self._on = on_in
        if on_in is True:
            # from False to True
            self.handle.write(':INIT:IMM')
        else:
            # from True to False
            self.handle.write(':ABOR')
        
    @property
    def power(self):
        return self._power
    
    @power.setter
    def power(self, power_in):
        self._power = power_in
        self.handle.write(f'VOLT {self._power}')
    
    @property
    def frequency(self):
        return self._frequency
    
    @frequency.setter
    def frequency(self, frequency_in):
        self.handle.write(':SOUR:FREQ:RAST 64000000000') # set sampling rate to 64G/s
        self._frequency = frequency_in
        self._generate_waveform_txt()
        self._load_waveform_txt()
        
    def _generate_waveform_txt(self):
        #if os.path.exists(self.addr):
        #    os.remove(self.addr)
        # aiming for 1/10000 frequency accuracy for 5GHz, and assuming 64GHz sampling rate
        # sample number must be 64*n otherwise error?
        n_period = int(round(1000/(64e9/self.frequency))) + 1
        n_sample = int(round(n_period*(64e9/self.frequency)))
        waveform = np.sin(np.linspace(0, 2*np.pi*n_period*16, n_sample*16))
        print('n', n_period, n_sample, np.min(waveform), np.max(waveform))
        np.savetxt(self.addr, waveform)
    
    def _load_waveform_txt(self):
        self.handle.write(f':TRAC1:IMP 1, "{self.addr}", TXT, IONLy, ON, ALEN')
        
def read_counts_6212(duration, parent):
    """
    software gated counter for USB-6211, and reset pulse every time, but maybe good enough
    """
    with nidaqmx.Task() as task:
        task.ci_channels.add_ci_count_edges_chan("Dev1/ctr1")
        task.triggers.pause_trigger.dig_lvl_src = '/Dev1/PFI4'
        task.triggers.pause_trigger.trig_type = nidaqmx.constants.TriggerType.DIGITAL_LEVEL
        task.triggers.pause_trigger.dig_lvl_when = nidaqmx.constants.Level.LOW
        task.start()
        time.sleep(duration)
        data_counts = task.read()
        task.stop()
    return data_counts


class TimeTaggerCounter():

    def __init__(self, click_channel, begin_channel, end_channel, n_values=int(1e6)):
        from TimeTagger import createTimeTagger, Histogram, CountBetweenMarkers
        tagger = createTimeTagger('1809000LGG')
        tagger.reset()

        self.counter_handle = CountBetweenMarkers(tagger=tagger, click_channel=click_channel, begin_channel=begin_channel, \
            end_channel=end_channel, n_values=n_values)

    def counter(self, duration, parent=None):
        self.counter_handle.startFor(int(duration*1e12))
        time.sleep(duration)
        counts = np.sum(self.counter_handle.getData())
        self.counter_handle.stop()
        return counts


class USB6212():
    """
    class for NI DAQ USB-6212
    
    will be used for scanner: ao0, ao1 for X and Y of Galvo
    
    and for counter, CTR1 PFI3, PFI4 for src and gate.
    
    exit_handler method defines how to close task when exit
    """

    
    def __init__(self):

        import atexit
        import nidaqmx
        from nidaqmx.constants import AcquisitionType
        from nidaqmx.constants import TerminalConfiguration
        from nidaqmx.stream_readers import AnalogMultiChannelReader


        self.task = nidaqmx.Task()
        self.task.ao_channels.add_ao_voltage_chan('Dev1/ao0', min_val=-1, max_val=1)
        self.task.ao_channels.add_ao_voltage_chan('Dev1/ao1', min_val=-1, max_val=1)
        
        #self.task.ci_channels.add_ci_count_edges_chan("Dev1/ctr0")
        #self.task.triggers.pause_trigger.dig_lvl_src = '/Dev1/PFI1'
        #self.task.triggers.pause_trigger.trig_type = nidaqmx.constants.TriggerType.DIGITAL_LEVEL
        #self.task.triggers.pause_trigger.dig_lvl_when = nidaqmx.constants.Level.LOW
        
        self.task.start()
        atexit.register(self.exit_handler)
        self._x = 0
        self._y = 0
        
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, x_in):
        self._x = int(x_in) # in mV 
        self.task.write([self._x/1000, self._y/1000], auto_start=True) # in V
        self.task.stop()
        
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, y_in):
        self._y = int(y_in) # in mV 
        self.task.write([self._x/1000, self._y/1000], auto_start=True) # in V
        self.task.stop()
    
    def read_counts(self, duration, parent):
        self.task.stop()
        self.task.start()
        time.sleep(duration)
        data_counts = self.task.read()
        self.task.stop()
        return data_counts
    
    def exit_handler(self):
        self.task.stop()
        self.task.close()


class USB6346(metaclass=SingletonMeta):
    """
    class for NI DAQ USB-6346
    
    will be used for scanner: ao0, ao1 for X and Y of Galvo
    
    and for counter, CTR1 PFI3, PFI4 for src and gate.
    
    exit_handler method defines how to close task when exit
    """
    
    def __init__(self, exposure=1):

        import atexit
        import nidaqmx
        from nidaqmx.constants import AcquisitionType
        from nidaqmx.constants import TerminalConfiguration
        from nidaqmx.stream_readers import AnalogMultiChannelReader
        import warnings 

        warnings.filterwarnings('ignore', category=nidaqmx.errors.DaqWarning)


        self.task = nidaqmx.Task()
        self.nidaqmx = nidaqmx
        self.task.ao_channels.add_ao_voltage_chan('Dev1/ao0', min_val=-1, max_val=1)
        self.task.ao_channels.add_ao_voltage_chan('Dev1/ao1', min_val=-1, max_val=1)
        
        self.task_counter_ai = nidaqmx.Task()
        self.task_counter_ai.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        self.exposure = None
        self.set_timing(exposure)
        #self.task.timing.cfg_samp_clk_timing(1000.0, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=1000)
        self.task_counter_ai.triggers.pause_trigger.dig_lvl_src = '/Dev1/PFI1'
        self.task_counter_ai.triggers.pause_trigger.trig_type = nidaqmx.constants.TriggerType.DIGITAL_LEVEL
        self.task_counter_ai.triggers.pause_trigger.dig_lvl_when = nidaqmx.constants.Level.LOW
        #self.task_counter_ai.triggers.pause_trigger.cfg_dig_lvl_pause_trig('/Dev1/PFI1', when=nidaqmx.constants.Level.LOW)
        self.task_counter_ai.in_stream.read_all_avail_samp = True
        
        self.task.start()
        self.task_counter_ai.start()
        atexit.register(self.exit_handler)
        self._x = 0
        self._y = 0

        
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, x_in):
        self._x = int(x_in) # in mV 
        self.task.write([self._x/1000, self._y/1000], auto_start=True) # in V
        self.task.stop()
        
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, y_in):
        self._y = int(y_in) # in mV 
        self.task.write([self._x/1000, self._y/1000], auto_start=True) # in V
        self.task.stop()


    def set_timing(self, exposure, clock=500000):
        self.clock = clock
        self.sample_num = int(round(self.clock*exposure))
        if exposure == self.exposure:
            return
        self.task_counter_ai.stop()
        self.exposure = exposure
        self.task_counter_ai.timing.cfg_samp_clk_timing(clock, sample_mode=self.nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=self.sample_num)
        #self.task_counter_ai.timing.cfg_samp_clk_timing(clock, sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=2000)

    def read_counts_inst(self, duration, parent):
        pass
    
    def read_counts(self, duration, parent):
        self.task_counter_ai.stop()
        self.set_timing(duration)
        self.task_counter_ai.start()
        time.sleep(duration)

        try:
            avai_samp = self.task_counter_ai.in_stream.avail_samp_per_chan
            avai_samp1 = self.task_counter_ai.in_stream.avail_samp_per_chan
            #print(self.task_counter_ai.in_stream.avail_samp_per_chan, 'all')
            data_array = self.task_counter_ai.read(self.nidaqmx.constants.READ_ALL_AVAILABLE)
            #print(len(data_array), self.clock, 'len')
            if len(data_array) == 0:
                data_counts = 0
            else:
                data_counts = float(np.sum(data_array)/self.sample_num)

        except Exception as e:
            data_counts = 0


        return data_counts
    
    @classmethod
    def exit_handler(cls):

        if cls._instance is not None:
            cls._instance.task.stop()
            cls._instance.task.close()

            cls._instance.task_counter_ai.stop()
            cls._instance.task_counter_ai.close()

            cls._instance = None
        
class SGS100A():
    """
    class for RF generator SGS100A
    
    power in dbm
    
    frequency for frequency
    
    iq for if iq is on
    
    on for if output is on
    """
    
    def __init__(self):
        rm = pyvisa.ResourceManager()
        self.handle = rm.open_resource('TCPIP::169.254.2.20::INSTR')
        self._power = int(self.handle.query('SOURce:Power?')[:-1])
        self._frequency = int(self.handle.query('SOURce:FREQuency?')[:-1])
        self._iq = False # if IQ modulation is on
        self._on = False # if output is on
        
    @property
    def power(self):
        self._power = int(self.handle.query('SOURce:Power?')[:-1])
        return self._power
    
    @power.setter
    def power(self, power_in):
        self._power = power_in
        self.handle.write(f'SOURce:Power {self._power}')
    
    @property
    def frequency(self):
        self._frequency = int(self.handle.query('SOURce:Frequency?')[:-1])
        return self._frequency
    
    @frequency.setter
    def frequency(self, frequency_in):
        self._frequency = frequency_in
        self.handle.write(f'SOURce:Frequency {self._frequency}')
        
        
    @property
    def on(self):
        return self._on
    
    @on.setter
    def on(self, on_in):
        #if on_in is self._on:
        #    return
        self._on = on_in
        if on_in is True:
            # from False to True
            self.handle.write('OUTPut:STATe ON')
        else:
            # from True to False
            self.handle.write('OUTPut:STATe OFF')
    
    @property
    def iq(self):
        return self._iq
    
    @iq.setter
    def iq(self, iq_in):
        if iq_in is self._iq:
            return
        
        self._iq = iq_in
        if iq_in is True:
            # iq from False to True
            self.handle.write('SOURce:IQ:IMPairment:STATe ON')
            self.handle.write('SOURce:IQ:STATe ON')
        else:
            # iq from True to False
            self.handle.write('SOURce:IQ:IMPairment:STATe OFF')
            self.handle.write('SOURce:IQ:STATe OFF')


class DSG836():
    """
    class for RF generator DSG836
    
    power in dbm
    
    frequency for frequency
    
    iq for if iq is on
    
    on for if output is on
    """
    
    def __init__(self):
        rm = pyvisa.ResourceManager()
        self.handle = rm.open_resource('USB0::0x1AB1::0x099C::DSG8M267M00006::INSTR')
        self._power = eval(self.handle.query('SOURce:Power?')[:-1])
        self._frequency = eval(self.handle.query('SOURce:FREQuency?')[:-1])
        self._iq = False # if IQ modulation is on
        self._on = False # if output is on
        
    @property
    def power(self):
        self._power = eval(self.handle.query('SOURce:Power?')[:-1])
        return self._power
    
    @power.setter
    def power(self, power_in):
        self._power = power_in
        self.handle.write(f'SOURce:Power {self._power}')
    
    @property
    def frequency(self):
        self._frequency = eval(self.handle.query('SOURce:Frequency?')[:-1])
        return self._frequency
    
    @frequency.setter
    def frequency(self, frequency_in):
        self._frequency = frequency_in
        self.handle.write(f'SOURce:Frequency {self._frequency}')
        
        
    @property
    def on(self):
        return self._on
    
    @on.setter
    def on(self, on_in):
        #if on_in is self._on:
        #    return
        self._on = on_in
        if on_in is True:
            # from False to True
            self.handle.write('OUTPut:STATe ON')
        else:
            # from True to False
            self.handle.write('OUTPut:STATe OFF')
    
    @property
    def iq(self):
        return self._iq
    
    @iq.setter
    def iq(self, iq_in):
        if iq_in is self._iq:
            return
        
        self._iq = iq_in
        if iq_in is True:
            # iq from False to True
            self.handle.write('SOURce:IQ:IMPairment:STATe ON')
            self.handle.write('SOURce:IQ:STATe ON')
        else:
            # iq from True to False
            self.handle.write('SOURce:IQ:IMPairment:STATe OFF')
            self.handle.write('SOURce:IQ:STATe OFF')
            


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


class AMI():
    def __init__(self):

        # x, y, z ip are strings
        import socket

        ip_dict = {"x": '169.254.157.64', "y": '169.254.26.126', "z": '169.254.155.71'}
        PORT = 7180
        magnet_state = [
            '0 Return Value Meaning',
            '1 RAMPING to target field/current',
            '2 HOLDING at the target field/current',
            '3 PAUSED',
            '4 Ramping in MANUAL UP mode',
            '5 Ramping in MANUAL DOWN mode',
            '6 ZEROING CURRENT (in progress)',
            '7 Quench detected',
            '8 At ZERO current',
            '9 Heating persistent switch',
            '10 Cooling persistent switch',
            '11 External Rampdown active',
        ]

        z = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        z.connect((ip_dict["z"], PORT))
        print(z.recv(2000).decode())

        #are these right for x and y?
        x = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        x.connect((ip_dict["x"], PORT))
        print(x.recv(2000).decode())

        y = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        y.connect((ip_dict["y"], PORT))
        print(y.recv(2000).decode())

        self.x = x
        self.y = y
        self.z = z
        self.magnet_state = magnet_state

    @property
    def field(self):
        self._field = [self.get_field(self.x), self.get_field(self.y), self.get_field(self.z)]
        return self._field

    @field.setter
    def field(self, field_in):

        if len(field_in)!=3:
            print('wrong field')
            return
        if max(field_in)>10 or min(field_in)<-10:
            print('wrong field')
            return

        self.set_target_field(self.x, field_in[0])
        self.set_target_field(self.y, field_in[1])
        self.set_target_field(self.z, field_in[2])

        self.ramp(self.x)
        self.ramp(self.y)
        self.ramp(self.z)

        while(1):
            time.sleep(0.5)
            state_x = self.get_state(self.x)
            state_y = self.get_state(self.y)
            state_z = self.get_state(self.z)

            if state_x==self.magnet_state[2] \
                and state_y==self.magnet_state[2] \
                and state_z==self.magnet_state[2]:
                print(f'x, y, z is {self.get_field(self.x)}, {self.get_field(self.y)}, {self.get_field(self.z)} (in kGauss)')
                break
            else:
                print(f'{self.field}'.ljust(50), end='\r')




    # below are private methods

    def get_field_unit(self, handler):
        handler.sendall("FIELD:UNITS?\n".encode())
        reply = handler.recv(2000).decode()
        reply = int(reply.strip())
        return ['kilogauss', 'tesla'][reply]

    def get_field(self, handler):
        handler.sendall("FIELD:Magnet?\n".encode())
        reply = handler.recv(2000).decode()
        return float(reply.strip())

    def get_error(self, handler):
        handler.sendall("SYSTEM:ERROR?\n".encode())
        reply = handler.recv(2000).decode()
        return reply.strip()

    def ramp(self, handler):
        handler.sendall("RAMP\n".encode())
        print('ramping')

    def set_target_field(self, handler, kilogauss):
        message = "CONFigure:FIELD:TARGet:{kilogauss:.5f}\n"
        message = message.format(kilogauss = kilogauss)
        print(message)
        handler.sendall(message.encode())

    def get_state(self, handler):
        handler.sendall("State?\n".encode())
        reply = self.magnet_state[int(handler.recv(2000).decode())]
        return reply

    def make_triangle(self, max_field, num_points):
        mat = np.linspace(0, max_field, num_points)
        mat = np.hstack([mat[:-1], mat[::-1]])
        return np.hstack([mat[:-1], -mat])


class Pulse():
    """
    class for calling pulse streamer in jupyter notebook, refer to pulse_stremer.py gui 

    self.set_timing(timing_matrix)
    timming matrix is n_sequence*(1+8) matrix, one for during, eight for channels
    duraing is in ns
    channels is 1 for on, 0 for off

    self.set_delay(delay_array)



    """

    def __init__(self, ip=None):
        from pulsestreamer import PulseStreamer, Sequence 
        if ip is None:
            self.ip = '169.254.8.2'
        else:
            self.ip = ip
        self.ps = PulseStreamer(self.ip)

        self.delay_array = np.array([0,]*8)

        self.data_matrix = np.array([[1e3, 1,1,1,1,1,1,1,1], [1e3, 1,1,1,1,1,1,1,1]])


    def off_pulse(self):


        # Create a sequence object
        sequence = self.ps.createSequence()

        # Create sequence and assign pattern to digital channel 0
        pattern_off = [(1e3, 0), (1e3, 0)]
        pattern_on = [(1e3, 1), (1e3, 1)]
        for channel in range(0, 8):
            sequence.setDigital(channel, pattern_off)# couter gate
        for channel in range(0, 2):
            sequence.setAnalog(channel, pattern_on)
        # Stream the sequence and repeat it indefinitely
        n_runs = PulseStreamer.REPEAT_INFINITELY
        self.ps.stream(sequence, n_runs)
        
    def on_pulse(self):
        
        self.off_pulse()
        time.sleep(0.5)
        
        def check_chs(array): # return a bool(0, 1) list for channels
            return array[1:]
        
        data_matrix = self.read_data()
        count = len(data_matrix)
        sequence = self.ps.createSequence()
        pattern_on = [(2, 1), (2, 1)]
        
        
        #if(count == 1):
        #    start = pb_inst_pbonly64(check_chs(data_matrix[0])+disable, Inst.CONTINUE, 0, data_matrix[0][0])
        #    pb_inst_pbonly64(check_chs(data_matrix[-1])+disable, Inst.BRANCH, start, data_matrix[-1][0])

        #else:
        #    start = pb_inst_pbonly64(check_chs(data_matrix[0])+disable, Inst.CONTINUE, 0, data_matrix[0][0])
        #    for i in range(count-2):
        #        pb_inst_pbonly64(check_chs(data_matrix[i+1])+disable, Inst.CONTINUE, 0, data_matrix[i+1][0])
         #   pb_inst_pbonly64(check_chs(data_matrix[-1])+disable, Inst.BRANCH, start, data_matrix[-1][0])

        for channel in range(0, 8):
            pattern = []
            pattern.append((data_matrix[0][0], check_chs(data_matrix[0])[channel]))
            for i in range(count-2):
                pattern.append((data_matrix[i+1][0], check_chs(data_matrix[i+1])[channel]))
            pattern.append((data_matrix[-1][0], check_chs(data_matrix[-1])[channel]))
            #print(channel, pattern)
            sequence.setDigital(channel, pattern)

        for channel in range(0, 2):
            sequence.setAnalog(channel, pattern_on)
        # Stream the sequence and repeat it indefinitely
        n_runs = PulseStreamer.REPEAT_INFINITELY
        #print(sequence.getData())
        #print('du', sequence.getDuration())
        self.ps.stream(sequence, n_runs)

    def set_timing_simple(self, timing_matrix):
        # set_timing_simple([[duration0, [channels0, ]], [], [], []])
        # eg. 
        # set_timing_simple([[100, (3)], [100, (3,5)]])
        # channel0 - channel7
        n_sequence = len(timing_matrix)
        data_matrix = np.zeros((n_sequence, 9))

        for i in range(n_sequence):
            data_matrix[i][0] = timing_matrix[i][0]

            for channel in timing_matrix[i][1]:
                data_matrix[i][channel+1] = 1


        self.data_matrix = data_matrix

    def set_timing(self, timing_matrix):
        # timming matrix is n_sequence*(1+8) matrix, one for during, eight for channels
        # duraing is in ns
        # channels is 1 for on, 0 for off
        self.data_matrix = timing_matrix

    def set_delay(self, delay_array):
        self.delay_array = delay_array



                    
    def read_data(self):
        
        data_delay_matrix = self.delay(self.data_matrix)
        
        return data_delay_matrix
    
    def delay(self, data_matrix):
        # add delay, separate by all channels' time slices
        
        def extract_time_slice(data_matrix):
            # extract data_matrix[:,i+1]'s time slice in format [(time_i, enable_i), ...] such that enable_i for time_i-1 to time_i
            time_slice_array = [[(0, 0)] for i in range(len(data_matrix[0]) - 1)]
            cur_time = 0
            for ii, pulse in enumerate(data_matrix):
                #print(pulse)
                cur_time += pulse[0]
                for channel in range(len(data_matrix[0]) - 1):
                    time_slice_array[channel].append((cur_time, pulse[channel+1]))

            return time_slice_array
    
        def combine_time_slice(time_slice_array):
            time_all = []
            for i, time_slice in enumerate(time_slice_array):
                for i, time_label in enumerate(time_slice):
                    if(time_label[0] not in time_all):
                        time_all.append(time_label[0])

            data_matrix = np.zeros((len(time_all), len(time_slice_array)+1))
            data_matrix[:, 0] = np.sort(time_all)

            time_all = np.sort(time_all)

            for i, time_slice in enumerate(time_slice_array):
                cur_ref_index = 0 #time_slice
                cur_status = 0
                for j in range(len(time_all)):
                    cur_status = time_slice[cur_ref_index + 1][1]
                    data_matrix[j, i+1] += cur_status
                    #print(time_slice, time_all, cur_status, i, j)
                    if(time_all[j]>=time_slice[cur_ref_index + 1][0]):
                        cur_ref_index += 1

            cur = 0
            last = 0
            for pulse in data_matrix[1:]:
                cur = pulse[0]
                pulse[0] = pulse[0] - last
                last = cur
            return np.array(data_matrix[1:], dtype=int)

        def delay_channel(time_slice_array, channel_i, delay_time):
            # channel_i from 0 to n
            total_time = time_slice_array[0][-1][0] # first channel, last time stamp, time
            delay_time = delay_time%total_time
            if delay_time==0:
                return time_slice_array
            time_slice = time_slice_array[channel_i]
            time_slice_delayed = []
            is_boundary = 0
            is_delay_at_boundary = 0
            for i, time_stamp in enumerate(time_slice[1:]):# skip (0,0) since the (total_time, i) works
                if not is_boundary and (time_stamp[0]+delay_time) >= total_time:
                    is_boundary = 1
                    boundary_i = i
                time_stamp_delayed = ((time_stamp[0]+delay_time)%total_time, time_stamp[1])
                if time_stamp_delayed[0]==0:
                    is_delay_at_boundary = 1
                time_slice_delayed.append(time_stamp_delayed)

            #print(boundary_i, time_slice_delayed)
            if is_delay_at_boundary:
                time_slice_delayed = [(0,0)] + time_slice_delayed[boundary_i:][1:] + time_slice_delayed[:boundary_i] \
                                    + [(total_time, time_slice_delayed[boundary_i][1])]
            else:
                time_slice_delayed = [(0,0)] + time_slice_delayed[boundary_i:] + time_slice_delayed[:boundary_i] \
                                    + [(total_time, time_slice_delayed[boundary_i][1])]
            time_slice_array[channel_i] = time_slice_delayed

            return time_slice_array

        def delay_sequence(data_matrix, channel_i, delay_time):
            time_slice_array = extract_time_slice(data_matrix)
            time_slice_array = delay_channel(time_slice_array, channel_i, delay_time)
            data_matrix = combine_time_slice(time_slice_array)
            return data_matrix
        
        for j in range(8):
            data_matrix =  delay_sequence(data_matrix, j, self.delay_array[j])
            
        return data_matrix


class Pulsev2():
    """
    class for calling pulse streamer in jupyter notebook, refer to pulse_stremer.py gui 

    self.set_timing(timing_matrix)
    timming matrix is n_sequence*(1+8) matrix, one for during, eight for channels
    duraing is in ns
    channels is 1 for on, 0 for off

    self.set_delay(delay_array)



    """

    def __init__(self, ip=None, analog_V=None):

        if ip is None:
            self.ip = '169.254.8.2'
        else:
            self.ip = ip
        self.ps = PulseStreamer(self.ip)

        self.delay_array = np.array([0,]*10)

        self.data_matrix = np.array([[1e3, 1,1,1,1,1,1,1,1,1,1], [1e3, 1,1,1,1,1,1,1,1,1,1]])

        if analog_V is None:
            self.analog_V = [1, 1]
        else:
            self.analog_V = analog_V


    def off_pulse(self):


        # Create a sequence object
        sequence = self.ps.createSequence()

        # Create sequence and assign pattern to digital channel 0
        pattern_off = [(1e3, 0), (1e3, 0)]
        pattern_on = [(1e3, 1), (1e3, 1)]
        for channel in range(0, 8):
            sequence.setDigital(channel, pattern_off)# couter gate
        for channel in range(8, 10):
            sequence.setAnalog(channel-8, pattern_off)
        # Stream the sequence and repeat it indefinitely
        n_runs = PulseStreamer.REPEAT_INFINITELY
        self.ps.stream(sequence, n_runs)
        
    def on_pulse(self):
        
        self.off_pulse()
        time.sleep(0.5)
        
        def check_chs(array): # return a bool(0, 1) list for channels
            return array[1:]
        
        data_matrix = self.read_data()
        count = len(data_matrix)
        sequence = self.ps.createSequence()
        pattern_on = [(2, 1), (2, 1)]
        
        
        #if(count == 1):
        #    start = pb_inst_pbonly64(check_chs(data_matrix[0])+disable, Inst.CONTINUE, 0, data_matrix[0][0])
        #    pb_inst_pbonly64(check_chs(data_matrix[-1])+disable, Inst.BRANCH, start, data_matrix[-1][0])

        #else:
        #    start = pb_inst_pbonly64(check_chs(data_matrix[0])+disable, Inst.CONTINUE, 0, data_matrix[0][0])
        #    for i in range(count-2):
        #        pb_inst_pbonly64(check_chs(data_matrix[i+1])+disable, Inst.CONTINUE, 0, data_matrix[i+1][0])
         #   pb_inst_pbonly64(check_chs(data_matrix[-1])+disable, Inst.BRANCH, start, data_matrix[-1][0])

        for channel in range(0, 8):
            pattern = []
            pattern.append((data_matrix[0][0], check_chs(data_matrix[0])[channel]))
            for i in range(count-2):
                pattern.append((data_matrix[i+1][0], check_chs(data_matrix[i+1])[channel]))
            pattern.append((data_matrix[-1][0], check_chs(data_matrix[-1])[channel]))
            #print(channel, pattern)
            sequence.setDigital(channel, pattern)

        for channel in range(8, 10):
            cur_V = self.analog_V[channel-8]
            pattern = []
            pattern.append((data_matrix[0][0], cur_V*check_chs(data_matrix[0])[channel]))
            for i in range(count-2):
                pattern.append((data_matrix[i+1][0], cur_V*check_chs(data_matrix[i+1])[channel]))
            pattern.append((data_matrix[-1][0], cur_V*check_chs(data_matrix[-1])[channel]))
            sequence.setAnalog(channel-8, pattern)
        # Stream the sequence and repeat it indefinitely
        n_runs = PulseStreamer.REPEAT_INFINITELY
        #print(sequence.getData())
        #print('du', sequence.getDuration())
        self.ps.stream(sequence, n_runs)

    def set_timing_simple(self, timing_matrix):
        # set_timing_simple([[duration0, [channels0, ]], [], [], []])
        # eg. 
        # set_timing_simple([[100, (3)], [100, (3,5)]])
        # channel0 - channel7
        n_sequence = len(timing_matrix)
        data_matrix = np.zeros((n_sequence, 11))

        for i in range(n_sequence):
            data_matrix[i][0] = timing_matrix[i][0]

            for channel in timing_matrix[i][1]:
                data_matrix[i][channel+1] = 1


        self.data_matrix = data_matrix

    def set_timing(self, timing_matrix):
        # timming matrix is n_sequence*(1+8) matrix, one for during, eight for channels
        # duraing is in ns
        # channels is 1 for on, 0 for off
        self.data_matrix = timing_matrix

    def set_delay(self, delay_array):
        self.delay_array = delay_array



                    
    def read_data(self):
        
        data_delay_matrix = self.delay(self.data_matrix)
        
        return data_delay_matrix
    
    def delay(self, data_matrix):
        # add delay, separate by all channels' time slices
        
        def extract_time_slice(data_matrix):
            # extract data_matrix[:,i+1]'s time slice in format [(time_i, enable_i), ...] such that enable_i for time_i-1 to time_i
            time_slice_array = [[(0, 0)] for i in range(len(data_matrix[0]) - 1)]
            cur_time = 0
            for ii, pulse in enumerate(data_matrix):
                #print(pulse)
                cur_time += pulse[0]
                for channel in range(len(data_matrix[0]) - 1):
                    time_slice_array[channel].append((cur_time, pulse[channel+1]))

            return time_slice_array
    
        def combine_time_slice(time_slice_array):
            time_all = []
            for i, time_slice in enumerate(time_slice_array):
                for i, time_label in enumerate(time_slice):
                    if(time_label[0] not in time_all):
                        time_all.append(time_label[0])

            data_matrix = np.zeros((len(time_all), len(time_slice_array)+1))
            data_matrix[:, 0] = np.sort(time_all)

            time_all = np.sort(time_all)

            for i, time_slice in enumerate(time_slice_array):
                cur_ref_index = 0 #time_slice
                cur_status = 0
                for j in range(len(time_all)):
                    cur_status = time_slice[cur_ref_index + 1][1]
                    data_matrix[j, i+1] += cur_status
                    #print(time_slice, time_all, cur_status, i, j)
                    if(time_all[j]>=time_slice[cur_ref_index + 1][0]):
                        cur_ref_index += 1

            cur = 0
            last = 0
            for pulse in data_matrix[1:]:
                cur = pulse[0]
                pulse[0] = pulse[0] - last
                last = cur
            return np.array(data_matrix[1:], dtype=int)

        def delay_channel(time_slice_array, channel_i, delay_time):
            # channel_i from 0 to n
            total_time = time_slice_array[0][-1][0] # first channel, last time stamp, time
            delay_time = delay_time%total_time
            if delay_time==0:
                return time_slice_array
            time_slice = time_slice_array[channel_i]
            time_slice_delayed = []
            is_boundary = 0
            is_delay_at_boundary = 0
            for i, time_stamp in enumerate(time_slice[1:]):# skip (0,0) since the (total_time, i) works
                if not is_boundary and (time_stamp[0]+delay_time) >= total_time:
                    is_boundary = 1
                    boundary_i = i
                time_stamp_delayed = ((time_stamp[0]+delay_time)%total_time, time_stamp[1])
                if time_stamp_delayed[0]==0:
                    is_delay_at_boundary = 1
                time_slice_delayed.append(time_stamp_delayed)

            #print(boundary_i, time_slice_delayed)
            if is_delay_at_boundary:
                time_slice_delayed = [(0,0)] + time_slice_delayed[boundary_i:][1:] + time_slice_delayed[:boundary_i] \
                                    + [(total_time, time_slice_delayed[boundary_i][1])]
            else:
                time_slice_delayed = [(0,0)] + time_slice_delayed[boundary_i:] + time_slice_delayed[:boundary_i] \
                                    + [(total_time, time_slice_delayed[boundary_i][1])]
            time_slice_array[channel_i] = time_slice_delayed

            return time_slice_array

        def delay_sequence(data_matrix, channel_i, delay_time):
            time_slice_array = extract_time_slice(data_matrix)
            time_slice_array = delay_channel(time_slice_array, channel_i, delay_time)
            data_matrix = combine_time_slice(time_slice_array)
            return data_matrix
        
        for j in range(10):
            data_matrix =  delay_sequence(data_matrix, j, self.delay_array[j])
            
        return data_matrix


    




