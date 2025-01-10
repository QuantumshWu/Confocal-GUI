from abc import ABC, abstractmethod
import time
import sys, os
# location of new focus laser driver file
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)
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

        #warnings.filterwarnings('ignore', category=nidaqmx.errors.DaqWarning)
        self.nidaqmx = nidaqmx

        self.task = nidaqmx.Task()
        self.nidaqmx = nidaqmx
        self.task.ao_channels.add_ao_voltage_chan('Dev1/ao0', min_val=-5, max_val=5)
        self.task.ao_channels.add_ao_voltage_chan('Dev1/ao1', min_val=-5, max_val=5)
        self.task.start()

        self.counter_mode = None
        # analog, apd
        self.valid_counter_mode = ['analog', 'apd']
        self.data_mode = None
        self.valid_data_mode = ['single', 'ref_div', 'ref_sub', 'dual']
        # single, ref_div, ref_sub, dual

        self.exposure = None
        self.clock = None
        self.tasks_to_close = [] # tasks need to be closed after swicthing counter mode  
        self.data_buffer = None
        self.reader = None
        # data_buffer for faster read

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


    def set_timing(self, exposure):

        # change match case to if elif to fit python before 3.10

        if self.counter_mode == 'apd':
            self.clock = 1e3 # sampling rate for edge counting, defines when transfer counts from DAQ to PC, should be not too large to accomdate long exposure
            self.sample_num = int(round(self.clock*exposure))
            self.task_counter_ctr.stop()
            self.task_counter_ctr_ref.stop()
            self.task_counter_ctr.timing.cfg_samp_clk_timing(self.clock, source = '/Dev1/Ctr3InternalOutput', \
                sample_mode=self.nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=self.sample_num)
            self.task_counter_ctr_ref.timing.cfg_samp_clk_timing(self.clock, source = '/Dev1/Ctr3InternalOutput', \
                sample_mode=self.nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=self.sample_num)

            self.exposure = exposure

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

    def set_counter(self, counter_mode = 'apd'):
        if counter_mode == 'apd':

            self.close_old_tasks()

            self.task_counter_ctr = self.nidaqmx.Task()
            self.task_counter_ctr.ci_channels.add_ci_count_edges_chan("Dev1/ctr1")
            # ctr1 source PFI3, gate PFI4
            self.task_counter_ctr.triggers.pause_trigger.dig_lvl_src = '/Dev1/PFI4'
            self.task_counter_ctr.ci_channels.all.ci_count_edges_term = '/Dev1/PFI3'
            self.task_counter_ctr.triggers.pause_trigger.trig_type = self.nidaqmx.constants.TriggerType.DIGITAL_LEVEL
            self.task_counter_ctr.triggers.pause_trigger.dig_lvl_when = self.nidaqmx.constants.Level.LOW

            self.task_counter_ctr_ref = self.nidaqmx.Task()
            self.task_counter_ctr_ref.ci_channels.add_ci_count_edges_chan("Dev1/ctr2")
            # ctr1 source PFI3, gate PFI5
            self.task_counter_ctr_ref.triggers.pause_trigger.dig_lvl_src = '/Dev1/PFI5'
            self.task_counter_ctr_ref.ci_channels.all.ci_count_edges_term = '/Dev1/PFI3'
            self.task_counter_ctr_ref.triggers.pause_trigger.trig_type = self.nidaqmx.constants.TriggerType.DIGITAL_LEVEL
            self.task_counter_ctr_ref.triggers.pause_trigger.dig_lvl_when = self.nidaqmx.constants.Level.LOW

            self.task_counter_clock = self.nidaqmx.Task()
            self.task_counter_clock.co_channels.add_co_pulse_chan_freq(counter="Dev1/ctr3", freq=1e3, duty_cycle=0.5)
            # ctr3 clock for buffered edge counting ctr1 and ctr2
            self.task_counter_clock.timing.cfg_implicit_timing(sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS)
            self.task_counter_clock.start()
            self.task_counter_ctr.start()
            self.task_counter_ctr_ref.start()

            self.counter_mode = counter_mode
            self.tasks_to_close = [self.task_counter_ctr, self.task_counter_ctr_ref, self.task_counter_clock]

        elif counter_mode == 'analog':

            self.close_old_tasks()

            self.task_counter_ai = self.nidaqmx.Task()
            self.task_counter_ai.ai_channels.add_ai_voltage_chan("Dev1/ai0")
            self.task_counter_ai.ai_channels.add_ai_voltage_chan("Dev1/ai1")
            self.task_counter_ai.ai_channels.add_ai_voltage_chan("Dev1/ai2")
            # for analog counter
            self.task_counter_ai.start()
            self.counter_mode = counter_mode
            self.tasks_to_close = [self.task_counter_ai]
            self.reader = self.nidaqmx.stream_readers.AnalogMultiChannelReader(self.task_counter_ai.in_stream)

        else:
            print(f'can only be one of the {self.valid_counter_mode}')


    def read_counts(self, exposure, parent=None, counter_mode = 'apd', data_mode = 'single'):

        self.data_mode = data_mode
        if (counter_mode != self.counter_mode):
            self.set_counter(counter_mode)
            self.set_timing(exposure)
        elif (exposure != self.exposure):
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
            data_main = float(np.mean(data[gate1 > threshold]))
            data_ref = float(np.mean(data[gate2 > threshold]))

            # seems better than np.sum()/np.sum(), don't know why?
            # may due to finite sampling rate than they have different array length


        elif self.counter_mode == 'apd':

            self.task_counter_clock.stop()
            self.task_counter_ctr.stop()
            self.task_counter_ctr_ref.stop()
            self.task_counter_ctr.start()
            self.task_counter_ctr_ref.start()
            self.task_counter_clock.start()

            time.sleep(exposure)

            counts_main_array = self.task_counter_ctr.read(number_of_samples_per_channel = self.sample_num)
            counts_ref_array = self.task_counter_ctr_ref.read(number_of_samples_per_channel = self.sample_num)
            data_main = float(counts_main_array[-1] - counts_main_array[0])
            data_ref = float(counts_ref_array[-1] - counts_ref_array[0])


        else:
            print(f'can only be one of the {self.valid_counter_mode}')


        if self.data_mode == 'single':

            return data_main

        elif self.data_mode == 'ref_div':

            if data_main==0 or data_ref==0:
                return 0
            else:
                return data_main/data_ref

        elif self.data_mode == 'ref_sub':
            return data_main - data_ref

        elif self.data_mode == 'dual':
            return data_main, data_ref

        else:
            print(f'can only be one of the {self.valid_data_mode}')
    
    @classmethod
    def exit_handler(cls):

        if cls._instance is not None:

            cls._instance.close_old_tasks()


            cls._instance = None


class USB6346_old(metaclass=SingletonMeta):
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

        #warnings.filterwarnings('ignore', category=nidaqmx.errors.DaqWarning)


        self.task = nidaqmx.Task()
        self.nidaqmx = nidaqmx
        self.task.ao_channels.add_ao_voltage_chan('Dev1/ao0', min_val=-5, max_val=5)
        self.task.ao_channels.add_ao_voltage_chan('Dev1/ao1', min_val=-5, max_val=5)
        
        self.task_counter_ai = nidaqmx.Task()
        self.task_counter_ai.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        self.task_counter_ai.ai_channels.add_ai_voltage_chan("Dev1/ai1")
        self.task_counter_ai.ai_channels.add_ai_voltage_chan("Dev1/ai2")
        # for analog counter

        self.task_counter_ctr = nidaqmx.Task()
        self.task_counter_ctr.ci_channels.add_ci_count_edges_chan("Dev1/ctr1")
        # ctr1 source PFI3, gate PFI4
        self.task_counter_ctr.triggers.pause_trigger.dig_lvl_src = '/Dev1/PFI4'
        #self.task_counter_ctr.triggers.start_trigger.dig_lvl_src = '/Dev1/PFI4'
        self.task_counter_ctr.ci_channels.all.ci_count_edges_term = '/Dev1/PFI3'
        self.task_counter_ctr.triggers.pause_trigger.trig_type = nidaqmx.constants.TriggerType.DIGITAL_LEVEL
        self.task_counter_ctr.triggers.pause_trigger.dig_lvl_when = nidaqmx.constants.Level.LOW


        self.task_counter_ctr2 = nidaqmx.Task()
        self.task_counter_ctr2.ci_channels.add_ci_count_edges_chan("Dev1/ctr2")
        # ctr1 source PFI3, gate PFI4
        self.task_counter_ctr2.triggers.pause_trigger.dig_lvl_src = '/Dev1/PFI5'
        #self.task_counter_ctr2.triggers.start_trigger.dig_lvl_src = '/Dev1/PFI4'
        self.task_counter_ctr2.ci_channels.all.ci_count_edges_term = '/Dev1/PFI3'
        self.task_counter_ctr2.triggers.pause_trigger.trig_type = nidaqmx.constants.TriggerType.DIGITAL_LEVEL
        self.task_counter_ctr2.triggers.pause_trigger.dig_lvl_when = nidaqmx.constants.Level.LOW



        self.task_counter_clock = nidaqmx.Task()
        self.task_counter_clock.co_channels.add_co_pulse_chan_freq(counter="Dev1/ctr3", freq=1e3, duty_cycle=0.5)
        # ctr2 clock for buffered edge counting ctr1
        self.task_counter_clock.timing.cfg_implicit_timing(sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS)
        self.task_counter_clock.start()

        self.exposure = None
        self.set_timing(exposure)
        
        self.task.start()
        self.task_counter_ai.start()
        self.task_counter_ctr.start()
        self.task_counter_ctr2.start()
        self.reader = self.nidaqmx.stream_readers.AnalogMultiChannelReader(self.task_counter_ai.in_stream)
        self.data_buffer = None
        # data_buffer for faster read
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


    def set_timing(self, exposure, clock=500e3):
        self.clock = clock
        self.sample_num = int(round(self.clock*exposure))
        self.task_counter_ai.stop()
        self.task_counter_ctr.stop()
        self.task_counter_ctr2.stop()
        self.exposure = exposure
        self.task_counter_ai.timing.cfg_samp_clk_timing(clock, sample_mode=self.nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=self.sample_num)
        self.task_counter_ctr.timing.cfg_samp_clk_timing(1e3, source = '/Dev1/Ctr3InternalOutput', \
            sample_mode=self.nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=int(np.ceil(1e3*exposure))+1)
        self.task_counter_ctr2.timing.cfg_samp_clk_timing(1e3, source = '/Dev1/Ctr3InternalOutput', \
            sample_mode=self.nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=int(np.ceil(1e3*exposure))+1)

        #self.task_counter_ctr.timing.cfg_samp_clk_timing(100e3, source = '/Dev1/Ctr2InternalOutput', \
        #    sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=int(round(100e3*exposure)))



    def read_counts(self, duration, parent, is_analog=False, is_dual=False, is_raw=False):

        self.task_counter_clock.stop()

        if duration != self.exposure:
            self.set_timing(duration)

        if not is_analog:

            if is_dual:
                self.task_counter_ctr.stop()
                self.task_counter_ctr2.stop()
                self.task_counter_ctr.start()
                self.task_counter_ctr2.start()
                self.task_counter_clock.start()
                time.sleep(duration)
                counts1_array = self.task_counter_ctr.read(number_of_samples_per_channel = int(np.ceil(1e3*duration))+1)
                counts2_array = self.task_counter_ctr2.read(number_of_samples_per_channel = int(np.ceil(1e3*duration))+1)
                counts1 = counts1_array[-1] - counts1_array[0]
                counts2 = counts2_array[-1] - counts2_array[0]

                if (counts1 == 0) or (counts2 == 0):
                    counts = 0
                else:
                    counts = counts1/counts2
                return counts
                
            else:

                self.task_counter_ctr.stop()
                self.task_counter_ctr.start()
                self.task_counter_clock.start()
                time.sleep(duration)
                counts1_array = self.task_counter_ctr.read(number_of_samples_per_channel = int(np.ceil(1e3*duration))+1)
                counts1 = counts1_array[-1] - counts1_array[0]
                counts = counts1
                return counts



        self.task_counter_ai.stop()

        if self.data_buffer is None:
            self.data_buffer = np.zeros((3, self.sample_num), dtype=np.float64)

        if duration != self.exposure:
            self.set_timing(duration)

            self.data_buffer = np.zeros((3, self.sample_num), dtype=np.float64)

        #reader = self.nidaqmx.stream_readers.AnalogMultiChannelReader(self.task_counter_ai.in_stream)

        self.task_counter_ai.start()
        time.sleep(duration)
        self.reader.read_many_sample(self.data_buffer, number_of_samples_per_channel = self.sample_num)
        self.task_counter_ai.stop()


        if is_raw:

            return self.data_buffer

        else:

            if is_dual:

                data = self.data_buffer[0, :]
                gate1 = self.data_buffer[1, :]
                gate2 = self.data_buffer[2, :]

                threshold = 2.7

                data_gate1 = np.mean(data[gate1 > threshold])
                data_gate2 = np.mean(data[gate2 > threshold])

                # seems better than np.sum()/np.sum(), don't know why?
                # may due to finite sampling rate than they have different array length

                if (data_gate1 == 0) or (data_gate2 == 0):
                    data_counts = 0
                else:
                    data_counts = data_gate1/data_gate2

            else:

                data = self.data_buffer[0, :]
                gate1 = self.data_buffer[1, :]

                threshold = 2.7

                data_gate1 = np.mean(data[gate1 > threshold])


                if (data_gate1 == 0):
                    data_counts = 0
                else:
                    data_counts = data_gate1


            return float(data_counts) # ratio between



    
    @classmethod
    def exit_handler(cls):

        if cls._instance is not None:
            cls._instance.task.stop()
            cls._instance.task.close()

            cls._instance.task_counter_ai.stop()
            cls._instance.task_counter_ai.close()


            cls._instance = None


class USB6346_extra(metaclass=SingletonMeta):
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
        self.task_counter_ai.ai_channels.add_ai_voltage_chan("Dev1/ai1")
        self.task_counter_ai.ai_channels.add_ai_voltage_chan("Dev1/ai2")


        self.task_counter_ctr = nidaqmx.Task()
        self.task_counter_ctr.ci_channels.add_ci_count_edges_chan("Dev1/ctr1")
        self.task_counter_ctr_ref = nidaqmx.Task()
        self.task_counter_ctr_ref.ci_channels.add_ci_count_edges_chan("Dev1/ctr2")
        #self.task_counter_ai_ref = nidaqmx.Task()
        #self.task_counter_ai_ref.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        # ref counter for eliminating experimental fluctuation longer than ~100us

        self.exposure = None
        self.set_timing(exposure)

        #self.task_counter_ai.triggers.pause_trigger.dig_lvl_src = '/Dev1/PFI0'
        #self.task_counter_ai.triggers.pause_trigger.trig_type = nidaqmx.constants.TriggerType.DIGITAL_LEVEL
        #self.task_counter_ai.triggers.pause_trigger.dig_lvl_when = nidaqmx.constants.Level.LOW
        #self.task_counter_ai.in_stream.read_all_avail_samp = True


        self.task_counter_ctr.triggers.pause_trigger.dig_lvl_src = '/Dev1/PFI4'
        self.task_counter_ctr.ci_channels.all.ci_count_edges_term = '/Dev1/PFI3'
        self.task_counter_ctr.triggers.pause_trigger.trig_type = nidaqmx.constants.TriggerType.DIGITAL_LEVEL
        self.task_counter_ctr.triggers.pause_trigger.dig_lvl_when = nidaqmx.constants.Level.LOW

        self.task_counter_ctr_ref.triggers.pause_trigger.dig_lvl_src = '/Dev1/PFI5'
        self.task_counter_ctr_ref.ci_channels.all.ci_count_edges_term = '/Dev1/PFI3'
        self.task_counter_ctr_ref.triggers.pause_trigger.trig_type = nidaqmx.constants.TriggerType.DIGITAL_LEVEL
        self.task_counter_ctr_ref.triggers.pause_trigger.dig_lvl_when = nidaqmx.constants.Level.LOW
        
        self.task.start()
        self.task_counter_ai.start()
        self.task_counter_ctr.start()
        self.task_counter_ctr_ref.start()
        self.reader = self.nidaqmx.stream_readers.AnalogMultiChannelReader(self.task_counter_ai.in_stream)
        self.data_buffer = None
        # data_buffer for faster read
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


    def set_timing(self, exposure, clock=500e3):
        self.clock = clock
        self.sample_num = int(round(self.clock*exposure))
        if exposure == self.exposure:
            return
        self.task_counter_ai.stop()
        #self.task_counter_ctr.stop()
        self.exposure = exposure
        self.task_counter_ai.timing.cfg_samp_clk_timing(clock, sample_mode=self.nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=self.sample_num)
        #self.task_counter_ctr.timing.cfg_samp_clk_timing(source='/Dev1/20MHzTimebase', \
        #    rate=clock, sample_mode=self.nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=self.sample_num)

    def read_counts_inst(self, duration, parent):
        pass


    def read_counts(self, duration, parent, is_analog=False, is_dual=False, is_raw=False):
        if not is_analog:
            return 0

        self.task_counter_ai.stop()

        if self.data_buffer is None:
            self.data_buffer = np.zeros((3, self.sample_num), dtype=np.float64)

        if duration != self.exposure:
            self.set_timing(duration)
            self.data_buffer = np.zeros((3, self.sample_num), dtype=np.float64)

        #reader = self.nidaqmx.stream_readers.AnalogMultiChannelReader(self.task_counter_ai.in_stream)

        self.task_counter_ai.start()
        time.sleep(duration)
        self.reader.read_many_sample(self.data_buffer, number_of_samples_per_channel = self.sample_num)
        self.task_counter_ai.stop()


        if is_raw:

            return self.data_buffer

        else:

            data = self.data_buffer[0, :]
            gate1 = self.data_buffer[1, :]
            gate2 = self.data_buffer[2, :]

            threshold = 2.7

            data_gate1 = np.mean(data[gate1 > threshold])
            data_gate2 = np.mean(data[gate2 > threshold])

            # seems better than np.sum()/np.sum(), don't know why?
            # may due to finite sampling rate than they have different array length

            if (data_gate1 == 0) or (data_gate2 == 0):
                data_counts = 0
            else:
                data_counts = data_gate1/data_gate2


            return float(data_counts) # ratio between


    
    def read_counts_extra(self, duration, parent, is_analog=False, is_dual=False):
        # not used

        if is_analog:
            self.task_counter_ai.stop()
        else:
            self.task_counter_ctr.stop()
            self.task_counter_ctr_ref.stop()

        self.set_timing(duration)

        if is_analog:
            self.task_counter_ai.start()
        else:
            self.task_counter_ctr_ref.start()
            self.task_counter_ctr.start()
            #self.task_counter_ctr_ref.start()

        time.sleep(duration)
        t0 = time.time()
        try:


            if is_analog:

                data_counts = self.task_counter_ai.read(self.sample_num)
                """
                while 1:
                    # tempory solution for read() returns empty array before all sample collected 
                    if self.task_counter_ai.in_stream.avail_samp_per_chan>0:
                        #print(time.time() - t0)
                        break
                    else:
                        if (time.time() - t0)>0.01:
                            break

                data_array = self.task_counter_ai.read(self.nidaqmx.constants.READ_ALL_AVAILABLE)


                if len(data_array) == 0:
                    data_counts = 0
                else:
                    #data_counts = float(np.sum(data_array)/self.sample_num)
                    data_counts = float(np.mean(data_array))
                """
            else:

                if is_dual:
                    data_counts_ref = self.task_counter_ctr_ref.read()
                    data_counts_ = self.task_counter_ctr.read()
                    #data_counts_ref = self.task_counter_ctr_ref.read()

                    if data_counts_ref == 0:
                        data_counts = 0
                    else:
                        data_counts = data_counts_/data_counts_ref

                else:

                    data_counts = self.task_counter_ctr.read()


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

            cls._instance.task_counter_ctr.stop()
            cls._instance.task_counter_ctr.close()

            cls._instance.task_counter_ctr_ref.stop()
            cls._instance.task_counter_ctr_ref.close()

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
        import pyvisa
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
        import pyvisa
        rm = pyvisa.ResourceManager()
        #self.handle = rm.open_resource('USB0::0x1AB1::0x099C::DSG8M267M00006::INSTR')
        self.handle = rm.open_resource('USB0::0x1AB1::0x099C::DSG8M223900103::INSTR')
        self._power = eval(self.handle.query('SOURce:Power?')[:-1])
        self._frequency = eval(self.handle.query('SOURce:FREQuency?')[:-1])
        self._iq = False # if IQ modulation is on
        self._on = False # if output is on
        self.power_ul = 10
        
    @property
    def power(self):
        self._power = eval(self.handle.query('SOURce:Power?')[:-1])
        return self._power
    
    @power.setter
    def power(self, power_in):
        if power_in > self.power_ul:
            power_in = self.power_ul
            print(f'can not exceed RF power {self.power_ul}dbm')
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
        import pyvisa   
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

    fixed 8ns limits of pulse streamer



    """

    def __init__(self, ip=None):
        from pulsestreamer import PulseStreamer, Sequence 

        self.PulseStreamer = PulseStreamer
        self.Sequence = Sequence
        if ip is None:
            self.ip = '169.254.8.2'
            # default ip address of pulse streamer
        else:
            self.ip = ip
        self.ps = PulseStreamer(self.ip)

        self.delay_array = np.array([0,]*8)

        self.data_matrix = np.array([[1e3, 1,1,1,1,1,1,1,1], [1e3, 1,1,1,1,1,1,1,1]])
        # example of data_matrix [[duration in ns, on or off for channel i, ...], ...]
        self.total_duration = 0


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

    def set_timing_simple(self, timing_matrix):
        # set_timing_simple([[duration0, [channels0, ]], [], [], []])
        # eg. 
        # set_timing_simple([[100, (3,)], [100, (3,5)]])
        # channel0 - channel7
        n_sequence = len(timing_matrix)
        if n_sequence <= 2:
            print('pulse length must larger than 2')
            return 
        data_matrix = [[0]*9 for _ in range(n_sequence)]

        for i in range(n_sequence):
            data_matrix[i][0] = int(timing_matrix[i][0]) # avoid possible float number ns duration input

            for channel in timing_matrix[i][1]:
                data_matrix[i][channel+1] = 1


        self.data_matrix = data_matrix

    def set_timing(self, data_matrix):
        # data_matrix is n_sequence*(1+8) matrix, one for during, eight for channels
        # duraing is in ns
        # channels is 1 for on, 0 for off
        # example of data_matrix [[duration in ns, on or off for channel i, ...], ...]
        # [[1e3, 1, 1, 1, 1, 1, 1, 1, 1], ...]
        n_sequence = len(data_matrix)
        if n_sequence <= 2:
            print('pulse length must larger than 2')
            return 
        self.data_matrix = data_matrix

    def set_delay(self, delay_array):
        self.delay_array = [int(delay) for delay in delay_array]



                    
    def read_data(self):

        # return delayed time_slices [[[t0, 1], [t1, 0], ...], [], [],...] for all channels
        
        data_matrix = self.data_matrix
        # data_matrix is [[1e3, 1, 1, 1, 1, 1, 1, 1, 1], ...]

        time_slices = []
        for channel in range(8):
            time_slice = [[period[0], period[channel+1]] for period in data_matrix]
            time_slice_delayed = self.delay(self.delay_array[channel], time_slice)
            time_slices.append(time_slice_delayed)
        
        return time_slices
    
    def delay(self, delay, time_slice):
        # accept time slice
        # example of time slice [[duration in ns, on or off], ...]
        # [[1e3, 1], [1e3, 0], ...] 
        # add delay to time slice (mod by total duration)

        total_duration = 0
        for period in time_slice:
            total_duration += period[0]

        self.total_duration = total_duration

        delay = delay%total_duration

        if delay == 0:
            return time_slice


        # below assumes delay > 0
        cur_time = 0
        for ii, period in enumerate(time_slice[::-1]):
            # count from end of pulse for delay > 0
            cur_time += period[0]
            if delay == cur_time:
                return time_slice[-(ii+1):] + time_slice[:-(ii+1)]
                # cycle roll the time slice to right (ii+1) elements
            if delay < cur_time:
                duration_lhs = cur_time - delay
                # duration left on the left hand side of pulse
                duration_rhs = period[0] - duration_lhs

                time_slice_lhs = time_slice[:-(ii+1)] + [[duration_lhs, period[1]], ]
                time_slice_rhs = [[duration_rhs, period[1]], ] + time_slice[-(ii+1):][1:] # skip the old [t_ii, enable_ii] period
                return time_slice_rhs + time_slice_lhs

            # else will be delay > cur_time and should continue 




    




