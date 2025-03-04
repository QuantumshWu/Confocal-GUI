import numpy as np
import time
from numbers import Number
import os
from abc import ABC, abstractmethod, ABCMeta
import atexit
from Confocal_GUI.gui import *


def initialize_classes(config, lookup_dict, namespace):
    """

    Args:
        config (dict): a dict contains all devices and corresponding classes 
        lookup_dict (dict): uses globals() to get references of corresponding classes
        namespace (dict): the namespace to assign devices instances back 

    Returns:
        dict: config_instances used for pl, ple, live etc.

    Example:
        config = {
        'scanner': {'type': 'VirtualScanner'},    
        'counter': {'type': 'VirtualCounter'},    
        'wavemeter': {'type': 'VirtualWaveMeter'},    
        'rf': {'type': 'VirtualRF'},
        'pulse': {'type': 'VirtualPulse'},
        'laser_stabilizer': {'type': 'VirtualLaserStabilizer','config_instances':'config_instances'},
        }

        config_instances = initialize_classes(config, lookup_dict=globals(), namespace=globals())

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

    if namespace is not None:
        namespace.update(instances)

    for k, v in instances.items():
        print(f"{k} => {v}")
    print('\nNow you can call devices using e.g. config_instances["rf"].gui() or rf.gui()')
    return instances

def simple_hashable_value(v):
    # if dict, converted to a tuple
    if isinstance(v, dict):
        return tuple(sorted(v.items()))
    return v

class SingletonAndCloseMeta(ABCMeta):
    # make sure all devices only have one instance
    # mutiple initialization will get the same instance if params for initilization are not changed
    # and also register close() to atexit

    # Dictionary to store the instance and its initialization key for each class
    _instance_map = {}

    def __call__(cls, *args, **kwargs):
        # Convert args: if any arg is a dict, convert it to a sorted tuple of items
        hashable_args = tuple(simple_hashable_value(x) for x in args)
        # For kwargs, sort the items and convert dict values if needed
        hashable_kwargs = tuple(sorted((k, simple_hashable_value(v)) for k, v in kwargs.items()))
        # Use these to form a unique key
        device_key = (hashable_args, hashable_kwargs)
        map_key = (cls, device_key)
        
        if map_key in cls._instance_map:
            old_key, old_instance = cls._instance_map[map_key]
            if old_key == device_key:
                # If the initialization parameters match, return the existing instance
                return old_instance
            else:
                # If the parameters differ, close the existing instance
                old_instance.close()
        
        # Create a new instance and register its close() method for program exit
        instance = super().__call__(*args, **kwargs)
        atexit.register(instance.close)
        cls._instance_map[map_key] = (device_key, instance)
        return instance
      

class BaseLaser(ABC):

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
    

class BaseRF(ABC):

    @property
    @abstractmethod
    def frequency(self):
        pass
    
    @frequency.setter
    @abstractmethod
    def frequency(self, value):
        pass

    @property
    @abstractmethod
    def power(self):
        pass
    
    @power.setter
    @abstractmethod
    def power(self, value):
        pass

    @property
    @abstractmethod
    def on(self):
        pass
    
    @on.setter
    @abstractmethod
    def on(self, value):
        pass

class BaseWavemeter(ABC):

    @property
    @abstractmethod
    def wavelength(self):
        pass
    
class BaseCounter(ABC):

    @property
    @abstractmethod
    def valid_counter_mode(self):
        pass

    @property
    @abstractmethod
    def valid_data_mode(self):
        pass

    @abstractmethod
    def read_counts(self, exposure, counter_mode, data_mode):
        pass

class BaseCounterNI(BaseCounter):
    # Basecounter class for NI-DAQ board/card
    # defines how to setup counter tasks and analogin tasks

    def __init__(self, port_config):

        import nidaqmx
        from nidaqmx.constants import AcquisitionType
        from nidaqmx.constants import TerminalConfiguration
        from nidaqmx.stream_readers import AnalogMultiChannelReader
        import warnings 


        # e.g. port_config = ({'apd_signal':'/Dev2/PFI3', 'apd_gate':'/Dev2/PFI4', 'apd_gate_ref':'/Dev2/PFI1',
        # 'analog_signal':'/Dev2/ai0', 'analog_gate':'/Dev2/ai1', 'analog_gate_ref':'/Dev2/ai2', 'apd_clock':'/Dev2/PFI5'})
        self.port_config = port_config
        if self.port_config.get('apd_signal') is not None:
            self.dev_num = '/'+self.port_config.get('apd_signal').split('/')[-2]+'/'
        else:
            self.dev_num = '/'+self.port_config.get('analog_signal').split('/')[-2]+'/'
        # get '/Dev2/'
        self.nidaqmx = nidaqmx
        self.counter_mode = None
        # analog, apd
        self.data_mode = None
        # single, ref_div, ref_sub, dual
        self.exposure = None
        self.tasks_to_close = [] # tasks need to be closed after swicthing counter mode 
        self.__valid_counter_mode = ['apd', 'apd_pg', 'analog']
        # apd_pg uses pulse sequence as sampling clock to sync counting clock with pulse clock in order to get best stability and accuracy
        # can only use when pulse.total_duration is fixed otherwise need to reset exposure/timing
        # with apd_clock be the sampling clock defined by pulse
        self.__valid_data_mode = ['single', 'ref_div', 'ref_sub', 'dual']
        self.valid_counter_mode = self.__valid_counter_mode
        self.valid_data_mode = self.__valid_data_mode
        self.exposure_min = 0

    @property
    def valid_counter_mode(self):
        return self._valid_counter_mode

    @valid_counter_mode.setter
    def valid_counter_mode(self, value):
        if not all(mode in self.__valid_counter_mode for mode in value):
            print(f'Can only be subset of the {self.__valid_counter_mode}')
        else:
            self._valid_counter_mode = value

    @property
    def valid_data_mode(self):
        return self._valid_data_mode

    @valid_data_mode.setter
    def valid_data_mode(self, value):
        if not all(mode in self.__valid_data_mode for mode in value):
            print(f'Can only be subset of the {self.__valid_data_mode}')
        else:
            self._valid_data_mode = value


    def set_timing(self, exposure):
        if self.counter_mode == 'apd':
            self.clock = 1e4 
            # sampling rate for edge counting, defines when transfer counts from DAQ to PC, should be not too large to accomdate long exposure
            self.buffer_size = int(1e6)
            self.task_counter_clock = self.nidaqmx.Task()
            self.task_counter_clock.co_channels.add_co_pulse_chan_freq(counter=self.dev_num+'ctr2', freq=self.clock, duty_cycle=0.5)
            # ctr2 clock for buffered edge counting ctr0 and ctr1
            self.task_counter_clock.timing.cfg_implicit_timing(sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS)

            self.tasks_to_close += [self.task_counter_clock,]

            self.sample_num = int(round(self.clock*exposure))+1
            self.task_counter_ctr.timing.cfg_samp_clk_timing(self.clock, source = self.dev_num+'Ctr2InternalOutput',
                sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=self.buffer_size
            )
            self.task_counter_ctr_ref.timing.cfg_samp_clk_timing(self.clock, source = self.dev_num+'Ctr2InternalOutput',
                sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=self.buffer_size
            )

            self.exposure = exposure

            self.counts_main_array = np.zeros(self.sample_num, dtype=np.uint32)
            self.counts_ref_array = np.zeros(self.sample_num, dtype=np.uint32)
            self.reader_ctr = self.nidaqmx.stream_readers.CounterReader(self.task_counter_ctr.in_stream)
            self.reader_ctr_ref = self.nidaqmx.stream_readers.CounterReader(self.task_counter_ctr_ref.in_stream)

            self.task_counter_ctr.start()
            self.task_counter_ctr_ref.start()
            # start clock after counter tasks
            self.task_counter_clock.start()

        elif self.counter_mode == 'apd_pg':
            self.clock = 1/(self.parent.config_instances['pulse'].total_duration/1e9)
            # sampling rate for edge counting, defines when transfer counts from DAQ to PC, should be not too large to accomdate long exposure
            self.buffer_size = int(1e6)
            # need to estimate clock rate to register every_n_sample_event if buffer_size not enough

            self.sample_num = int(round(self.clock*exposure))+1
            self.task_counter_ctr.timing.cfg_samp_clk_timing(self.clock, source = self.port_config['apd_clock'],
                sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=self.buffer_size
            )
            self.task_counter_ctr_ref.timing.cfg_samp_clk_timing(self.clock, source = self.port_config['apd_clock'],
                sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=self.buffer_size
            )

            self.exposure = (self.sample_num-1)*self.parent.config_instances['pulse'].total_duration/1e9

            self.counts_main_array = np.zeros(self.sample_num, dtype=np.uint32)
            self.counts_ref_array = np.zeros(self.sample_num, dtype=np.uint32)
            self.reader_ctr = self.nidaqmx.stream_readers.CounterReader(self.task_counter_ctr.in_stream)
            self.reader_ctr_ref = self.nidaqmx.stream_readers.CounterReader(self.task_counter_ctr_ref.in_stream)

            self.task_counter_ctr.start()
            self.task_counter_ctr_ref.start()

        elif self.counter_mode == 'analog':
            self.clock = 500e3 # sampling rate for analog input, should be fast enough to capture gate signal for postprocessing
            self.buffer_size = int(1e6)
            self.sample_num = int(round(self.clock*exposure))
            self.task_counter_ai.timing.cfg_samp_clk_timing(self.clock, sample_mode=self.nidaqmx.constants.AcquisitionType.CONTINUOUS
                , samps_per_chan=self.buffer_size)
            self.exposure = exposure
            self.counts_array = np.zeros((3, self.sample_num), dtype=np.float64)
            self.reader_analog = self.nidaqmx.stream_readers.AnalogMultiChannelReader(self.task_counter_ai.in_stream)
            self.task_counter_ai.start()
        else:
            print(f'Can only be one of the {self.valid_counter_mode}')



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
            self.task_counter_ctr.ci_channels.add_ci_count_edges_chan(self.dev_num+'ctr0')
            self.task_counter_ctr.triggers.pause_trigger.dig_lvl_src = self.port_config['apd_gate']
            self.task_counter_ctr.ci_channels.all.ci_count_edges_term = self.port_config['apd_signal']
            self.task_counter_ctr.triggers.pause_trigger.trig_type = self.nidaqmx.constants.TriggerType.DIGITAL_LEVEL
            self.task_counter_ctr.triggers.pause_trigger.dig_lvl_when = self.nidaqmx.constants.Level.LOW

            self.task_counter_ctr_ref = self.nidaqmx.Task()
            self.task_counter_ctr_ref.ci_channels.add_ci_count_edges_chan(self.dev_num+'ctr1')
            self.task_counter_ctr_ref.triggers.pause_trigger.dig_lvl_src = self.port_config['apd_gate_ref']
            self.task_counter_ctr_ref.ci_channels.all.ci_count_edges_term = self.port_config['apd_signal']
            self.task_counter_ctr_ref.triggers.pause_trigger.trig_type = self.nidaqmx.constants.TriggerType.DIGITAL_LEVEL
            self.task_counter_ctr_ref.triggers.pause_trigger.dig_lvl_when = self.nidaqmx.constants.Level.LOW

            self.task_counter_ctr.in_stream.relative_to = self.nidaqmx.constants.ReadRelativeTo.FIRST_SAMPLE
            self.task_counter_ctr_ref.in_stream.relative_to = self.nidaqmx.constants.ReadRelativeTo.FIRST_SAMPLE
            # relative to beginning of buffer, change offset instead

            self.counter_mode = counter_mode
            self.tasks_to_close += [self.task_counter_ctr, self.task_counter_ctr_ref]

        elif counter_mode == 'apd_pg':
            self.close_old_tasks()

            self.task_counter_ctr = self.nidaqmx.Task()
            self.task_counter_ctr.ci_channels.add_ci_count_edges_chan(self.dev_num+'ctr0')
            self.task_counter_ctr.triggers.pause_trigger.dig_lvl_src = self.port_config['apd_gate']
            self.task_counter_ctr.ci_channels.all.ci_count_edges_term = self.port_config['apd_signal']
            self.task_counter_ctr.triggers.pause_trigger.trig_type = self.nidaqmx.constants.TriggerType.DIGITAL_LEVEL
            self.task_counter_ctr.triggers.pause_trigger.dig_lvl_when = self.nidaqmx.constants.Level.LOW

            self.task_counter_ctr_ref = self.nidaqmx.Task()
            self.task_counter_ctr_ref.ci_channels.add_ci_count_edges_chan(self.dev_num+'ctr1')
            self.task_counter_ctr_ref.triggers.pause_trigger.dig_lvl_src = self.port_config['apd_gate_ref']
            self.task_counter_ctr_ref.ci_channels.all.ci_count_edges_term = self.port_config['apd_signal']
            self.task_counter_ctr_ref.triggers.pause_trigger.trig_type = self.nidaqmx.constants.TriggerType.DIGITAL_LEVEL
            self.task_counter_ctr_ref.triggers.pause_trigger.dig_lvl_when = self.nidaqmx.constants.Level.LOW

            self.task_counter_ctr.in_stream.relative_to = self.nidaqmx.constants.ReadRelativeTo.FIRST_SAMPLE
            self.task_counter_ctr_ref.in_stream.relative_to = self.nidaqmx.constants.ReadRelativeTo.FIRST_SAMPLE
            # relative to beginning of buffer, change offset instead

            self.counter_mode = counter_mode
            self.tasks_to_close += [self.task_counter_ctr, self.task_counter_ctr_ref]

        elif counter_mode == 'analog':
            self.close_old_tasks()

            self.task_counter_ai = self.nidaqmx.Task()
            self.task_counter_ai.ai_channels.add_ai_voltage_chan(self.port_config['analog_signal'])
            self.task_counter_ai.ai_channels.add_ai_voltage_chan(self.port_config['analog_gate'])
            self.task_counter_ai.ai_channels.add_ai_voltage_chan(self.port_config['analog_gate_ref'])
            # for analog counter
            self.counter_mode = counter_mode
            self.tasks_to_close += [self.task_counter_ai,]

        else:
            print(f'can only be one of the {self.valid_counter_mode}')


    def read_counts(self, exposure, counter_mode = 'apd', data_mode='single',**kwargs):

        if exposure < self.exposure_min:
            exposure = self.exposure_min

        self.parent = kwargs.get('parent', None)
        if (self.parent.config_instances.get('pulse', None) is None) and (counter_mode == 'apd_pg'):
            counter_mode = 'apd'
        # if no pulse return to 'apd' mode

        self.data_mode = data_mode
        if (counter_mode != self.counter_mode) or (exposure != self.exposure):
            self.set_counter(counter_mode)
            self.set_timing(exposure)


        if self.counter_mode == 'apd':
            total_sample = self.task_counter_ctr.in_stream.total_samp_per_chan_acquired
            self.task_counter_ctr.in_stream.offset = total_sample
            self.task_counter_ctr_ref.in_stream.offset = total_sample
            # update read pos accrodingly to keep reading most recent self.sample_num samples

            sample_remain = self.sample_num
            while sample_remain>0:
                read_sample_num = np.min([self.buffer_size, sample_remain])
                self.reader_ctr.read_many_sample_uint32(self.counts_main_array
                    , number_of_samples_per_channel = read_sample_num, timeout=5*self.exposure
                )
                self.reader_ctr_ref.read_many_sample_uint32(self.counts_ref_array
                    , number_of_samples_per_channel = read_sample_num, timeout=5*self.exposure
                )

                data_main_0 = float(self.counts_main_array[0])
                data_ref_0 = float(self.counts_ref_array[0])
                sample_remain -= read_sample_num
            data_main = float(self.counts_main_array[-1] - data_main_0)
            data_ref = float(self.counts_ref_array[-1] - data_ref_0)

        elif self.counter_mode == 'apd_pg':
            total_sample = self.task_counter_ctr.in_stream.total_samp_per_chan_acquired
            self.task_counter_ctr.in_stream.offset = total_sample
            self.task_counter_ctr_ref.in_stream.offset = total_sample
            # update read pos accrodingly to keep reading most recent self.sample_num samples

            sample_remain = self.sample_num
            while sample_remain>0:
                read_sample_num = np.min([self.buffer_size, sample_remain])
                self.reader_ctr.read_many_sample_uint32(self.counts_main_array
                    , number_of_samples_per_channel = read_sample_num, timeout=5*self.exposure
                )
                self.reader_ctr_ref.read_many_sample_uint32(self.counts_ref_array
                    , number_of_samples_per_channel = read_sample_num, timeout=5*self.exposure
                )

                data_main_0 = float(self.counts_main_array[0])
                data_ref_0 = float(self.counts_ref_array[0])
                sample_remain -= read_sample_num
            data_main = float(self.counts_main_array[-1] - data_main_0)
            data_ref = float(self.counts_ref_array[-1] - data_ref_0)

        elif self.counter_mode == 'analog':
            total_sample = self.task_counter_ai.in_stream.total_samp_per_chan_acquired
            self.task_counter_ai.in_stream.offset = total_sample
            # update read pos accrodingly to keep reading most recent self.sample_num+1 samples

            sample_remain = self.sample_num
            data_main = 0
            data_ref = 0
            while sample_remain>0:
                read_sample_num = np.min([self.buffer_size, sample_remain])
                self.reader_analog.read_many_sample(self.data_buffer, 
                    number_of_samples_per_channel = read_sample_num, timeout=5*self.exposure
                )

                data = self.counts_array[0, :]
                gate1 = self.counts_array[1, :]
                gate2 = self.counts_array[2, :]
                threshold = 2.7

                gate1_index = np.where(gate1 > threshold)[0]
                gate2_index = np.where(gate2 > threshold)[0]

                data_main += float(np.sum(data[gate1_index]))
                data_ref += float(np.sum(data[gate2_index]))
                # seems better than np.sum()/np.sum(), don't know why?
                # may due to finite sampling rate than they have different array length
                sample_remain -= read_sample_num

            data_main = data_main/self.sample_num
            data_ref = data_ref/self.sample_num

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

class BaseScanner(ABC):

    @property
    @abstractmethod
    def x(self):
        pass
    
    @x.setter
    @abstractmethod
    def x(self, value):
        pass

    @property
    @abstractmethod
    def y(self):
        pass
    
    @y.setter
    @abstractmethod
    def y(self, value):
        pass


class BaseLaserStabilizer(ABC):
    """
    Base class to 
    
    """
    def __init__(self, config_instances):
        import threading
        self.is_ready = False
        self._wavelength = None # wavelength that user inputs
        self.desired_wavelength = None # used for feedback
        self.is_running = True
        self._on = False
        # indicate if wavelnegth has changed
        self.config_instances = config_instances
        self.wavemeter = self.config_instances.get('wavemeter', None)
        if self.wavemeter is None:
            raise KeyError('Missing devices in config_instances')
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        # will be killed if main thread is killed

    @property
    def wavelength(self):
        return self._wavelength
    
    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value

    @property
    def on(self):
        return self._on
    
    @on.setter
    def on(self, value):
        self._on = value
                
    
    def _run(self):
        while self.is_running:
            
            if self.desired_wavelength != self._wavelength:
                self.desired_wavelength = self._wavelength
                self.is_ready = False
                
            if self.desired_wavelength == None:
                time.sleep(0.01)
                # waiting for a valid wavelength input
                continue
            else:
                time.sleep(0.01) 

                if self._on:
                    self._stabilizer_core()


    @abstractmethod
    def _stabilizer_core(self):
        pass


    def close(self):
        if self.thread.is_alive(): #use () cause it's a method not property
            self.is_running = False
            self.join()


class BasePulse(ABC):
    """
    Base class for pulse control
    """
    def __init__(self, t_resolution=(1,1)):
        self.t_resolution = t_resolution 
        # minumum allowed pulse (width, resolution), (10, 2) for spin core, will round all time durations beased on this
        self._valid_str = ['+', '-', 'x', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        self.delay_array = np.array([0,0,0,0,0,0,0,0])
        self.delay_array_tmp = np.array([0,0,0,0,0,0,0,0])
        self.data_matrix = np.array([[1000, 1,0,0,0,0,0,0,0], [1000, 1,0,0,0,0,0,0,0]])
        # example of data_matrix [[duration in ns, on or off for channel i, ...], ...]
        # self._data_matrix can also be np.array([['x', 1,1,1,1,1,1,1,1], ['1000-x', 1,1,1,1,1,1,1,1]])
        self.data_matrix_tmp = np.array([[1000, 1,0,0,0,0,0,0,0], [1000, 1,0,0,0,0,0,0,0]])
        self.repeat_info = [0, -1, 1] # start_index, end_index(include), repeat_times
        self.repeat_info_tmp = [0, -1, 1]

        self._ref_info = {"is_ref":False, "signal":None, "DAQ":None, "DAQ_ref":None, 'clock':None}
        self.ref_info = {"is_ref":False, "signal":None, "DAQ":None, "DAQ_ref":None, 'clock':None} #
        self.ref_info_tmp = {"is_ref":False, "signal":None, "DAQ":None, "DAQ_ref":None, 'clock':None}

        # _data_matrix for on_pulse(), data_matrix_tmp for reopen gui display
        self.channel_names = ['', '', '', '', '', '', '', '']
        self.channel_names_tmp = ['', '', '', '', '', '', '', '']
        # names of all channels, such as 'RF', 'Green', 'DAQ', 'DAQ_ref'
        self.total_duration = 0
        self.x = 10 # postive int in ns, a timing varible used to replace all 'x' in timing array and matrix, effective only in read_data()

    def round_up(self, t, type='resolution'):
        # round up t into multiple of self.t_resolution or keep t if str
        valid_type = ['width', 'resolution']
        if type not in valid_type:
            print(f'type must be one of {valid_type}')
            return
        if type=='resolution':
            if isinstance(t, Number):
                if t%self.t_resolution[1]==0:
                    return int(t)
                else:
                    print(f'Due to resolution limit, rounded time resolution to {self.t_resolution[1]}')
                    return int((t//self.t_resolution[1] + 1)*self.t_resolution[1])
            elif isinstance(t, str):
                return t

        elif type=='width':
            if isinstance(t, Number):
                if t>=self.t_resolution[0]:
                    return int(t)
                else:
                    print(f'Due to resolution limit, rounded width to {self.t_resolution[0]}')
                    return int(self.t_resolution[0])
            else:
                print('Wrong input width, must be a number')


    @property
    def delay_array(self):
        return self._delay_array
    
    @delay_array.setter
    def delay_array(self, value):
        if len(value)!= 8:
            print('invalid delay array length')
            return
        if not all(isinstance(item, (Number, str)) for item in value):
            print('Invalid delay array content. Must only contain int numbers in ns or str contains x for time variable.')
            return
        if not all(isinstance(item, Number) or all(elem in self._valid_str for elem in item) for item in value):
            print(f"Invalid input. Can only be one of {self._valid_str}.")
            return
        self._delay_array = [self.round_up(delay) for delay in value]

    @property
    def data_matrix(self):
        return self._data_matrix
    
    @data_matrix.setter
    def data_matrix(self, value):
        if len(value)< 2:
            print('invalid data_matrix length')
            return
        if not all(
            len(item) == 9 and 
            (isinstance(item[0], (Number, str))) and 
            all(elem in (0, 1) for elem in item[1:])
            for item in value
        ):
            # must be length 9, item[0] must be int in ns or 'x' str, item in item[1:] must be 1 or 0
            print("Invalid input. Each item must meet the conditions.")
            return
        for period in value:
            if not isinstance(period[0], Number):
                if not all(letter in self._valid_str for letter in period[0]):
                    print(f"Invalid input. Can only be one of {self._valid_str}.") 
                    return

        self._data_matrix = [[self.round_up(item) if i==0 else item for i, item in enumerate(period)] for period in value]

    @property
    def repeat_info(self):
        return self._repeat_info
    
    @repeat_info.setter
    def repeat_info(self, value):
        if len(value)!= 3:
            print('invalid repeat_info length')
            return
        if not all(isinstance(item, (Number, str)) for item in value):
            print('Invalid repeat_info content.')
            return
        if not ((0<=value[0]<=(len(self.data_matrix)-2)) and ((value[1]-value[0])>=1 or (value[1]==-1))):
            print('Invalid repeat_info content.')
        self._repeat_info = [int(item) for item in value]
        # must all be integers

    @property
    def ref_info(self):
        return self._ref_info
    
    @ref_info.setter
    def ref_info(self, value):
        if not isinstance(value, dict):
            print('invalid ref_info, example {"is_ref":False, "signal":None, "DAQ":None, "DAQ_ref":None, "clock":None}')
            return
        if value.get('is_ref', None) not in [True, False]:
            print('Invalid ref_info["is_ref"].')
            return
        if not all(value.get(key, True) in [None, 0, 1, 2, 3, 4, 5, 6, 7] for key in ['signal', 'DAQ', 'DAQ_ref', 'clock']):
            print('Invalid ref_info["signal"] or ref_info["DAQ"] or ref_info["DAQ_ref"] or ref_info["clock"]')

        for key, channel in value.items():
            self._ref_info[key] = channel
        # must all be integers

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        if not isinstance(int(value), Number):
            print('x must be a number')
            return
        self._x = self.round_up(value)


    @abstractmethod    
    def off_pulse(self):
        # rewrite this method for real pulse
        pass

    @abstractmethod    
    def on_pulse(self):
        # rewrite this method for real pulse
        pass

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
            if isinstance(timing_matrix[i][0], str):
                data_matrix[i][0] = timing_matrix[i][0]
            else:
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
        self.delay_array = [delay if isinstance(delay, str) else int(delay) for delay in delay_array]

    def load_x_to_str(self, timing, time_type):
        if time_type == 'delay':
            if isinstance(timing, Number):
                return int(timing)
            if isinstance(timing, str):
                return eval(f'{timing}'.replace('x', str(self.x)))
        elif time_type == 'duration':
            # must larger than 0
            if isinstance(timing, Number):
                if int(timing) > 0:
                    return int(timing)
                else:
                    print('Duration must larger than 0ns')
                    return 1
            if isinstance(timing, str):
                if eval(f'{timing}'.replace('x', str(self.x)))>0:
                    return eval(f'{timing}'.replace('x', str(self.x)))
                else:
                    print('Duration must larger than 0ns')
                    return 1
                    
    def read_data(self, type='time_slices'):
        # type ='time_slices' or 'data_matrix'
        valid_type = ['time_slices', 'data_matrix']
        if type not in valid_type:
            print(f'type must be one of {valid_type}')
            return

        # return delayed time_slices [[[t0, 1], [t1, 0], ...], [], [],...] for all channels
        
        data_matrix = self.data_matrix
        # data_matrix is [[1e3, 1, 1, 1, 1, 1, 1, 1, 1], ...]
        start_index = self.repeat_info[0]
        end_index = len(self.data_matrix) if (self.repeat_info[1]==-1) else (self.repeat_info[1]+1)
        time_slices = []
        for channel in range(8):
            time_slice = []

            for period in data_matrix[:start_index]:
                time_slice.append([self.load_x_to_str(period[0], 'duration'), period[1:][channel]])
            # before repeat sequence
            for repeat in range(int(self.repeat_info[2])):
                for period in data_matrix[start_index:end_index]:
                    time_slice.append([self.load_x_to_str(period[0], 'duration'), period[1:][channel]])
            # repeat sequence
            for period in data_matrix[end_index:]:
                time_slice.append([self.load_x_to_str(period[0], 'duration'), period[1:][channel]])
            # after repeat sequence

            if self.ref_info['is_ref']:
                # repeat one more time with disabling 'signal' and replace 'DAQ' with 'DAQ_ref' channel
                def apply_ref(channel, period):
                    if channel==self.ref_info['signal']:
                        return 0
                    if channel==self.ref_info['DAQ']:
                        return 0
                    if channel==self.ref_info['clock']:
                        return 0
                    if channel==self.ref_info['DAQ_ref']:
                        return period[self.ref_info['DAQ']]
                    return period[channel]

                for period in data_matrix[:start_index]:
                    time_slice.append([self.load_x_to_str(period[0], 'duration'), apply_ref(channel, period[1:])])
                # before repeat sequence
                for repeat in range(int(self.repeat_info[2])):
                    for period in data_matrix[start_index:end_index]:
                        time_slice.append([self.load_x_to_str(period[0], 'duration'), apply_ref(channel, period[1:])])
                # repeat sequence
                for period in data_matrix[end_index:]:
                    time_slice.append([self.load_x_to_str(period[0], 'duration'), apply_ref(channel, period[1:])])
                # after repeat sequence


            time_slice_delayed = self.delay(self.load_x_to_str(self.delay_array[channel], 'delay'), time_slice)
            time_slices.append(time_slice_delayed)

        if type == 'time_slices':
        
            return [[[self.round_up(period[0], type='width'), period[1]] for period in time_slice] for time_slice in time_slices]

        elif type == 'data_matrix':
            # process, convert time_slices to data_matrix_delayed
            data_matrix_delayed = self._time_slices_to_data_matrix(time_slices)
            return [[self.round_up(period[i], type='width') if i==0 else period[i] for i in range(len(period))] for period in data_matrix_delayed]

    def _time_slices_to_data_matrix(self, time_slices):
        data_matrix = []
        while len(time_slices[0])!=0:
            t_cur = np.min([time_slice[0][0] for time_slice in time_slices])
            # find the minimum time of first period of all channels
            period_enable = [time_slice[0][1] for time_slice in time_slices]
            data_matrix.append([t_cur, ] + period_enable)
            for i in range(len(time_slices)):
                time_slices[i][0][0] -= t_cur
                if time_slices[i][0][0]==0:
                    time_slices[i] = time_slices[i][1:]

        return data_matrix


    def save_to_file(self, addr=''):

        current_time = time.localtime()
        current_date = time.strftime("%Y-%m-%d", current_time)
        current_time_formatted = time.strftime("%H:%M:%S", current_time)
        time_str = current_date.replace('-', '_') + '_' + current_time_formatted.replace(':', '_')

        if addr=='' or ('/' not in addr):
            pass
        else:
            directory = os.path.dirname(addr)
            if not os.path.exists(directory):
                os.makedirs(directory)

        
        np.savez(addr + '_pulse' + '.npz', data_matrix = np.array(self.data_matrix, dtype=object),
            delay_array = np.array(self.delay_array, dtype=object),
            channel_names = np.array(self.channel_names, dtype=object), 
            repeat_info = np.array(self.repeat_info, dtype=object),
            ref_info = np.array(self.ref_info, dtype=object)
        )

    def load_from_file(self, addr):
        import glob
        if '*' in addr:
            files_all = glob.glob(addr)# includes .jpg, .npz etc.
            files = []
            for file in files_all:
                if '_pulse.npz' in file:
                    files.append(file)
            if len(files) > 1:
                print(files)
            if len(files) == 0:
                print('no such pulse file')
                return
            address=files[0]
        else:
            address=addr

        loaded = np.load(address, allow_pickle=True)
        self.data_matrix = loaded['data_matrix']
        self.delay_array = loaded['delay_array']
        self.channel_names = loaded['channel_names']
        self.repeat_info = loaded['repeat_info']
        self.ref_info = loaded['ref_info'].item()

        self.data_matrix_tmp = loaded['data_matrix']
        self.delay_array_tmp = loaded['delay_array']
        self.channel_names_tmp = loaded['channel_names']
        self.repeat_info_tmp = loaded['repeat_info']
        self.ref_info_tmp = loaded['ref_info'].item()
        # also load to GUI

    
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



# ----------------------------------- above are Base classes, below are virtual classes for testing ---------------------------------------

class VirtualLaser(BaseLaser):
    """
    VirtualLaser class to simulate laser 
        
    """
      
    
    def __init__(self):
        
        pass

    @property
    def wavelength(self):
        pass
    
    @wavelength.setter
    def wavelength(self, value):
        pass

    @property
    def piezo(self):
        pass
    
    @piezo.setter
    def piezo(self, value):
        pass 



class VirtualRF(BaseRF):
    """
    VirtualRF class to simulate rf source,
    call rf.gui() to see all configurable parameters 

    """

    def __init__(self):
        self._frequency = 0
        self._power = 0
        self._on = False
        self.frequency_lb = 0.1e9 #GHz
        self.frequency_ub = 3.5e9 #GHz
        self.power_lb = -50 #dbm
        self.power_ub = 10 #dbm


    def gui(self):
        """
        Use self.gui_property and self.gui_property_type to determine how to display configurable parameters
        """
        self.gui_property = ['frequency', 'power', 'on']
        self.gui_property_type = ['float', 'float', 'str']
        GUI_Device(self)

    @property
    def frequency(self):
        return self._frequency
    
    @frequency.setter
    def frequency(self, value):
        if self.frequency_lb<=value<=self.frequency_ub:
            self._frequency = value
        else:
            print('out of range')

    @property
    def power(self):
        return self._power
    
    @power.setter
    def power(self, value):
        if self.power_lb<=value<=self.power_ub:
            self._power = value
        else:
            print('out of range')

    @property
    def on(self):
        return self._on
    
    @on.setter
    def on(self, value):
        #if on_in is self._on:
        #    return
        self._on = value
        
    
        
class VirtualWaveMeter(BaseWavemeter):
    """
    VirtualWaveMeter class to simulate wavemeter
    
    """
    

    def __init__(self):
        self._wavelength = 0


    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value


class VirtualCounter(BaseCounter):
    """
    VirtualCounter class to simulate counter,
    defines how counts are changing depends on rf.frequency, laser.wavelength, etc.
    
    """

    def __init__(self):
        pass

    @property
    def valid_counter_mode(self):
        return ['analog', 'apd']

    @property
    def valid_data_mode(self):
        return ['single', 'ref_div', 'ref_sub', 'dual']


    def read_counts(self, exposure, counter_mode='apd', data_mode='single', **kwargs):
        """
        simulated counter for test
        """

        self.data_mode = data_mode
        parent = kwargs.get('parent')
        if parent is None:
            time.sleep(exposure)
            return np.random.poisson(exposure*10000)
        
        _class = parent.__class__.__name__
        if _class == 'PLEMeasurement':
            
            ple_dict = {'ple_height':None, 'ple_width':None, 'ple_center':None, 'ple_bg':None}
            for key in ple_dict.keys():
                ple_dict[key] = parent.config_instances.get(key, 1)
            
            time.sleep(exposure)
            wavelength = parent.laser_stabilizer.desired_wavelength if parent.laser_stabilizer.desired_wavelength is not None else 0
            #print(wavelength, parent.laser_stabilizer)
            lambda_counts = exposure*(ple_dict['ple_height']*(ple_dict['ple_width']/2)**2
                                     /((wavelength-ple_dict['ple_center'])**2 + (ple_dict['ple_width']/2)**2) + ple_dict['ple_bg']
            )
            lambda_ref = exposure*(ple_dict['ple_bg'])
            if data_mode=='dual':
                return [np.random.poisson(lambda_counts),np.random.poisson(lambda_ref)]
            elif self.data_mode == 'ref_sub':
                return [np.random.poisson(lambda_counts)-np.random.poisson(lambda_ref),]
            elif self.data_mode == 'ref_div':
                ref = np.random.poisson(lambda_ref)
                return [np.random.poisson(lambda_counts)/ref if ref!=0 else 0,]
            elif self.data_mode == 'single':
                return [np.random.poisson(lambda_counts),]
        
        elif _class == 'PLMeasurement':
            
            pl_dict = {'pl_height':None, 'pl_width':None, 'pl_center':None, 'pl_bg':None}
            for key in pl_dict.keys():
                pl_dict[key] = parent.config_instances.get(key, 1)
            
            time.sleep(exposure)
            position = np.array([parent.scanner.x, parent.scanner.y])
            distance = np.linalg.norm(np.array(pl_dict['pl_center']) - position)
            
            lambda_counts = exposure*(pl_dict['pl_height']*(pl_dict['pl_width']/2)**2
                                     /((distance)**2 + (pl_dict['pl_width']/2)**2) + pl_dict['pl_bg']
            )
            return [np.random.poisson(lambda_counts),]

        elif _class == 'ODMRMeasurement' or _class == 'LiveMeasurement':
            odmr_dict = {'odmr_height':None, 'odmr_width':None, 'odmr_center':None}
            for key in odmr_dict.keys():
                odmr_dict[key] = parent.config_instances.get(key, 1)
            
            time.sleep(exposure)
            rf = parent.config_instances['rf']
            frequency = rf.frequency

            if counter_mode == 'analog':
                lambda_counts = exposure*odmr_dict['odmr_height']*(1-0.01*(odmr_dict['odmr_width']/2)**2
                                         /((frequency-odmr_dict['odmr_center'])**2 + (odmr_dict['odmr_width']/2)**2)
                )
                lambda_counts_ref = 0.995*exposure*odmr_dict['odmr_height']
                if self.data_mode == 'single':
                    return [np.random.normal(loc=0.1, scale=0.1/10000),]
                if self.data_mode == 'ref_div':
                    return [np.random.normal(loc=lambda_counts, scale=lambda_counts/10000)
                    /np.random.normal(loc=lambda_counts_ref, scale=lambda_counts_ref/10000),
                    ]
                if self.data_mode == 'ref_sub':
                    return [np.random.normal(loc=lambda_counts, scale=lambda_counts/10000)
                    -np.random.normal(loc=lambda_counts_ref, scale=lambda_counts_ref/10000),
                    ]
                if self.data_mode == 'dual':
                    return [np.random.normal(loc=lambda_counts, scale=lambda_counts/10000)
                    ,np.random.normal(loc=lambda_counts_ref, scale=lambda_counts_ref/10000)
                    ]


            if self.data_mode == 'single':
                lambda_counts = exposure*odmr_dict['odmr_height']*(1-(odmr_dict['odmr_width']/2)**2
                                         /((frequency-odmr_dict['odmr_center'])**2 + (odmr_dict['odmr_width']/2)**2)
                )

                if not rf.on:
                    return [np.random.poisson(exposure*odmr_dict['odmr_height']),]
                else:
                    return [np.random.poisson(lambda_counts),]
            elif self.data_mode == 'ref_sub':
                lambda_counts = exposure*odmr_dict['odmr_height']*(1-(odmr_dict['odmr_width']/2)**2
                                         /((frequency-odmr_dict['odmr_center'])**2 + (odmr_dict['odmr_width']/2)**2)
                )

                lambda_counts_ref = exposure*odmr_dict['odmr_height']
                if not rf.on:
                    return [np.random.poisson(exposure*odmr_dict['odmr_height']) - np.random.poisson(exposure*odmr_dict['odmr_height']),]
                else:
                    return [np.random.poisson(lambda_counts) - np.random.poisson(lambda_counts_ref),]
            elif self.data_mode == 'ref_div':
                lambda_counts = exposure*odmr_dict['odmr_height']*(1-(odmr_dict['odmr_width']/2)**2
                                         /((frequency-odmr_dict['odmr_center'])**2 + (odmr_dict['odmr_width']/2)**2)
                )

                lambda_counts_ref = exposure*odmr_dict['odmr_height']
                if not rf.on:
                    return [np.random.poisson(exposure*odmr_dict['odmr_height'])/np.random.poisson(exposure*odmr_dict['odmr_height']),]
                else:
                    return [np.random.poisson(lambda_counts)/np.random.poisson(lambda_counts_ref),]

            elif self.data_mode == 'dual':
                lambda_counts = exposure*odmr_dict['odmr_height']*(1-(odmr_dict['odmr_width']/2)**2
                                         /((frequency-odmr_dict['odmr_center'])**2 + (odmr_dict['odmr_width']/2)**2)
                )

                lambda_counts_ref = exposure*odmr_dict['odmr_height']
                if not rf.on:
                    return [np.random.poisson(lambda_counts_ref), np.random.poisson(lambda_counts_ref)]
                else:
                    return [np.random.poisson(lambda_counts), np.random.poisson(lambda_counts_ref)]

        elif _class == 'ModeSearchMeasurement':
            mode_dict = {'mode_height':None, 'mode_width':None, 'mode_center':None, 'mode_bg':None}
            for key in mode_dict.keys():
                mode_dict[key] = parent.config_instances.get(key, 1)
            
            time.sleep(exposure)
            frequency = parent.rf_1550.frequency
            lambda_counts = exposure*(mode_dict['mode_height']*(mode_dict['mode_width']/2)**2
                                     /((frequency-mode_dict['mode_center'])**2 + (mode_dict['mode_width']/2)**2) + mode_dict['mode_bg']
            )
            lambda_ref = exposure*(mode_dict['mode_bg'])
            if data_mode=='dual':
                return [np.random.poisson(lambda_counts),np.random.poisson(lambda_ref)]
            elif self.data_mode == 'ref_sub':
                return [np.random.poisson(lambda_counts)-np.random.poisson(lambda_ref),]
            elif self.data_mode == 'ref_div':
                ref = np.random.poisson(lambda_ref)
                return [np.random.poisson(lambda_counts)/ref if ref!=0 else 0,]
            elif self.data_mode == 'single':
                return [np.random.poisson(lambda_counts),]
            
        else: # None of these cases
            pl_dict = {'pl_height':None, 'pl_width':None, 'pl_center':None, 'pl_bg':None}
            for key in pl_dict.keys():
                pl_dict[key] = parent.config_instances.get(key, 1)
            
            time.sleep(exposure)
            position = np.array([parent.scanner.x, parent.scanner.y])
            distance = np.linalg.norm(np.array(pl_dict['pl_center']) - position)
            
            lambda_counts = exposure*(pl_dict['pl_height']*(pl_dict['pl_width']/2)**2
                                     /((distance)**2 + (pl_dict['pl_width']/2)**2) + pl_dict['pl_bg']
            )
            return [np.random.poisson(lambda_counts),]


class VirtualScanner(BaseScanner):
    """
    VirtualScanner class to scanner,
    call scanner.gui() to see all configurable parameters
    
    """
    def __init__(self):
        self.x = 0
        self.y = 0

        self.x_lb = -5000 #mV
        self.x_ub = 5000 #mV
        self.y_lb = -5000 #mV
        self.y_ub = 5000 #mV


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
        self._x = value
        
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        self._y = value
    



class VirtualLaserStabilizer(BaseLaserStabilizer, metaclass=SingletonAndCloseMeta):
    """
    VirtualLaserStabilizer class to simulate laserstabilizer which stabilize laser wavelength using feedback,
    call laserstabilizer.gui() to see all configurable parameters
    
    """
    def __init__(self, config_instances):
        super().__init__(config_instances)


    def gui(self):
        """
        Use self.gui_property and self.gui_property_type to determine how to display configurable parameters
        """
        self.gui_property = ['on', 'wavelength']
        self.gui_property_type = ['str', 'float']
        GUI_Device(self)
        

    def _stabilizer_core(self):
        # defines the core logic of feedback stabilization
        self.wavemeter.wavelength = self.desired_wavelength
        self.is_ready = True



class VirtualPulse(BasePulse):
    """
    VirtualPulse class to simulate pulse control (e.g. pulse streamer),
    call pulse.gui() to access to all functions and pulse sequence editing

    notes:
    pulse duration, delay can be a int in ns or str contains 'x' which will be later replaced by self.x,
    therefore enabling fast pulse control by self.x = x and self.on_pulse() without reconfiguring self.data_matrix
    and sel.delay_array

    Save Pulse: save pulse sequence to self.data_matrix, self.delay_array but not in file
    Save to file: save pulse sequence to a .npz file


    """

    def __init__(self):
        super().__init__()

    def gui(self, is_in_GUI=False):

        GUI_Pulse(self, is_in_GUI)

    gui.__doc__ = GUI_Pulse.__doc__


    def off_pulse(self):
        # rewrite this method for real pulse
        pass
        
    def on_pulse(self):
        # rewrite this method for real pulse
        
        self.off_pulse()
        
        
        time_slices = self.read_data(type='time_slices')
        for ii, time_slice in enumerate(time_slices):
            print(f'Ch{ii}', time_slice)
        return time_slices


        
