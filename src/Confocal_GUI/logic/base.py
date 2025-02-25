import numpy as np
import sys
import time
import threading
from decimal import Decimal
from threading import Event
from abc import ABC, abstractmethod
from Confocal_GUI.live_plot import *
from Confocal_GUI.gui import *


class BaseMeasurement(ABC):
    """
    Base class defines measurements

    base procedure is

    >>> measurement = Mesurement()
    # choose measurement class
    >>> mesurement.init_assign(params)
    # which will reset initial state 
    >>> mesurement.start()
    # data_generator starts
    >>> mesurement.stop()

    #or 

    >>> ple(params)
    # equivalent to above plus choose live_plot class

    #or

    >>> ple(params, is_GUI)
    # to GUI, now params is optional

    GUI uses measurement.device_to_state() methods etc.
    """

    def __init__(self, config_instances):
        super().__init__()
        self.config_instances = config_instances
        self.assign_names()
        self.is_running = True
        # enable using all method even before load_params()

    def _iterate_data_x(self):
        """
        Generate indices and corresponding x values for iteration.
        Supports both 1D and 2D data_x.
        """
        for i, x in enumerate(self.data_x):
            yield i, x

    def _data_generator(self):
        # defines core of data_generator, how .start() call will lead to
        self.to_initial_state()

        for self.repeat_done in range(self.repeat):
            for indices, x in self._iterate_data_x():
                if not self.is_running:
                    self.is_done = True
                    self.to_final_state()
                    return
                self.device_to_state(x)
                self.update_data_y(indices)
                self.points_done += 1

        self.is_done = True
        self.to_final_state()

    def start(self):
        if not self.loaded_params:
            print('missing params, use measurement.load_params()')
            return

        self.is_running = True
        self.is_done = False
        self.points_done = 0
        self.repeat_done = 1
        # reset data_generator

        # how data_generator is called by live_plot
        self.thread = threading.Thread(target=self._data_generator, daemon=True)
        self.thread.start()

    def load_params(self, **kwargs):
        self._load_params(**kwargs)

        len_counts = len(self.counter.read_counts(0.01, parent = self, counter_mode=self.counter_mode, data_mode=self.data_mode))
        valid_update_mode = ['normal', 'roll', 'new', 'adaptive', 'single']
        if self.update_mode not in valid_update_mode:
            print(f'update_mode must be one of {valid_update_mode}')
            return

        if self.update_mode=='new':
            self.data_y = np.full((len(self.data_x), self.repeat), np.nan)
        elif self.update_mode=='single':
            self.data_y = np.full((len(self.data_x), 1), np.nan)
        else:
            self.data_y = np.full((len(self.data_x), len_counts), np.nan)


    def update_data_y(self, i):
        # defines how to update data_y
        # normally will be
        # counts = self.counter(self.exposure, parent=self) 
        # data_y[i] += counts
        counts = self.counter.read_counts(self.exposure, parent = self, counter_mode=self.counter_mode, data_mode=self.data_mode)

        if self.update_mode == 'normal':
            self.data_y[i] = counts if np.isnan(self.data_y[i][0]) else [(self.data_y[i][j] + counts[j]) for j in range(len(counts))]

        elif self.update_mode == 'single':
            self.data_y[i] = counts[:1] if np.isnan(self.data_y[i][0]) else [(self.data_y[i][j] + counts[j]) for j in range(len(counts[:1]))]

        elif self.update_mode == 'roll':
            self.data_y[:] = np.roll(self.data_y, shift=1, axis=0)
            self.data_y[0] = counts

        elif self.update_mode == 'new':
            self.data_y[i, self.repeat_done] = counts[0]

        elif self.update_mode == 'adaptive':
            self.threshold = 2*np.sqrt(1000)
            exposure = self.exposure
            counts = counts[0]
            while counts/(np.sqrt(exposure)*self.threshold) > 1:
                counts += self.counter.read_counts(self.exposure, parent = self, counter_mode=self.counter_mode, data_mode=self.data_mode)[0]
                exposure += self.exposure
                if exposure/self.exposure >= 10:
                    break
            self.data_y[i] = [counts/(np.sqrt(exposure)*self.threshold),]


    def stop(self):
        if self.thread.is_alive():
            self.is_running = False
            self.thread.join()
        self.loaded_params = False

    @abstractmethod
    def _load_params(self, **kwargs):
        pass


    @abstractmethod
    def device_to_state(self, x):
        # move device state to x from data_x
        # also need to include such as rf.on = True etc.
        # to make sure device_to_state(x) call will bring device_x to desired state x
        # but power can be set from set device
        pass

    @abstractmethod
    def to_initial_state(self):
        # move device/data state to initial state before measurement
        pass

    @abstractmethod
    def to_final_state(self):
        # move device/data state to final state after measurement
        pass

    @abstractmethod
    def read_x(self):
        # read_x for GUI uses
        pass

    @abstractmethod
    def assign_names(self):
        # defines measurement names etc.
        # init assignment of config instances
        pass

    @abstractmethod
    def plot(self):
        # defines how to call in jupyter
        pass




measurement_registry = {}

def run_measurement(name, **kwargs):
    """
    connect run_measurement(name) func to measurement_registry['name'] func
    """
    cls = measurement_registry.get(name)
    if cls is None:
        raise ValueError(f"No measurement registered with name '{name}'")
    measurement = cls({**kwargs}['config_instances'])
    return measurement.plot(**kwargs)

def register_measurement(name: str):
    """
    register 'name' func to package and allow a faster call
    e.g. from confocal_gui.logic import ple
    e.g. ple(**kwargs) is same as calling PLEMeasurement(**kwargs).plot(**kwargs)
    """
    def decorator(cls):
        measurement_registry[name] = cls

        
        def measure_func(**kwargs):
            return run_measurement(name, **kwargs)

        if hasattr(cls, '_load_params'):
            measure_func.__doc__ = cls._load_params.__doc__

        measure_func.__name__ = name


        parent_module = sys.modules[__name__] 
        #print(parent_module, 'parent')
        setattr(parent_module, name, measure_func)


        if '__all__' in parent_module.__dict__:
            if name not in parent_module.__all__:
                parent_module.__all__.append(name)
        else:
            parent_module.__all__ = [name]


        return cls
    return decorator


@register_measurement('live') #allow a faster call to .load_params() using ple()
class LiveMeasurement(BaseMeasurement):

    def device_to_state(self, frequency):
        # move device state to x from data_x, defaul frequency in GHz
        pass


    def to_initial_state(self):
        # move device/data state to initial state before measurement
        pass


    def to_final_state(self):
        # move device/data state to final state after measurement
        pass

    def read_x(self):
        pass

    def assign_names(self):

        self.x_name = 'Data'
        self.x_unit = '1'
        self.measurement_name = 'Live'
        self.x_device_name = ''
        self.plot_type = '1D'
        self.is_change_unit = False
        self.loaded_params = False
        self.counter = self.config_instances.get('counter', None)
        self.scanner = self.config_instances.get('scanner', None)
        # init assignment
        if (self.counter is None):
            raise KeyError('Missing devices in config_instances')


    def _load_params(self, data_x=None, exposure=0.1, config_instances=None, is_finite=False, is_GUI=False, repeat=1, \
        counter_mode='apd', data_mode='single', relim_mode='normal', is_plot=True):
        """
        live

        args:
        (data_x=None, exposure=0.1, config_instances=None, is_finite=False, repeat=1, 
        counter_mode='apd', data_mode='single', relim_mode='normal'):

        example:
        fig, data_figure = live(data_x = np.arange(100), exposure=0.1, 
                                config_instances=config_instances, repeat=1, is_finite=False,
                                counter_mode='apd', data_mode='single', relim_mode='normal')

        """
        self.loaded_params = True
        if is_finite==False:
            self.repeat = int(1e6)
        else:
            self.repeat = int(repeat)
        # large enough and in practical infinite
        if data_x is None:
            data_x = np.arange(100)
        self.data_x = data_x
        self.update_mode = 'roll'
        self.exposure = exposure
        self.is_GUI = is_GUI
        self.counter_mode = counter_mode
        self.data_mode = data_mode
        self.relim_mode = relim_mode
        self.is_plot = is_plot
        self.info = {'measurement_name':self.measurement_name, 'plot_type':self.plot_type, 'exposure':self.exposure\
                    , 'repeat':self.repeat, 'scanner':(None if self.scanner is None else (self.scanner.x, self.scanner.y))}

    def plot(self, **kwargs):
        self.load_params(**kwargs)
        if not self.is_plot:
            return self

        if self.is_GUI:
            GUI_Live(config_instances = self.config_instances, measurement_Live=self)
            return None, None
        else:
            data_x = self.data_x
            data_y = self.data_y
            data_generator = self
            liveplot = LiveAndDisLive(labels=[f'{self.x_name} ({self.x_unit})', f'Counts/{self.exposure}s'], \
                                update_time=0.02, data_generator=data_generator, data=[data_x, data_y], \
                                config_instances = self.config_instances, relim_mode=self.relim_mode)

            fig, selector = liveplot.plot()
            data_figure = DataFigure(liveplot)
            return fig, data_figure

