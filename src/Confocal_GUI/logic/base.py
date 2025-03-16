import numpy as np
import sys
import time
import threading
from decimal import Decimal
from threading import Event
from abc import ABC, abstractmethod
from Confocal_GUI.live_plot import *
from Confocal_GUI.gui import *
from Confocal_GUI.device import config_instances

class BaseMeasurement(ABC):
    """
    Base class defines measurements

    base procedure is

    >>> measurement = Mesurement() or shortcall(is_plot=False)
    # choose measurement class
    >>> mesurement.load_params(**kwargs) or shortcall(**kwargs)
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

    def __init__(self):
        super().__init__()
        self.config_instances = config_instances
        self.load_config()
        self.info = {}
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
                if indices==0 and self.repeat_done==0:
                    time.sleep(1)
                # wait for stabilization for the first data point, can be removed
                self.update_data_y(indices)
                self.points_done += 1

        self.is_done = True
        self.to_final_state()
    def start(self):
        if not self.loaded_params:
            print('missing params, use measurement.load_params()')
            return

        if (self.pulse_file is not None) and (self.pulse is not None):
            self.pulse.off_pulse()
            self.pulse.load_from_file(self.pulse_file)
            self.pulse.on_pulse()
            print(f'Loading pulse file {self.pulse_file}')
        else:
            print('No pulse file or pulse is assigned, using exsited pulse')

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

        len_counts = self.counter.check_data_len(data_mode=self.data_mode)
        if self.update_mode=='new':
            self.data_y = np.full((len(self.data_x), self.repeat), np.nan)
        elif self.update_mode=='single':
            self.data_y = np.full((len(self.data_x), 1), np.nan)
        else:
            self.data_y = np.full((len(self.data_x), len_counts), np.nan)
        if not hasattr(self, 'pulse_file'):
            self.pulse_file = None
        self.pulse = self.config_instances.get('pulse', None)
        self.loaded_params = True


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

        else:
            valid_update_mode = ['normal', 'roll', 'new', 'single']
            print(f'update_mode must be one of {valid_update_mode}')
            # single for PL which only display one set of data, normal for PLE which enables show more sets depends on counter return
            # roll for live which shift data
            # new for repeat to add newline instead of adding to exsiting line
            # adaptive for adaptive exposure



    def stop(self):
        if self.thread.is_alive():
            self.is_running = False
            self.thread.join()
        self.loaded_params = False

    def plot(self, **kwargs):
        self.is_plot = kwargs.get('is_plot', True)
        self.load_params(**kwargs)
        if not self.is_plot:
            return self
        else:
            fig, data_figure = self.load_GUI()
            return fig, data_figure


    @abstractmethod
    def load_GUI(self):
        # defines how to connect to GUI
        # e.g.
        #if self.is_GUI:
        #    GUI_Live(measurement_Live=self)
        #    return None, None
        #else:
        #    data_x = self.data_x
        #    data_y = self.data_y
        #    data_generator = self
        #    liveplot = LiveAndDisLive(labels=[f'{self.x_name} ({self.x_unit})', f'Counts/{self.exposure}s'],
        #                        update_time=0.02, data_generator=data_generator, data=[data_x, data_y],
        #                        relim_mode=self.relim_mode)
        #
        #    fig, selector = liveplot.plot()
        #    data_figure = DataFigure(live_plot=liveplot)
        #    return fig, data_figure        
        pass

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
    def load_config(self):
        # defines measurement names etc.
        # init assignment of config instances
        pass




measurement_registry = {}

def run_measurement(name, **kwargs):
    """
    connect run_measurement(name) func to measurement_registry['name'] func
    """
    cls = measurement_registry.get(name)
    if cls is None:
        raise ValueError(f"No measurement registered with name '{name}'")
    measurement = cls()
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

        parent_module = sys.modules[__name__]
        setattr(parent_module, name, measure_func)

        if '__all__' in parent_module.__dict__:
            if name not in parent_module.__all__:
                parent_module.__all__.append(name)
        else:
            parent_module.__all__ = [name]

        return cls
    return decorator


