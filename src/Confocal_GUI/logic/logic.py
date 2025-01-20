import numpy as np
import sys
import time
import threading
from decimal import Decimal
from threading import Event
from scipy.optimize import curve_fit
from abc import ABC, abstractmethod
import numbers

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

    def _iterate_data_x(self):
        """
        Generate indices and corresponding x values for iteration.
        Supports both 1D and 2D data_x.
        """
        if isinstance(self.data_x[0], numbers.Number):
            for i, x in enumerate(self.data_x):
                yield i, x
        else:
            for j, y in enumerate(self.data_x[1]):
                for i, x in enumerate(self.data_x[0]):
                    yield (i, j), (x, y)




    def _data_generator(self):
        # defines core of data_generator, how .start() call will lead to
        self.to_initial_state()

        for self.repeat_done in range(self.repeat):
            for indices, x in self._iterate_data_x():
                if not self.is_running:
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



    def stop(self):
        if self.thread.is_alive():
            self.is_running = False
            self.thread.join()
        self.to_final_state()
        self.loaded_params = False


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
    def update_data_y(self, i):
        # defines how to update data_y
        # normally will be
        # counts = self.counter(self.exposure, parent=self) 
        # data_y[i] += counts
        pass

    @abstractmethod
    def assign_names(self):
        # defines measurement names etc.
        # init assignment of config instances
        pass

    @abstractmethod
    def load_params(self):
        # defines how to load params
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

        if hasattr(cls, 'load_params'):
            measure_func.__doc__ = cls.load_params.__doc__

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


@register_measurement("ple") #allow a faster call to .load_params() using ple()
class PLEMeasurement(BaseMeasurement):

    def device_to_state(self, wavelength):
        # move device state to x from data_x
        self.laser_stabilizer.on = True
        self.laser_stabilizer.wavelength = wavelength
        while self.is_running:
            time.sleep(0.01)
            if self.laser_stabilizer.is_ready:
                break

    def to_initial_state(self):
        # move device/data state to initial state before measurement
        self.laser_stabilizer.on = True

    def to_final_state(self):
        # move device/data state to final state after measurement
        self.laser_stabilizer.on = False

    def read_x(self):
        return self.wavemeter.wavelength

    def update_data_y(self, i):
        counts = self.counter.read_counts(self.exposure, parent = self, counter_mode=self.counter_mode, data_mode=self.data_mode)
        self.data_y[i] = counts if np.isnan(self.data_y[i]) else (self.data_y[i] + counts)

    def assign_names(self):
        # only assign once measurement is created
        self.x_name = 'Wavelength'
        self.x_unit = 'nm'
        self.measurement_name = 'PLE'
        self.x_device_name = 'wavemeter'
        # defines the label name used in GUI
        self.plot_type = '1D'
        self.fit_func = 'lorent'
        self.loaded_params = False

        self.counter = self.config_instances.get('counter', None)
        self.wavemeter = self.config_instances.get('wavemeter', None)
        self.laser_stabilizer = self.config_instances.get('laser_stabilizer', None)
        self.scanner = self.config_instances.get('scanner', None)
        # init assignment
        if (self.counter is None) or (self.wavemeter is None) or (self.laser_stabilizer is None):
            raise KeyError('Missing devices in config_instances')


    def load_params(self, data_x=None, exposure=0.1, config_instances=None, repeat=1, is_GUI=False, \
        counter_mode='apd', data_mode='single', relim_mode='normal', is_plot=True):
        """
        ple func doc str
        """
        self.loaded_params = True
        if data_x is None:
            data_x = np.arange(737.1-0.005, 737.1+0.005, 0.0005)
        self.data_x = data_x
        self.data_y = np.full(np.shape(self.data_x), np.nan)
        self.exposure = exposure
        self.repeat = repeat
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
            self.measurement_Live = live(config_instances=self.config_instances, is_plot=False)
            GUI_PLE(config_instances = self.config_instances, measurement_PLE=self, measurement_Live=self.measurement_Live)
            return None, None
        else:
            data_x = self.data_x
            data_y = self.data_y
            data_generator = self
            liveplot = PLELive(labels=[f'{self.x_name} ({self.x_unit})', f'Counts/{self.exposure}s'], \
                                update_time=0.1, data_generator=data_generator, data=[data_x, data_y], \
                                config_instances = self.config_instances, relim_mode=self.relim_mode)
            fig, selector = liveplot.plot()
            data_figure = DataFigure(liveplot)
            return fig, data_figure


@register_measurement('odmr') #allow a faster call to .load_params() using ple()
class ODMRMeasurement(BaseMeasurement):

    def device_to_state(self, frequency):
        # move device state to x from data_x, defaul frequency in GHz
        self.rf.on = True
        self.rf.frequency = frequency*1e9


    def to_initial_state(self):
        # move device/data state to initial state before measurement
        self.rf.power = self.power
        self.rf.on = True


    def to_final_state(self):
        # move device/data state to final state after measurement
        self.rf.on = False


    def read_x(self):
        return self.rf.frequency/1e9

    def update_data_y(self, i):
        counts = self.counter.read_counts(self.exposure, parent = self, counter_mode=self.counter_mode, data_mode=self.data_mode)
        self.data_y[i] = counts if np.isnan(self.data_y[i]) else (self.data_y[i] + counts)

    def assign_names(self):

        self.x_name = 'Frequency'
        self.x_unit = 'GHz'
        self.measurement_name = 'ODMR'
        self.x_device_name = 'RF'
        self.plot_type = '1D'
        self.fit_func = 'lorent'
        self.loaded_params = False

        self.counter = self.config_instances.get('counter', None)
        self.rf = self.config_instances.get('rf', None)
        self.scanner = self.config_instances.get('scanner', None)
        # init assignment
        if (self.counter is None) or (self.rf is None):
            raise KeyError('Missing devices in config_instances')


    def load_params(self, data_x=None, exposure=0.1, power=-10, config_instances=None, repeat=1, is_GUI=False, \
        counter_mode='apd', data_mode='single', relim_mode='tight', is_plot=True):
        """
        odmr func doc str
        """
        self.loaded_params = True
        if data_x is None:
            data_x = np.arange(2.88-0.1, 2.88+0.1, 0.001)
        self.data_x = data_x
        self.data_y = np.full(np.shape(self.data_x), np.nan)
        self.exposure = exposure
        self.repeat = repeat
        self.is_GUI = is_GUI
        self.counter_mode = counter_mode
        self.data_mode = data_mode
        self.relim_mode = relim_mode
        self.power = power
        self.is_plot = is_plot
        self.info = {'measurement_name':self.measurement_name, 'plot_type':self.plot_type, 'exposure':self.exposure\
                    , 'repeat':self.repeat, 'power':self.power, 'scanner':(None if self.scanner is None else (self.scanner.x, self.scanner.y))}

    def plot(self, **kwargs):
        self.load_params(**kwargs)
        if not self.is_plot:
            return self

        if self.is_GUI:
            self.measurement_Live = live(config_instances=self.config_instances, is_plot=False)
            GUI_PLE(config_instances = self.config_instances, measurement_PLE=self, measurement_Live=self.measurement_Live)
            return None, None
        else:
            data_x = self.data_x
            data_y = self.data_y
            data_generator = self
            liveplot = PLELive(labels=[f'{self.x_name} ({self.x_unit})', f'Counts/{self.exposure}s'], \
                                update_time=0.1, data_generator=data_generator, data=[data_x, data_y], \
                                config_instances = self.config_instances, relim_mode=self.relim_mode)
            fig, selector = liveplot.plot()
            data_figure = DataFigure(liveplot)
            return fig, data_figure


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

    def update_data_y(self, i):
        counts = self.counter.read_counts(self.exposure, parent = self, counter_mode=self.counter_mode, data_mode=self.data_mode)
        self.data_y[:] = np.roll(self.data_y, 1)
        self.data_y[0] = counts 

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


    def load_params(self, data_x=None, exposure=0.1, config_instances=None, is_finite=False, is_GUI=False, repeat=1, \
        counter_mode='apd', data_mode='single', relim_mode='normal', is_plot=True):
        """
        live func doc str
        """
        self.loaded_params = True
        if is_finite==False:
            self.repeat = int(1e6)
        else:
            self.repeat = repeat
        # large enough and in practical infinite
        if data_x is None:
            data_x = np.arange(100)
        self.data_x = data_x
        self.data_y = np.full(np.shape(self.data_x), np.nan)
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
            return None, None
        else:
            data_x = self.data_x
            data_y = self.data_y
            data_generator = self
            liveplot = PLELive(labels=[f'{self.x_name} ({self.x_unit})', f'Counts/{self.exposure}s'], \
                                update_time=0.01, data_generator=data_generator, data=[data_x, data_y], \
                                config_instances = self.config_instances, relim_mode=self.relim_mode)
            fig, selector = liveplot.plot()
            data_figure = DataFigure(liveplot)
            return fig, data_figure



@register_measurement('pl') #allow a faster call to .plot() using pl()
class PLMeasurement(BaseMeasurement):

    def device_to_state(self, x_value):
        # move device state to x from data_x, defaul frequency in GHz
        x, y = x_value
        self.scanner.x = x
        self.scanner.y = y


    def to_initial_state(self):
        # move device/data state to initial state before measurement
        if self.is_stable:
        
            self.laser_stabilizer.on = True
            self.laser_stabilizer.wavelength = self.wavelength
            
            while self.is_running:
                time.sleep(0.01)
                if self.laser_stabilizer.is_ready:
                    break


    def to_final_state(self):
        # move device/data state to final state after measurement
        if self.is_stable:
            self.laser_stabilizer.on = False

    def read_x(self):
        x = self.scanner.x
        y = self.scanner.y
        return (x, y)

    def update_data_y(self, index):
        counts = self.counter.read_counts(self.exposure, parent = self, counter_mode=self.counter_mode, data_mode=self.data_mode)
        i, j = index
 
        self.data_y[j][i] = counts if np.isnan(self.data_y[j][i]) else (self.data_y[j][i]+counts)
        # if has data then set to np.nan

    def assign_names(self):

        self.plot_type = '2D'
        self.measurement_name = 'PL'
        self.x_device_name = 'None'
        self.x_unit = ''

        self.counter = self.config_instances.get('counter', None)
        self.scanner = self.config_instances.get('scanner', None)
        self.loaded_params = False
        # init assignment
        if (self.counter is None) or (self.scanner is None):
            raise KeyError('Missing devices in config_instances')

    def load_params(self, data_x=None, exposure=0.1, config_instances=None, repeat = 1, wavelength=None, is_GUI=False, is_dis=True, \
        counter_mode='apd', data_mode='single', relim_mode='normal', is_plot=True):
        """
        pl func doc str
        """
        self.loaded_params = True
        self.is_dis = is_dis
        if wavelength is None:
            self.is_stable = False
            self.wavelength = None
        else:
            self.is_stable = True
            self.wavelength = wavelength
            self.wavemeter = self.config_instances.get('wavemeter')
            self.laser_stabilizer = self.config_instances.get('laser_stabilizer')

        if data_x is None:
            data_x = [np.arange(-10, 10, 1), np.arange(-10, 10, 1)]
        self.data_x = data_x
        self.data_y = np.full((len(self.data_x[1]), len(self.data_x[0])), np.nan)
        # set nan as default of no data

        self.exposure = exposure
        self.is_GUI = is_GUI
        self.repeat = repeat
        self.counter_mode = counter_mode
        self.data_mode = data_mode
        self.relim_mode = relim_mode
        self.is_plot = is_plot
        self.info = {'measurement_name':self.measurement_name, 'plot_type':self.plot_type, 'exposure':self.exposure\
                    , 'repeat':self.repeat, 'wavelength':self.wavelength}

    def plot(self, **kwargs):
        self.load_params(**kwargs)
        if not self.is_plot:
            return self
        if self.is_GUI:
            # defines the label name used in GUI
            self.measurement_Live = live(config_instances=self.config_instances, is_plot=False)
            GUI_PL(config_instances = self.config_instances, measurement_PL=self, measurement_Live=self.measurement_Live)
            return None, None
        else:
            data_x = self.data_x
            data_y = self.data_y
            data_generator = self
            if self.is_dis:
                liveplot = PLDisLive(labels=[['X', 'Y'], f'Counts/{self.exposure}s'], \
                                    update_time=1, data_generator=data_generator, data=[data_x, data_y], \
                                    config_instances = self.config_instances, relim_mode=self.relim_mode)
            else:
                liveplot = PLLive(labels=[['X', 'Y'], f'Counts/{self.exposure}s'], \
                                    update_time=1, data_generator=data_generator, data=[data_x, data_y], \
                                    config_instances = self.config_instances, relim_mode=self.relim_mode)

            fig, selector = liveplot.plot()
            data_figure = DataFigure(liveplot)
            return fig, data_figure


        
  

# ----------------------- below will write soon-----------------------------------------------

class TaggerAcquire(threading.Thread):

    #class for time tagger measurement
    def __init__(self, click_channel, start_channel, binwidth, n_bins, data_x, data_y, duration, config_instances):
        super().__init__()

        from TimeTagger import createTimeTagger, Histogram

        tagger = createTimeTagger('1809000LGG')
        tagger.reset()
        self.tagger = tagger
        self.Histogram = Histogram
        self.duration = duration
        self.data_x = data_x
        self.data_y = data_y
        self.daemon = True
        self.is_running = True
        self.is_done = False
        self.config_instances = config_instances
        self.points_done = n_bins
        self.binwidth = binwidth
        self.scanner = config_instances['scanner']
        self.wavemeter = config_instances['wavemeter']

        self.info = {'data_generator':'PLEAcquire', 'exposure':self.duration, 'binwidth':self.binwidth, \
            'wavelength':self.wavemeter.wavelength, 'scanner':[self.scanner.x, self.scanner.y]}
        # important information to be saved with figures 

        self.click_channel = click_channel
        self.start_channel = start_channel
        self.n_bins = n_bins
        
    
    def run(self):
        
        self.histogram = self.Histogram(tagger=self.tagger, click_channel=self.click_channel , \
            start_channel=self.start_channel , binwidth=self.binwidth , n_bins=self.n_bins )
        self.histogram.startFor(int(self.duration*1e12))
        while self.histogram.isRunning():
            time.sleep(1)
            self.data_y[:] = self.histogram.getData()
            
        self.is_done = True
        #finish all data
        self.histogram.stop()
        # stop and join child thread
        
    def stop(self):
        self.histogram.stop()
        if self.is_alive():
            self.is_running = False
            self.join()
        
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()

def time_tagger(click_channel, start_channel, binwidth, n_bins, duration, config_instances):
                
    data_x = np.linspace(0, n_bins*binwidth/1e3, n_bins)
    data_y = np.zeros(len(data_x))
    data_generator = TaggerAcquire(click_channel=click_channel , \
            start_channel=start_channel , binwidth=binwidth , n_bins=n_bins, data_x=data_x, data_y=data_y, \
            config_instances = config_instances, duration = duration)
    liveplot = PLELive(labels=['Time (ns)', f'Counts'], \
                        update_time=0.5, data_generator=data_generator, data=[data_x, data_y], config_instances = config_instances\
                        , relim_mode='tight')
    fig, selector = liveplot.plot()
    data_figure = DataFigure(liveplot)
    return fig, data_figure