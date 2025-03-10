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
from .base import register_measurement, BaseMeasurement, live



@register_measurement("ple") #allow a faster call to .load_params() using ple()
class PLEMeasurement(BaseMeasurement):

    def device_to_state(self, wavelength):
        # move device state to x from data_x
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


    def _load_params(self, data_x=None, exposure=0.1, repeat=1, is_GUI=False,
        counter_mode='apd', data_mode='single', relim_mode='tight', update_mode='normal', is_plot=True):
        """
        ple

        args:
        (data_x=None, exposure=0.1, repeat=1, is_GUI=False,
        counter_mode='apd', data_mode='single', relim_mode='tight'):

        example:
        fig, data_figure = ple(data_x=np.arange(737.1-0.005, 737.1+0.005, 0.0005), exposure=0.1,
                                repeat=1, is_GUI=False,
                                counter_mode='apd', data_mode='single', relim_mode='tight')

        """
        self.loaded_params = True
        if data_x is None:
            data_x = np.arange(737.1-0.005, 737.1+0.005, 0.0005)
        self.data_x = data_x
        self.set_update_mode(update_mode=update_mode)
        self.exposure = exposure
        self.repeat = repeat
        self.is_GUI = is_GUI
        self.counter_mode = counter_mode
        self.data_mode = data_mode
        self.relim_mode = relim_mode
        self.is_plot = is_plot
        self.info.update({'measurement_name':self.measurement_name, 'plot_type':self.plot_type, 'exposure':self.exposure
                , 'repeat':self.repeat, 'scanner':(None if self.scanner is None else (self.scanner.x, self.scanner.y))})

    def plot(self, **kwargs):
        self.load_params(**kwargs)
        if not self.is_plot:
            return self

        if self.is_GUI:
            self.measurement_Live = live(is_plot=False)
            GUI_PLE(measurement_PLE=self, measurement_Live=self.measurement_Live)
            return None, None
        else:
            data_x = self.data_x
            data_y = self.data_y
            data_generator = self
            update_time = float(np.max([1, self.exposure*len(data_x)/1000]))
            liveplot = PLELive(labels=[f'{self.x_name} ({self.x_unit})', f'Counts/{self.exposure}s'],
                                update_time=update_time, data_generator=data_generator, data=[data_x, data_y],
                                relim_mode=self.relim_mode)
            fig, selector = liveplot.plot()
            data_figure = DataFigure(liveplot)
            return fig, data_figure


@register_measurement('odmr') #allow a faster call to .load_params() using ple()
class ODMRMeasurement(BaseMeasurement):

    def device_to_state(self, frequency):
        # move device state to x from data_x, defaul frequency in GHz
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


    def _load_params(self, data_x=None, exposure=0.1, power=None, repeat=1, is_GUI=False,
        counter_mode='apd', data_mode='single', relim_mode='tight', update_mode='normal', is_plot=True):
        """
        odmr

        args:
        (data_x=None, exposure=0.1, power=-10, repeat=1, is_GUI=False,
        counter_mode='apd', data_mode='single', relim_mode='tight'):

        example:
        fig, data_figure = odmr(data_x=np.arange(2.88-0.1, 2.88+0.1, 0.001), exposure=0.1,
                                power=-10,
                                repeat=1, is_GUI=False,
                                counter_mode='apd', data_mode='single', relim_mode='tight')

        """
        self.loaded_params = True
        if data_x is None:
            data_x = np.arange(2.88-0.1, 2.88+0.1, 0.001)
        self.data_x = data_x       
        self.exposure = exposure
        self.repeat = repeat
        self.is_GUI = is_GUI
        self.counter_mode = counter_mode
        self.data_mode = data_mode
        self.relim_mode = relim_mode
        self.set_update_mode(update_mode=update_mode)
        # for non basic params, load state from device's state
        self.power = self.rf.power if power is None else power

        self.is_plot = is_plot
        self.info.update({'measurement_name':self.measurement_name, 'plot_type':self.plot_type, 'exposure':self.exposure
                    , 'repeat':self.repeat, 'power':self.power, 'scanner':(None if self.scanner is None else (self.scanner.x, self.scanner.y))}
        )

    def plot(self, **kwargs):
        self.load_params(**kwargs)
        if not self.is_plot:
            return self

        if self.is_GUI:
            self.measurement_Live = live(is_plot=False)
            GUI_PLE(measurement_PLE=self, measurement_Live=self.measurement_Live)
            return None, None
        else:
            data_x = self.data_x
            data_y = self.data_y
            data_generator = self
            update_time = float(np.max([1, self.exposure*len(data_x)/1000]))
            liveplot = PLELive(labels=[f'{self.x_name} ({self.x_unit})', f'Counts/{self.exposure}s'],
                                update_time=update_time, data_generator=data_generator, data=[data_x, data_y],
                                relim_mode=self.relim_mode)
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

    def _load_params(self, x_array=None, y_array=None, exposure=0.1, repeat = 1, wavelength=None, is_GUI=False, is_dis=True,
        counter_mode='apd', data_mode='single', relim_mode='tight', is_plot=True):
        """
        pl

        args:
        (x_array=None, y_array=None, exposure=0.1, repeat=1, wavelength=None, 
        is_GUI=False, is_dis=True, 
        counter_mode='apd', data_mode='single', relim_mode='tight'):

        example:
        fig, data_figure = pl(x_array = np.arange(-10, 10, 1), y_array = np.arange(-10, 10, 1), exposure=0.1,
                                repeat=1, is_GUI=False, is_dis=True,
                                counter_mode='apd', data_mode='single', relim_mode='tight')

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

        if (x_array is None) or (y_array is None):
            x_array = np.arange(-10, 10, 1)
            y_array = np.arange(-10, 10, 1)
        self.x_array = x_array
        self.y_array = y_array
        self.data_x = []
        for y in self.y_array:
            for x in self.x_array:
                self.data_x.append((x, y))
        self.data_x = np.array(self.data_x)        
        # set nan as default of no data

        self.exposure = exposure
        self.is_GUI = is_GUI
        self.repeat = repeat
        self.counter_mode = counter_mode
        self.data_mode = data_mode
        self.relim_mode = relim_mode
        self.set_update_mode(update_mode='single')
        self.is_plot = is_plot
        self.info.update({'measurement_name':self.measurement_name, 'plot_type':self.plot_type, 'exposure':self.exposure
                    , 'repeat':self.repeat, 'wavelength':self.wavelength})

    def plot(self, **kwargs):
        self.load_params(**kwargs)
        if not self.is_plot:
            return self
        if self.is_GUI:
            # defines the label name used in GUI
            self.measurement_Live = live(is_plot=False)
            GUI_PL(measurement_PL=self, measurement_Live=self.measurement_Live)
            return None, None
        else:
            data_x = self.data_x
            data_y = self.data_y
            data_generator = self
            update_time = float(np.max([1, self.exposure*len(data_x)/1000]))
            if self.is_dis:
                liveplot = PLDisLive(labels=[['X', 'Y'], f'Counts/{self.exposure}s'],
                                    update_time=update_time, data_generator=data_generator, data=[data_x, data_y],
                                    relim_mode=self.relim_mode)
            else:
                liveplot = PLLive(labels=[['X', 'Y'], f'Counts/{self.exposure}s'],
                                    update_time=update_time, data_generator=data_generator, data=[data_x, data_y],
                                    relim_mode=self.relim_mode)

            fig, selector = liveplot.plot()
            data_figure = DataFigure(liveplot)
            return fig, data_figure


@register_measurement("mode_search") #allow a faster call to .load_params() using mode_search()
class ModeSearchMeasurement(BaseMeasurement):

    def load_params(self, **kwargs):
        self._load_params(**kwargs)

        len_counts = len(self.counter.read_counts(0.1, parent = self, counter_mode=self.counter_mode, data_mode=self.data_mode))

        self.data_y = np.full((len(self.data_x), 1), np.nan)

        self.data_y_counts = np.full(np.shape(self.data_y), np.nan)
        self.data_y_exposure = np.full(np.shape(self.data_x), np.nan)
        self.data_y_ref = np.full(np.shape(self.data_y), np.nan)
        self.data_y_ref_index = np.full(np.shape(self.data_x), np.nan)

        self.info.update({'threshold_in_sigma':self.threshold_in_sigma, 'ref_gap':self.data_y_ref_gap,
         'ref_exposure_repeat':self.ref_exposure_repeat, 'max_exposure_repeat':self.max_exposure_repeat,
         'data_y_counts':self.data_y_counts, 'data_y_exposure':self.data_y_exposure, 'data_y_ref':self.data_y_ref,
         'data_y_ref_index':self.data_y_ref_index}
        )
        # update info finally to record self.data_y_counts etc.

    def update_data_y(self, i):
        # defines how to update data_y
        # normally will be
        # counts = self.counter(self.exposure, parent=self) 
        # data_y[i] += counts

        if (time.time()-self.data_y_ref_time)>=self.data_y_ref_gap or (not hasattr(self, 'recent_ref')):
            self.recent_ref = 0
            self.device_to_state(self.data_x[0])
            for ii in range(self.ref_exposure_repeat):
                self.recent_ref += self.counter.read_counts(self.exposure, parent = self, 
                    counter_mode=self.counter_mode, data_mode=self.data_mode)[0]
            self.data_y_ref[i] = self.recent_ref
            self.data_y_ref_index[i] = self.data_x[i]
            self.data_y_ref_time = time.time()
            self.device_to_state(self.data_x[i])

        counts = self.counter.read_counts(self.exposure, parent = self, counter_mode=self.counter_mode, data_mode=self.data_mode)[0]
        exposure = self.exposure
        ref_counts_norm = self.recent_ref*exposure/(self.ref_exposure_repeat*self.exposure)
        while (((counts -  ref_counts_norm) > self.threshold_in_sigma*np.sqrt(ref_counts_norm)) 
            and exposure<=(self.max_exposure_repeat*self.exposure)
        ):
            counts += self.counter.read_counts(self.exposure, parent = self, counter_mode=self.counter_mode, data_mode=self.data_mode)[0]
            exposure += self.exposure
            ref_counts_norm = self.recent_ref*exposure/(self.ref_exposure_repeat*self.exposure)
        self.data_y[i] = [(counts -  ref_counts_norm)/np.sqrt(ref_counts_norm),]
        self.data_y_counts[i] = [counts,]
        self.data_y_exposure[i] = exposure

    def set_update_mode(self, **kwargs):
        self.data_y_ref_time = time.time()
        self.counter.read_counts(self.exposure, parent = self, counter_mode=self.counter_mode, data_mode=self.data_mode)
        if self.counter_mode == 'apd_gp':
            self.exposure = self.counter.exposure
        # reload exposure based on counter setting

        self.threshold_in_sigma = kwargs.get('threshold_in_sigma', 4)
        self.data_y_ref_gap = kwargs.get('ref_gap', 10)
        self.ref_exposure_repeat = int(np.ceil(kwargs.get('ref_exposure', 1)/self.exposure))
        self.max_exposure_repeat = int(np.ceil(kwargs.get('max_exposure', 1)/self.exposure))

# --------------------------------- overwrite BaseMeasurement methods above ----------------------

    def device_to_state(self, frequency):
        # move device state to x from data_x
        self.rf_1550.frequency = frequency

    def to_initial_state(self):
        # move device/data state to initial state before measurement
        self.laser_stabilizer.on = True
        self.laser_stabilizer.wavelength = self.wavelength
        while self.is_running:
            time.sleep(0.01)
            if self.laser_stabilizer.is_ready:
                break

    def to_final_state(self):
        # move device/data state to final state after measurement
        self.laser_stabilizer.on = False

    def read_x(self):
        return self.rf_1550.frequency

    def assign_names(self):
        # only assign once measurement is created
        self.x_name = 'Frequency'
        self.x_unit = 'Hz'
        self.measurement_name = 'Mode'
        self.x_device_name = 'rf_1550'
        # defines the label name used in GUI
        self.plot_type = '1D'
        self.fit_func = 'lorent'
        self.loaded_params = False

        self.counter = self.config_instances.get('counter', None)
        self.wavemeter = self.config_instances.get('wavemeter', None)
        self.laser_stabilizer = self.config_instances.get('laser_stabilizer', None)
        self.scanner = self.config_instances.get('scanner', None)
        self.rf_1550 = self.config_instances.get('rf_1550', None)
        # init assignment
        if (self.counter is None) or (self.wavemeter is None) or (self.laser_stabilizer is None) or (self.rf_1550 is None):
            raise KeyError('Missing devices in config_instances')


    def _load_params(self, data_x=None, exposure=0.01, wavelength=737.1, repeat=1, is_GUI=False,
        counter_mode='apd', data_mode='single', relim_mode='tight', is_plot=True, threshold_in_sigma=3, 
        ref_gap=10, ref_exposure=1, max_exposure=1):
        """
        ple

        args:
        (data_x=None, exposure=0.01, wavelength=737.1, repeat=1, is_GUI=False,
        counter_mode='apd', data_mode='single', relim_mode='tight', is_plot=True, **kwargs):

        example:
        fig, data_figure = mode_search(data_x=np.arange(1e9-10e6, 1e9+10e6, 0.1e3), exposure=0.01,
                                wavelength=737.1, repeat=1, is_GUI=False,
                                counter_mode='apd', data_mode='single', relim_mode='tight',
                                threshold_in_sigma=3, ref_gap=10, ref_exposure=1, max_exposure=1)

        """
        self.loaded_params = True
        if data_x is None:
            data_x = np.arange(1e9-10e6, 1e9+10e6, 0.1e3)
        self.data_x = data_x
        self.exposure = exposure
        self.repeat = repeat
        self.wavelength = wavelength
        self.is_GUI = is_GUI
        self.counter_mode = counter_mode
        self.data_mode = data_mode
        self.relim_mode = relim_mode
        self.is_plot = is_plot
        self.info.update({'measurement_name':self.measurement_name, 'plot_type':self.plot_type, 'exposure':self.exposure
                , 'repeat':self.repeat, 'scanner':(None if self.scanner is None else (self.scanner.x, self.scanner.y))})
        self.set_update_mode(threshold_in_sigma=threshold_in_sigma, ref_gap=ref_gap, ref_exposure=ref_exposure, max_exposure=max_exposure)

    def plot(self, **kwargs):
        self.load_params(**kwargs)
        if not self.is_plot:
            return self

        if self.is_GUI:
            self.measurement_Live = live(is_plot=False)
            GUI_PLE(measurement_PLE=self, measurement_Live=self.measurement_Live)
            return None, None
        else:
            data_x = self.data_x
            data_y = self.data_y
            data_generator = self
            update_time = float(np.max([1, self.exposure*len(data_x)/1000]))
            liveplot = PLELive(labels=[f'{self.x_name} ({self.x_unit})', f'Sigma/{self.exposure}s'],
                                update_time=update_time, data_generator=data_generator, data=[data_x, data_y],
                                relim_mode=self.relim_mode)
            fig, selector = liveplot.plot()
            data_figure = DataFigure(liveplot)
            return fig, data_figure

  

# ----------------------- below will write soon-----------------------------------------------

class TaggerAcquire(threading.Thread):

    #class for time tagger measurement
    def __init__(self, click_channel, start_channel, binwidth, n_bins, data_x, data_y, duration):
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