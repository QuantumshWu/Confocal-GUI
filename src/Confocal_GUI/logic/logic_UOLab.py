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



class RabiAcquire(threading.Thread):
    """
    class for rabi measurement
    """
    def __init__(self, exposure, data_x, data_y, power, frequency, time_array, config_instances, repeat=1, counter_mode = 'apd', data_mode = 'ref_div'):
        super().__init__()
        from .device import Pulse
        self.exposure = exposure
        self.data_x = data_x
        self.data_y = data_y
        self.daemon = True
        self.is_running = True
        self.is_done = False
        self.counter = config_instances.get('counter')
        self.rf = config_instances.get('rf')
        self.config_instances = config_instances
        self.points_done = 0
        self.repeat = repeat
        self.counter_mode = counter_mode
        self.data_mode = data_mode
        self.repeat_done = 1
        self.power = power
        self.frequency = frequency
        self.time_array = time_array
        # time array is [init duration, gap1, RF total duration, gap2, Readout duration, gap3 with laser on] in ns
        self.info = {'data_generator':'PLEAcquire', 'exposure':self.exposure, 'repeat':self.repeat}
        # important information to be saved with figures 

        self.rf.power = self.power
        self.rf.frequency = frequency
        self.rf.on = True
        self.pulse = Pulse()

    def set_duration(self, duration):

        # ch0 is green, ch1 is RF, ch4 is counter
        delay_array = [-1e3, 0, 0, 0, 0, 0, 0, 0]
        data_matrix = [[self.time_array[0], (0, )], [self.time_array[1], ()], [duration, (1,)], [self.time_array[3]+self.time_array[2]-duration, ()], \
        [self.time_array[4], (0, 4)], [self.time_array[5], (0,)],\
                        [self.time_array[0], (0, )], [self.time_array[1], ()], [duration, ()], [self.time_array[3]+self.time_array[2]-duration, ()], \
        [self.time_array[4], (0, 5)], [self.time_array[5], (0,)]]

        self.pulse.set_timing_simple(data_matrix)
        self.pulse.set_delay(delay_array)
        self.pulse.on_pulse()

    
    def run(self):
        
        
        for self.repeat_done in range(self.repeat):

            for i, duration in enumerate(self.data_x):

                if not self.is_running:
                    self.rf.on = False
                    return 


                self.set_duration(duration)


                counts = self.counter.read_counts(self.exposure, counter_mode = self.counter_mode, data_mode = self.data_mode)
                self.points_done += 1

                self.data_y[i] += counts
            
        self.is_done = True
        self.rf.on = False
        #finish all data
        # stop and join child thread
        
    def stop(self):
        if self.is_alive():
            self.is_running = False
            self.join()
        
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()

def rabi(duration_array, exposure, power, frequency, time_array, config_instances, repeat=1, counter_mode = 'apd', data_mode = 'ref_div'):
                
    data_x = duration_array
    data_y = np.zeros(len(data_x))
    data_generator = RabiAcquire(exposure = exposure, data_x=data_x, data_y=data_y, power = power, frequency = frequency, time_array = time_array, \
        config_instances = config_instances, repeat=repeat, counter_mode=counter_mode, data_mode=data_mode)
    liveplot = PLELive(labels=['Duration (ns)', f'Counts/{exposure}s'], \
                        update_time=0.1, data_generator=data_generator, data=[data_x, data_y], config_instances = config_instances, relim_mode='tight')
    fig, selector = liveplot.plot()
    data_figure = DataFigure(liveplot)
    return fig, data_figure


class RamseyAcquire(threading.Thread):
    """
    class for ramsey measurement

    pulses pi/2-tau-pi/2

    """
    def __init__(self, exposure, data_x, data_y, power, frequency, time_array, config_instances, repeat=1, counter_mode = 'apd', data_mode = 'ref_div'):
        super().__init__()
        from .device import Pulse
        self.exposure = exposure
        self.data_x = data_x
        self.data_y = data_y
        self.daemon = True
        self.is_running = True
        self.is_done = False
        self.counter = config_instances.get('counter')
        self.rf = config_instances.get('rf')
        self.config_instances = config_instances
        self.points_done = 0
        self.repeat = repeat
        self.counter_mode = counter_mode
        self.data_mode = data_mode
        self.repeat_done = 1
        self.power = power
        self.frequency = frequency
        self.time_array = time_array
        # time array is [init duration, gap1, pi/2, RF total duration, pi/2, gap2, Readout duration, gap3 with laser on] in ns
        self.info = {'data_generator':'PLEAcquire', 'exposure':self.exposure, 'repeat':self.repeat}
        # important information to be saved with figures 

        self.rf.power = self.power
        self.rf.frequency = frequency
        self.rf.on = True
        self.pulse = Pulse()

    def set_duration(self, duration):

        # ch0 is green, ch1 is RF, ch4 is counter
        delay_array = [-3e3, -2e3, 0, 0, 0, 0, 0, 0]
        data_matrix = [[self.time_array[0], (0,)], [self.time_array[1], ()], [self.time_array[2], (1,)], [duration, ()], \
        [self.time_array[4], (1,)], [self.time_array[5]+self.time_array[3]-duration, ()], [self.time_array[6], (0, 4)], [self.time_array[7], (0, )], \
                        [self.time_array[0], (0,)], [self.time_array[1], ()], [self.time_array[2], ()], [duration, ()], \
        [self.time_array[4], ()], [self.time_array[5]+self.time_array[3]-duration, ()], [self.time_array[6], (0, 5)], [self.time_array[7], (0,)]] 

        # repat second time for ref, and last terms are compensation for total length

        self.pulse.set_timing_simple(data_matrix)
        self.pulse.set_delay(delay_array)
        self.pulse.on_pulse()

    
    def run(self):
        
        
        for self.repeat_done in range(self.repeat):

            for i, duration in enumerate(self.data_x):

                if not self.is_running:
                    self.rf.on = False
                    return 


                self.set_duration(duration)


                counts = self.counter.read_counts(self.exposure, counter_mode = self.counter_mode, data_mode = self.data_mode)
                self.points_done += 1

                self.data_y[i] += counts
            
        self.is_done = True
        self.rf.on = False
        #finish all data
        # stop and join child thread
        
    def stop(self):
        if self.is_alive():
            self.is_running = False
            self.join()
        
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()

def ramsey(duration_array, exposure, power, frequency, time_array, config_instances, repeat=1, counter_mode = 'apd', data_mode = 'ref_div'):
                
    data_x = duration_array
    data_y = np.zeros(len(data_x))
    data_generator = RamseyAcquire(exposure = exposure, data_x=data_x, data_y=data_y, power = power, frequency = frequency, time_array = time_array, \
        config_instances = config_instances, repeat=repeat, counter_mode=counter_mode, data_mode=data_mode)
    liveplot = PLELive(labels=['Duration (ns)', f'Counts/{exposure}s'], \
                        update_time=0.1, data_generator=data_generator, data=[data_x, data_y], config_instances = config_instances, relim_mode='tight')
    fig, selector = liveplot.plot()
    data_figure = DataFigure(liveplot)
    return fig, data_figure


class SpinechoAcquire(threading.Thread):
    """
    class for spin echo measurement

    pulses pi/2-tau/2-pi-tau/2-pi/2

    """
    def __init__(self, exposure, data_x, data_y, power, frequency, time_array, config_instances, repeat=1, counter_mode = 'apd', data_mode = 'ref_div'):
        super().__init__()
        from .device import Pulse
        self.exposure = exposure
        self.data_x = data_x
        self.data_y = data_y
        self.daemon = True
        self.is_running = True
        self.is_done = False
        self.counter = config_instances.get('counter')
        self.rf = config_instances.get('rf')
        self.config_instances = config_instances
        self.points_done = 0
        self.repeat = repeat
        self.counter_mode = counter_mode
        self.data_mode = data_mode
        self.repeat_done = 1
        self.power = power
        self.frequency = frequency
        self.time_array = time_array
        # time array is [init duration, gap1, pi/2, RF total duration, pi, pi/2, gap2, Readout duration, gap3 with laser on] in ns
        self.info = {'data_generator':'PLEAcquire', 'exposure':self.exposure, 'repeat':self.repeat}
        # important information to be saved with figures 

        self.rf.power = self.power
        self.rf.frequency = frequency
        self.rf.on = True
        self.pulse = Pulse()

    def set_duration(self, duration):

        # ch0 is green, ch1 is RF, ch4 is counter
        delay_array = [-3e3, -2e3, 0, 0, 0, 0, 0, 0]
        data_matrix = [[self.time_array[0], (0,)], [self.time_array[1], ()], [self.time_array[2], (1,)], [duration, ()], \
        [self.time_array[4], (1,)], [duration, ()], [self.time_array[5], (1,)], \
        [self.time_array[6]+self.time_array[3]-2*duration, ()], [self.time_array[7], (0, 4)], [self.time_array[8], (0,)], \
                        [self.time_array[0], (0,)], [self.time_array[1], ()], [self.time_array[2], ()], [duration, ()], \
        [self.time_array[4], ()], [duration, ()], [self.time_array[5], ()], \
        [self.time_array[6]+self.time_array[3]-2*duration, ()], [self.time_array[7], (0, 5)], [self.time_array[8], (0,)]]

        self.pulse.set_timing_simple(data_matrix)
        self.pulse.set_delay(delay_array)
        self.pulse.on_pulse()

    
    def run(self):
        
        
        for self.repeat_done in range(self.repeat):

            for i, duration in enumerate(self.data_x):

                if not self.is_running:
                    self.rf.on = False
                    return 


                self.set_duration(int(duration/2))


                counts = self.counter.read_counts(self.exposure, counter_mode = self.counter_mode, data_mode = self.data_mode)
                self.points_done += 1

                self.data_y[i] += counts
            
        self.is_done = True
        self.rf.on = False
        #finish all data
        # stop and join child thread
        
    def stop(self):
        if self.is_alive():
            self.is_running = False
            self.join()
        
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()


def spinecho(duration_array, exposure, power, frequency, time_array, config_instances, repeat=1, counter_mode = 'apd', data_mode = 'ref_div'):
                
    data_x = duration_array
    data_y = np.zeros(len(data_x))
    data_generator = SpinechoAcquire(exposure = exposure, data_x=data_x, data_y=data_y, power = power, frequency = frequency, time_array = time_array, \
        config_instances = config_instances, repeat=repeat, counter_mode=counter_mode, data_mode=data_mode)
    liveplot = PLELive(labels=['Duration (ns)', f'Counts/{exposure}s'], \
                        update_time=0.1, data_generator=data_generator, data=[data_x, data_y], config_instances = config_instances, relim_mode='tight')
    fig, selector = liveplot.plot()
    data_figure = DataFigure(liveplot)
    return fig, data_figure


class ROdurationAcquire(threading.Thread):
    """
    class for opt RO duration measurement

    init - RF - RO

    """
    def __init__(self, exposure, data_x, data_y, power, frequency, time_array, config_instances, repeat=1, counter_mode = 'apd', data_mode = 'ref_div'):
        super().__init__()
        from .device import Pulse
        self.exposure = exposure
        self.data_x = data_x
        self.data_y = data_y
        self.daemon = True
        self.is_running = True
        self.is_done = False
        self.counter = config_instances.get('counter')
        self.rf = config_instances.get('rf')
        self.config_instances = config_instances
        self.points_done = 0
        self.repeat = repeat
        self.counter_mode = counter_mode
        self.data_mode = data_mode
        self.repeat_done = 1
        self.power = power
        self.frequency = frequency
        self.time_array = time_array
        # time array is [init duration + RF, gap1, RO, Readout green duration, gap2 without laser] in ns
        self.info = {'data_generator':'PLEAcquire', 'exposure':self.exposure, 'repeat':self.repeat}
        # important information to be saved with figures 

        self.rf.power = self.power
        self.rf.frequency = frequency
        self.rf.on = True
        self.pulse = Pulse()

    def set_delay(self, delay):

        # ch0 is green, ch1 is RF, ch4 is counter
        delay_array = [-1e3, 0, 0, 0, delay, delay, 0, 0]
        #data_matrix = [[self.time_array[0], (0, 5)], [self.time_array[1], ()], [self.time_array[2], (1,)], \
        #[self.time_array[3], ()], [duration, (0, )], [self.time_array[0], (0, 4)], [self.time_array[5]+self.time_array[4] - duration - self.time_array[0], (0,)]]

        #data_matrix = [[self.time_array[0], (0, 1)], [self.time_array[1], ()], [self.time_array[2], (4, )], [self.time_array[3], (0,)], \
        #    [self.time_array[4], ()],\
        #                [self.time_array[0], (0,)], [self.time_array[1], ()], [self.time_array[2], (5, )], [self.time_array[3], (0,)], \
        #    [self.time_array[4], ()]]

        #self.pulse.set_timing_simple(data_matrix)
        self.pulse.set_timing_simple(self.time_array)
        self.pulse.set_delay(delay_array)
        self.pulse.on_pulse()

    
    def run(self):
        
        
        for self.repeat_done in range(self.repeat):

            for i, duration in enumerate(self.data_x):

                if not self.is_running:
                    self.rf.on = False
                    return 


                self.set_delay(duration)


                counts = self.counter.read_counts(self.exposure, counter_mode = self.counter_mode, data_mode = self.data_mode)
                self.points_done += 1

                self.data_y[i] += counts
            
        self.is_done = True
        self.rf.on = False
        #finish all data
        # stop and join child thread
        
    def stop(self):
        if self.is_alive():
            self.is_running = False
            self.join()
        
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()

def ROduration(duration_array, exposure, power, frequency, time_array, config_instances, repeat=1, counter_mode = 'apd', data_mode = 'ref_div'):
                
    data_x = duration_array
    data_y = np.zeros(len(data_x))
    data_generator = ROdurationAcquire(exposure = exposure, data_x=data_x, data_y=data_y, power = power, frequency = frequency, time_array = time_array, \
        config_instances = config_instances, repeat=repeat, counter_mode=counter_mode, data_mode=data_mode)
    liveplot = PLELive(labels=['Duration (ns)', f'Counts/{exposure}s'], \
                        update_time=0.1, data_generator=data_generator, data=[data_x, data_y], config_instances = config_instances, relim_mode='tight')
    fig, selector = liveplot.plot()
    data_figure = DataFigure(liveplot)
    return fig, data_figure



class NVT1Acquire(threading.Thread):
    """
    class for NV center T1,spin measurement

    green+RF - wait - RO

    """
    def __init__(self, exposure, data_x, data_y, power, frequency, time_array, config_instances, repeat=1, counter_mode = 'apd', data_mode = 'ref_div'):
        super().__init__()
        from .device import Pulse
        self.exposure = exposure
        self.data_x = data_x
        self.data_y = data_y
        self.daemon = True
        self.is_running = True
        self.is_done = False
        self.counter = config_instances.get('counter')
        self.rf = config_instances.get('rf')
        self.config_instances = config_instances
        self.points_done = 0
        self.repeat = repeat
        self.counter_mode = counter_mode
        self.data_mode = data_mode
        self.repeat_done = 1
        self.power = power
        self.frequency = frequency
        self.time_array = time_array
        # time array is [init+RF duration , gap1, wait_total, gap2, Readout duration, gap3 with laser] in ns
        self.info = {'data_generator':'PLEAcquire', 'exposure':self.exposure, 'repeat':self.repeat}
        # important information to be saved with figures 

        self.rf.power = self.power
        self.rf.frequency = frequency
        self.rf.on = True
        self.pulse = Pulse()

    def set_duration(self, duration):

        # ch0 is green, ch1 is RF, ch4 is counter
        delay_array = [-3e3, -2e3, 0, 0, 0, 0, 0, 0]

        data_matrix = [[self.time_array[0], (0, 1)], [self.time_array[1] + duration + self.time_array[3], ()], [self.time_array[4], (0, 4)], \
                        [self.time_array[5]+self.time_array[2]-duration, (0,)], \
                        [self.time_array[0], (0, )], [self.time_array[1] + duration + self.time_array[3], ()], [self.time_array[4], (0, 5)], \
                        [self.time_array[5]+self.time_array[2]-duration, (0,)]]

        self.pulse.set_timing_simple(data_matrix)
        self.pulse.set_delay(delay_array)
        self.pulse.on_pulse()

    
    def run(self):
        
        
        for self.repeat_done in range(self.repeat):

            for i, duration in enumerate(self.data_x):

                if not self.is_running:
                    self.rf.on = False
                    return 


                self.set_duration(duration)


                counts = self.counter.read_counts(self.exposure, counter_mode = self.counter_mode, data_mode = self.data_mode)
                self.points_done += 1

                self.data_y[i] += counts
            
        self.is_done = True
        self.rf.on = False
        #finish all data
        # stop and join child thread
        
    def stop(self):
        if self.is_alive():
            self.is_running = False
            self.join()
        
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()

def t1(duration_array, exposure, power, frequency, time_array, config_instances, repeat=1, counter_mode = 'apd', data_mode = 'ref_div'):
                
    data_x = duration_array
    data_y = np.zeros(len(data_x))
    data_generator = NVT1Acquire(exposure = exposure, data_x=data_x, data_y=data_y, power = power, frequency = frequency, time_array = time_array, \
        config_instances = config_instances, repeat=repeat, counter_mode=counter_mode, data_mode=data_mode)
    liveplot = PLELive(labels=['Duration (ns)', f'Counts/{exposure}s'], \
                        update_time=0.1, data_generator=data_generator, data=[data_x, data_y], config_instances = config_instances, relim_mode='tight')
    fig, selector = liveplot.plot()
    data_figure = DataFigure(liveplot)
    return fig, data_figure