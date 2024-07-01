import numpy as np


import sys
import time
import threading
from decimal import Decimal
from threading import Event
from scipy.optimize import curve_fit








class LaserStabilizer(threading.Thread):
    """
    stablize laser
    
    >>> laser_stable = LaserStabilizer()
    >>> laser_stable.start()
    >>> laser_stable.set_wavelength(1)
    >>> laser_stable.is_ready
    True
    >>> laser_stable.stop()
    
    or 
    
    >>> with laser_stable = LaserStabilizer():
    >>>    something
    
    """
    def __init__(self, config_instances):
        super().__init__()
        self.is_ready = False
        self.wavelength = None
        self.desired_wavelength = None
        self.is_running = True
        self.is_change = False
        # indicate if wavelnegth has changed
        self.daemon = True
        self.laser_stabilizer_core = config_instances.get('laser_stabilizer_core')
        # will be killed if main thread is killed
        
    def set_wavelength(self, desired_wavelength):
        if self.desired_wavelength != desired_wavelength:
            self.is_ready = False
            self.desired_wavelength = desired_wavelength
            
        # make sure no unexpected order
                
    
    def run(self):
        while self.is_running:
            
            if self.wavelength!=self.desired_wavelength:
                self.is_change = True
                self.wavelength = self.desired_wavelength
                self.is_ready = False
                
            if self.wavelength == None:
                continue

            self.laser_stabilizer_core.wavelength = self.wavelength
            self.laser_stabilizer_core.run()
            # laser_stabilizer_core(), core logic to connect hardware
            
            if self.is_change and self.laser_stabilizer_core.is_ready:
                # only change flags after wavelength changes
                self.is_ready = True
                self.is_change = False

    def stop(self):
        if self.is_alive(): #use () cause it's a method not property
            self.is_running = False
            self.join()
        
        
    # below __enter__ and __exit__ allow class be called using with
    # 
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()
        
        
class VirtualLaserStabilizerCore():
    """
    core logic for stabilizer, test
    
    no real operations, just time.sleep(0.1)
    """
    
    def __init__(self, config_instances, ratio=None):
        self.ratio = ratio
        self.wavemeter = config_instances.get('wavemeter')
        self.laser = config_instances.get('laser')
        self.is_ready = True

        
    def run(self):
        
        time.sleep(0.1) 
        return
        
class LaserStabilizerCore():
    """
    core logic for stabilizer,
    
    .run() will read wavemeter and change laser piezo
    .is_ready=True when wavelength_desired = wavelength_actual
    .wavelength is the wavelength desired
    .ratio = -0.85GHz/V defines the ratio for feedback
    """
    
    def __init__(self, config_instances, ratio=-0.85):
        self.ratio = ratio
        self.wavemeter = config_instances.get('wavemeter')
        self.laser = config_instances.get('laser')
        self.is_ready = False
        self.spl = 299792458
        self.v_mid = 0.5*(self.laser.piezo_max + self.laser.piezo_min)
        self.v_min = self.laser.piezo_min + 0.05*(self.laser.piezo_max - self.laser.piezo_min)
        self.v_max = self.laser.piezo_min + 0.95*(self.laser.piezo_max - self.laser.piezo_min)
        self.freq_recent = self.spl/self.wavemeter.wavelength
        self.freq_thre = 0.05 #50MHz threshold defines when to return is_ready
        # leaves about 10% extra space
        
    @property
    def wavelength(self):
        return self._wavelength
    
    @wavelength.setter
    def wavelength(self, wavelength_in):
        self._wavelength = wavelength_in
        
    def run(self):
        
        freq_desired = self.spl/self.wavelength
        freq_diff_guess = freq_desired - self.freq_recent
        v_diff = freq_diff_guess/self.ratio 
        v_0 = self.laser.piezo
        
        if (v_0+v_diff)<self.v_min or (v_0+v_diff)>self.v_max:
            
            wavelength_now = self.laser.wavelength
            
            freq_by_piezo = (self.v_mid - v_0)*self.ratio
            freq_by_wavelength = freq_diff_guess - freq_by_piezo
            wavelength_set = 0.01*round((self.spl/(freq_by_wavelength + self.spl/wavelength_now))/0.01)
            # freq_diff_guess = freq_wavelength + freq_piezo
            self.laser.piezo = self.v_mid 
            # reset piezo to center
            self.laser.wavelength = wavelength_set
            wavelength_delta = wavelength_set - wavelength_now
            time.sleep(max(np.abs(wavelength_delta)*50, 5))
        else:
            self.laser.piezo = v_0+v_diff

        
        
        freq_actual = self.spl/self.wavemeter.wavelength #wait
        freq_diff = freq_desired - freq_actual
        if np.abs(freq_diff) <= self.freq_thre:
            self.is_ready = True
        else:
            self.is_ready = False
        self.freq_recent = freq_actual    
        return
        
        
class PLEAcquire(threading.Thread):
    """
    class for ple measurement
    """
    def __init__(self, exposure, data_x, data_y, config_instances):
        super().__init__()
        self.exposure = exposure
        self.data_x = data_x
        self.data_y = data_y
        self.daemon = True
        self.is_running = True
        self.is_done = False
        self.counter = config_instances.get('counter')
        self.config_instances = config_instances
        self.points_done = 0
        self.scanner = config_instances.get('scanner')
        self.info = {'data_generator':'PLEAcquire', 'exposure':self.exposure, 'scanner':[self.scanner.x, self.scanner.y]}
        # important information to be saved with figures 
        
    
    def run(self):
        
        
        self.laser_stabilizer = LaserStabilizer(config_instances = self.config_instances)
        self.laser_stabilizer.start()
        
        for i, wavelength in enumerate(self.data_x):
            self.laser_stabilizer.set_wavelength(wavelength)
            while self.is_running:
                time.sleep(0.01)
                if self.laser_stabilizer.is_ready:
                    break
            else:
                return 

            counts = self.counter(self.exposure, self)
            self.points_done += 1

            self.data_y[i] += counts
            
        self.is_done = True
        #finish all data
        self.laser_stabilizer.stop()
        # stop and join child thread
        
    def stop(self):
        self.laser_stabilizer.stop()
        if self.is_alive():
            self.is_running = False
            self.join()
        
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()
        
        
class PLAcquire(threading.Thread):
    """
    class for pl measurement
    
    data_x, data_y, center are coordinates
    
    data_z is returned data
    
    wavelength=None as default
    
    """
    def __init__(self, exposure, data_x, data_y, data_z, config_instances, wavelength=None):
        super().__init__()
        self.exposure = exposure
        self.data_x = data_x
        self.data_y = data_y
        self.data_z = data_z
        self.scanner = config_instances.get('scanner')
        self.counter = config_instances.get('counter')
        self.config_instances = config_instances
        
        self.wavelength = wavelength
        if self.wavelength is None:
            self.is_stable = False
        else:
            self.is_stable = True
        
        self.daemon = True
        self.is_running = True
        self.is_done = False
        self.points_done = 0
        self.info = {'data_generator':'PLAcquire', 'exposure':self.exposure, 'wavelength':self.wavelength}
        # important information to be saved with figures 
        
    
    def run(self):
        
        if self.is_stable:
            self.laser_stabilizer = LaserStabilizer(config_instances = self.config_instances)
            self.laser_stabilizer.start()
            self.laser_stabilizer.set_wavelength(self.wavelength)
            
            while self.is_running:
                time.sleep(0.01)
                if self.laser_stabilizer.is_ready:
                    break
        
        for j, y in enumerate(self.data_y):
            self.scanner.y = y
            for i, x in enumerate(self.data_x):
                self.scanner.x = x
                # reverse x, y order because how imshow displays data
                if not self.is_running:
                    return
                # break loop when interrupt
                counts = self.counter(self.exposure, self)
                self.points_done += 1

                self.data_z[j][i] += counts
            
        self.is_done = True
        #finish all data
        if self.is_stable:
            self.laser_stabilizer.stop()
        
    def stop(self):
        
        if self.is_stable:
            self.laser_stabilizer.stop()
            
        if self.is_alive():
            self.is_running = False
            self.join()
        
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()
        
        
class LiveAcquire(threading.Thread):
    """
    class for live counts measurement
    
    data_x means time/points
    
    data_y is returned data, and rolled by LiveAcquire
    
    wavelength=None as default
    
    """
    def __init__(self, exposure, data_x, data_y, config_instances, wavelength=None, is_finite=False):
        super().__init__()
        self.exposure = exposure
        self.data_x = data_x
        self.data_y = data_y
        self.counter = config_instances.get('counter')
        self.config_instances = config_instances
        
        self.wavelength = wavelength
        if self.wavelength is None:
            self.is_stable = False
        else:
            self.is_stable = True
        self.is_finite = is_finite
        
        self.daemon = True
        self.is_running = True
        self.is_done = False
        self.points_done = 0
        self.scanner = config_instances.get('scanner')
        self.info = {'data_generator':'LiveAcquire', 'exposure':self.exposure, 
                    'wavelength':self.wavelength, 'scanner':[self.scanner.x, self.scanner.y]}
        # important information to be saved with figures 
        
    
    def run(self):
        
        if self.is_stable:
            self.laser_stabilizer = LaserStabilizer(config_instances = self.config_instances)
            self.laser_stabilizer.start()
            self.laser_stabilizer.set_wavelength(self.wavelength)
            
            while self.is_running:
                time.sleep(0.01)
                if self.laser_stabilizer.is_ready:
                    break
        
        
        finite_counter = 0
        while 1 and finite_counter<len(self.data_x):

            if not self.is_running:
                return
            # break loop when interrupt

            if self.is_finite:
                finite_counter += 1
            # roll data as live counts, from left most to right most, [:] makes sure not create new arr

            counts = self.counter(self.exposure, self)
            self.points_done += 1

            self.data_y[:] = np.roll(self.data_y, 1)
            self.data_y[0] = counts
            
        self.is_done = True
        #finish all data
        if self.is_stable:
            self.laser_stabilizer.stop()
        
    def stop(self):
        
        if self.is_stable:
            self.laser_stabilizer.stop()
            
        if self.is_alive():
            self.is_running = False
            self.join()
        
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        #print('exit')
        self.stop()


class AreaAcquire(threading.Thread):
    """
    class for ple measurement (sum) for SiVs in selected area

    data_x, data_y for wavelength

    data_x_area, data_y_area for scanner

    type = 'PLE' for PLE type sum, 'PL' for PL type sum
    """
    def __init__(self, exposure, data_x, data_y, data_x_area, data_y_area, config_instances, mode = 'PLE'):
        super().__init__()
        self.exposure = exposure
        self.data_x = data_x
        self.data_y = data_y
        self.daemon = True
        self.is_running = True
        self.is_done = False
        self.counter = config_instances.get('counter')
        self.scanner = config_instances.get('scanner')
        self.config_instances = config_instances
        self.points_done = 0
        self.data_x_area = data_x_area
        self.data_y_area = data_y_area
        self.mode = mode
        self.info = {'data_generator':'AreaAcquire', 'exposure':self.exposure, 
                    'wavelength_range':self.data_x, 'scanner_range':[self.data_x_area, self.data_y_area]}
        # important information to be saved with figures 
        
    
    def run(self):
        
        
        self.laser_stabilizer = LaserStabilizer(config_instances = self.config_instances)
        self.laser_stabilizer.start()

        if self.mode == 'PLE':
        
            for i, wavelength in enumerate(self.data_x):
                self.laser_stabilizer.set_wavelength(wavelength)
                while self.is_running:
                    time.sleep(0.01)
                    if self.laser_stabilizer.is_ready:
                        break
                else:
                    return 
                
                counts = 0
                for x in self.data_x_area:
                    self.scanner.x = x
                    for y in self.data_y_area:
                        self.scanner.y = y
                        
                        counts += self.counter(self.exposure, self)
                        
                self.points_done += 1

                self.data_y[i] += counts

        elif self.mode == 'PL':
               
            for j, y in enumerate(self.data_y_area):
                self.scanner.y = y
                for i, x in enumerate(self.data_x_area):
                    self.scanner.x = x
                    counts = 0
                    for ii, wavelength in enumerate(self.data_x):
                        self.laser_stabilizer.set_wavelength(wavelength)
                        while self.is_running:
                            time.sleep(0.01)
                            if self.laser_stabilizer.is_ready:
                                break
                        else:
                            return
                    
                        counts += self.counter(self.exposure, self)
                    
                    self.points_done += 1

                    self.data_y[j][i] += counts


            
        self.is_done = True
        #finish all data
        self.laser_stabilizer.stop()
        # stop and join child thread
        
    def stop(self):
        self.laser_stabilizer.stop()
        if self.is_alive():
            self.is_running = False
            self.join()
        
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()
