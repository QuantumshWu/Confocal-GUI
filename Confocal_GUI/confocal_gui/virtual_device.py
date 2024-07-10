import numpy as np
import time
        

        
    
class VirtualLaser():
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
      
    
    def __init__(self):
        
        pass
        
    
        
class VirtualWaveMeter():
    """
    Control code for 671 Wavelength Meter
    
    wavemeter671 = WaveMeter671()
    
    >>> wavemeter671.wavelength
    >>> 737.105033
    # read wavelength from wavemeter
    
    """
    

    def __init__(self):
        pass


        
def virtual_read_counts(duration, parent):
    """
    simulated counter for test
    """
    if parent is None:
        time.sleep(duration)
        return np.random.poisson(duration*10000)
    
    _class = parent.__class__.__name__
    if _class == 'PLEAcquire':
        
        ple_dict = {'ple_height':None, 'ple_width':None, 'ple_center':None}
        for key in ple_dict.keys():
            ple_dict[key] = parent.config_instances.get(key)
        
        time.sleep(duration)
        wavelength = parent.laser_stabilizer.desired_wavelength
        lambda_counts = parent.exposure*ple_dict['ple_height']*(ple_dict['ple_width']/2)**2\
                                 /((wavelength-ple_dict['ple_center'])**2 + (ple_dict['ple_width']/2)**2)
        return np.random.poisson(lambda_counts)
    
    elif _class == 'PLAcquire':
        
        pl_dict = {'pl_height':None, 'pl_width':None, 'pl_center':None, 'pl_bg':None}
        for key in pl_dict.keys():
            pl_dict[key] = parent.config_instances.get(key)
        
        time.sleep(duration)
        position = np.array([parent.scanner.x, parent.scanner.y])
        distance = np.linalg.norm(np.array(pl_dict['pl_center']) - position)
        
        lambda_counts = parent.exposure*(pl_dict['pl_height']*(pl_dict['pl_width']/2)**2\
                                 /((distance)**2 + (pl_dict['pl_width']/2)**2) + pl_dict['pl_bg'])
        return np.random.poisson(lambda_counts)
        
        
    else: # None of these cases
        pl_dict = {'pl_height':None, 'pl_width':None, 'pl_center':None, 'pl_bg':None}
        for key in pl_dict.keys():
            pl_dict[key] = parent.config_instances.get(key)
        
        time.sleep(duration)
        position = np.array([parent.scanner.x, parent.scanner.y])
        distance = np.linalg.norm(np.array(pl_dict['pl_center']) - position)
        
        lambda_counts = parent.exposure*(pl_dict['pl_height']*(pl_dict['pl_width']/2)**2\
                                 /((distance)**2 + (pl_dict['pl_width']/2)**2) + pl_dict['pl_bg'])
        return np.random.poisson(lambda_counts)
        #time.sleep(duration)
        #return np.random.poisson(duration*10000)


class VirtualScanner():
    """
    class for scanner AFG3052C
    
    example:
    >>> afg3052c = AFG3052C()
    >>> afg3052c.x
    10
    >>> afg3052c.x = 20
    
    """
    def __init__(self):
        self.x = 0
        self.y = 0
    
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, x_in):
        self._x = x_in
        
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, y_in):
        self._y = y_in
    


