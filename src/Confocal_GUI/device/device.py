from abc import ABC, abstractmethod
import time
import sys, os
current_directory = os.path.dirname(os.path.abspath(__file__))
# location of new focus laser driver file
sys.path.append(current_directory)
import numpy as np
import threading
from .base import *
from Confocal_GUI.gui import *
                
    
class TLB6700(BaseLaser, metaclass=SingletonAndCloseMeta):
    """
    laser = TLB6700()
    
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

        import clr
        from System.Text import StringBuilder
        from System import Int32
        from System.Reflection import Assembly
        import Newport

        clr.AddReference(r'mscorlib')
        sys.path.append('C:\\Program Files\\New Focus\\New Focus Tunable Laser Application\\')
        # location of new focus laser driver file
        clr.AddReference('UsbDllWrap')

        self.tlb = Newport.USBComm.USB()
        self.answer = StringBuilder(64)

        self.ProductID = 4106
        self.DeviceKey = '6700 SN37711'
        
        self.model = model
        self._wavelength = None
        self._piezo = None

        self.connect()
        
        self.piezo_min = 0
        self.piezo_max = 100

    def gui(self):
        """
        Use self.gui_property and self.gui_property_type to determine how to display configurable parameters
        """
        self.gui_property = ['wavelength', 'piezo']
        self.gui_property_type = ['float', 'float']
        GUI_Device(self)
        
    def connect(self):
        self.__tlb_open()
        
    def close(self):
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


# need rewrite !!!
class __LaserStabilizerCore(BaseLaserStabilizer, metaclass=SingletonAndCloseMeta):
    """
    core logic for stabilizer,
    
    .run() will read wavemeter and change laser piezo
    .is_ready=True when wavelength_desired = wavelength_actual
    .wavelength is the wavelength desired
    .ratio = -0.85GHz/V defines the ratio for feedback
    """
    
    def __init__(self, ratio=-0.85):
        self.ratio = ratio
        self.laser = config_instances.get('laser')
        self.spl = 299792458
        self.v_mid = 0.5*(self.laser.piezo_max + self.laser.piezo_min)
        self.v_min = self.laser.piezo_min + 0.05*(self.laser.piezo_max - self.laser.piezo_min)
        self.v_max = self.laser.piezo_min + 0.95*(self.laser.piezo_max - self.laser.piezo_min)
        self.freq_thre = 0.05 #50MHz threshold defines when to return is_ready
        self.P = 0.8 #scaling factor of PID control
        # leaves about 10% extra space
        super().__init__()

    def gui(self):
        """
        Use self.gui_property and self.gui_property_type to determine how to display configurable parameters
        """
        self.gui_property = ['on', 'wavelength']
        self.gui_property_type = ['str', 'float']
        GUI_Device(self)
        
    def _run(self):
        
        freq_desired = self.spl/self.wavelength
        wave_cache = self.wavemeter.wavelength
        freq_diff_guess = freq_desired - self.spl/wave_cache#freq_desired - self.freq_recent
        v_diff = self.P*freq_diff_guess/self.ratio 
        v_0 = self.laser.piezo
        #print(f'read wave {wave_cache}')
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
            #print(f'move wave')
        else:
            self.laser.piezo = v_0+v_diff
            #print(f'move piezo')
        #print(f'piezo {self.laser.piezo}, v_diff: {v_diff}')
        time.sleep(0.2)# wait for piezo stable and measurement converge
        freq_actual = self.spl/self.wavemeter.wavelength #wait
        #print(f'wave actual {self.wavemeter.wavelength}')
        freq_diff = freq_desired - freq_actual
        if np.abs(freq_diff) <= self.freq_thre:
            self.is_ready = True
        else:
            self.is_ready = False
        #print(f'recent:{freq_actual}, actual:{self.spl/self.wavemeter.wavelength}, ')
        return

        
class WaveMeter671(BaseWavemeter, metaclass=SingletonAndCloseMeta):
    """
    Control code for 671 Wavelength Meter
    
    wavemeter671 = WaveMeter671()
    
    >>> wavemeter671.wavelength
    >>> 737.105033
    # read wavelength from wavemeter
    
    """
    

    def __init__(self, ip=None):
        import telnetlib
        if ip is None:
            ip = '10.199.199.1'
        self.HOST = ip
        self.tn = telnetlib.Telnet(self.HOST, timeout=1)
        self.tn.write(b'*IDN?\r\n')
        time.sleep(0.5)
        self.tn.read_very_eager()
        self.lock = threading.Lock()
        
    
    @property
    def wavelength(self):
        with self.lock:
            self.tn.write(b':READ:WAV?\r\n')
            data = self.tn.expect([b'\r\n'])[-1]
            self._wavelength = float(data.decode('utf-8')[:-2])
            return self._wavelength
    
    def connect(self):
        self.tn = telnetlib.Telnet(self.HOST, timeout=1)
        self.tn.write(b'*IDN?\r\n')
        time.sleep(0.5)
        self.tn.read_very_eager()
    
    def disconnect(self):
        self.tn.close()

    def close(self):
        self.disconnect()


class SGCounter(BaseCounter):
    """
    Software gated counter, using time.sleep, therefore
    duration stability may not be sufficienct for some cases
    """

    def __init__(self, apd_signal='/Dev2/ctr1', apd_gate='/Dev2/PFI4'):
        import nidaqmx
        self.apd_signal = apd_signal
        self.apd_gate = apd_gate
        self.nidaqmx = nidaqmx

    @property
    def valid_counter_mode(self):
        return ['apd']

    @property
    def valid_data_mode(self):
        return ['single']

    def read_counts(self, exposure, counter_mode='apd', data_mode='single',**kwargs):
        """
        software gated counter for USB-6211, and reset pulse every time
        """
        with self.nidaqmx.Task() as task:
            task.ci_channels.add_ci_count_edges_chan(self.apd_signal)
            task.triggers.pause_trigger.dig_lvl_src = self.apd_gate
            task.triggers.pause_trigger.trig_type = self.nidaqmx.constants.TriggerType.DIGITAL_LEVEL
            task.triggers.pause_trigger.dig_lvl_when = self.nidaqmx.constants.Level.LOW
            task.start()
            time.sleep(exposure)
            data_counts = task.read()
            task.stop()
        return [data_counts, ]


class HighFiness(BaseWavemeter):
    """
    Class for HighFiness wavemeter
    
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
    

class Keysight(BaseRF):
    """
    class for Keysight RF generator
    
    amp in V-pp
    frequency for frequency of sine wave
    """
    
    def __init__(self, visa_str = 'TCPIP0::localhost::hislip0::INSTR'):
        rm = pyvisa.ResourceManager()
        self.handle = rm.open_resource(visa_str)
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
    def on(self, value):
        self._on = value
        if value is True:
            # from False to True
            self.handle.write(':INIT:IMM')
        else:
            # from True to False
            self.handle.write(':ABOR')
        
    @property
    def power(self):
        return self._power
    
    @power.setter
    def power(self, value):
        self._power = value
        self.handle.write(f'VOLT {self._power}')
    
    @property
    def frequency(self):
        return self._frequency
    
    @frequency.setter
    def frequency(self, value):
        self.handle.write(':SOUR:FREQ:RAST 64000000000') # set sampling rate to 64G/s
        self._frequency = value
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
        


class TimeTaggerCounter(BaseCounter):
    """
    Uses timetagger as daq to count counts (edge counting) from APDs
    """

    def __init__(self, click_channel, begin_channel, end_channel, n_values=int(1e6)):
        from TimeTagger import createTimeTagger, Histogram, CountBetweenMarkers
        tagger = createTimeTagger('1809000LGG')
        tagger.reset()

        self.counter_handle = CountBetweenMarkers(tagger=tagger, click_channel=click_channel, begin_channel=begin_channel, \
            end_channel=end_channel, n_values=n_values)

    @property
    def valid_counter_mode(self):
        return ['apd']

    @property
    def valid_data_mode(self):
        return ['single']

    def read_counts(self, exposure, counter_mode='apd', data_mode='single', **kwargs):
        self.counter_handle.startFor(int(exposure*1e12))
        time.sleep(exposure)
        counts = np.sum(self.counter_handle.getData())
        self.counter_handle.stop()
        return counts
        
        
class SGS100A(BaseRF, metaclass=SingletonAndCloseMeta):
    """
    Class for RF generator SGS100A
    
    power in dbm
    
    frequency for frequency
    
    iq for if iq is on
    
    on for if output is on
    """
    
    def __init__(self, visa_str = 'TCPIP::169.254.2.20::INSTR'):
        import pyvisa
        rm = pyvisa.ResourceManager()
        self.handle = rm.open_resource(visa_str)
        self._power = int(self.handle.query('SOURce:Power?')[:-1])
        self._frequency = int(self.handle.query('SOURce:FREQuency?')[:-1])
        self._iq = False # if IQ modulation is on
        self._on = False # if output is on

    def gui(self):
        """
        Use self.gui_property and self.gui_property_type to determine how to display configurable parameters
        """
        self.gui_property = ['on', 'power', 'frequency', 'iq']
        self.gui_property_type = ['str', 'float', 'float', 'str']
        GUI_Device(self)
        
    @property
    def power(self):
        self._power = int(self.handle.query('SOURce:Power?')[:-1])
        return self._power
    
    @power.setter
    def power(self, value):
        self._power = value
        self.handle.write(f'SOURce:Power {self._power}')
    
    @property
    def frequency(self):
        self._frequency = int(self.handle.query('SOURce:Frequency?')[:-1])
        return self._frequency
    
    @frequency.setter
    def frequency(self, value):
        self._frequency = value
        self.handle.write(f'SOURce:Frequency {self._frequency}')
        
        
    @property
    def on(self):
        return self._on
    
    @on.setter
    def on(self, value):
        #if on_in is self._on:
        #    return
        self._on = value
        if value is True:
            # from False to True
            self.handle.write('OUTPut:STATe ON')
        else:
            # from True to False
            self.handle.write('OUTPut:STATe OFF')
    
    @property
    def iq(self):
        return self._iq
    
    @iq.setter
    def iq(self, value):
        if value is self._iq:
            return
        
        self._iq = value
        if value is True:
            # iq from False to True
            self.handle.write('SOURce:IQ:IMPairment:STATe ON')
            self.handle.write('SOURce:IQ:STATe ON')
        else:
            # iq from True to False
            self.handle.write('SOURce:IQ:IMPairment:STATe OFF')
            self.handle.write('SOURce:IQ:STATe OFF')

    def close(self):
        pass


class DSG836(BaseRF, metaclass=SingletonAndCloseMeta):
    """
    Class for RF generator DSG836
    
    power in dbm
    
    frequency for frequency
    
    on for if output is on
    """
    
    def __init__(self, visa_str='USB0::0x1AB1::0x099C::DSG8M223900103::INSTR', power_ub=-5, power_lb=None):
        import pyvisa
        rm = pyvisa.ResourceManager()
        self.handle = rm.open_resource(visa_str)
        self._power = eval(self.handle.query('SOURce:Power?')[:-1])
        self._frequency = eval(self.handle.query('SOURce:FREQuency?')[:-1])
        self._iq = False # if IQ modulation is on
        self._on = False # if output is on
        self.power_ub = power_ub
        self.power_lb = power_lb

    def gui(self):
        """
        Use self.gui_property and self.gui_property_type to determine how to display configurable parameters
        """
        self.gui_property = ['on', 'power', 'frequency']
        self.gui_property_type = ['str', 'float', 'float']
        GUI_Device(self)
        
    @property
    def power(self):
        self._power = eval(self.handle.query('SOURce:Power?')[:-1])
        return self._power
    
    @power.setter
    def power(self, value):
        if value > self.power_ub:
            value = self.power_ub
            print(f'can not exceed RF power {self.power_ub}dbm')
        self._power = value
        self.handle.write(f'SOURce:Power {self._power}')
    
    @property
    def frequency(self):
        self._frequency = eval(self.handle.query('SOURce:Frequency?')[:-1])
        return self._frequency
    
    @frequency.setter
    def frequency(self, value):
        self._frequency = value
        self.handle.write(f'SOURce:Frequency {self._frequency}')
        
        
    @property
    def on(self):
        return self._on
    
    @on.setter
    def on(self, value):
        #if on_in is self._on:
        #    return
        self._on = value
        if value is True:
            # from False to True
            self.handle.write('OUTPut:STATe ON')
        else:
            # from True to False
            self.handle.write('OUTPut:STATe OFF')

    def close(self):
        pass
            


class AFG3052C(BaseScanner):
    """
    class for scanner AFG3052C
    
    example:
    >>> afg3052c = AFG3052C()
    >>> afg3052c.x
    10
    >>> afg3052c.x = 20
    
    """
    def __init__(self, visa_str = 'GPIB2::11::INSTR'):    
        import pyvisa   
        rm = pyvisa.ResourceManager()
        self.handle = rm.open_resource(visa_str)

    def gui(self):
        """
        Use self.gui_property and self.gui_property_type to determine how to display configurable parameters
        """
        self.gui_property = ['x', 'y']
        self.gui_property_type = ['x', 'y']
        GUI_Device(self)
    
    @property
    def x(self):
        result_str = self.handle.query('SOURce1:VOLTage:LEVel:IMMediate:OFFSet?')
        self._x = int(1000*eval(result_str[:-1]))
        return self._x
    
    @x.setter
    def x(self, value):
        self.handle.write(f'SOURce1:VOLTage:LEVel:IMMediate:OFFSet {value}mV')
        result_str = self.handle.query('SOURce1:VOLTage:LEVel:IMMediate:OFFSet?')
        self._x = int(1000*eval(result_str[:-1]))
        
    @property
    def y(self):
        result_str = self.handle.query('SOURce2:VOLTage:LEVel:IMMediate:OFFSet?')
        self._y = int(1000*eval(result_str[:-1]))
        return self._y
    
    @y.setter
    def y(self, value):
        self.handle.write(f'SOURce2:VOLTage:LEVel:IMMediate:OFFSet {value}mV')
        result_str = self.handle.query('SOURce2:VOLTage:LEVel:IMMediate:OFFSet?')
        self._y = int(1000*eval(result_str[:-1]))


class AMI():

    """
    Class for AMI magnet control for diluation fridge
    """
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




class Pulse(BasePulse):
    """
    Pulse class to pulse streamer control,
    call pulse.gui() to access to all functions and pulse sequence editing

    notes:
    pulse duration, delay can be a int in ns or str contains 'x' which will be later replaced by self.x,
    therefore enabling fast pulse control by self.x = x and self.on_pulse() without reconfiguring self.data_matrix
    and sel.delay_array

    Save Pulse: save pulse sequence to self.data_matrix, self.delay_array but not in file
    Save to file: save pulse sequence to a .npz file


    """

    def __init__(self, ip=None):
        super().__init__()
        from pulsestreamer import PulseStreamer, Sequence 
        self.PulseStreamer = PulseStreamer
        self.Sequence = Sequence
        if ip is None:
            self.ip = '169.254.8.2'
            # default ip address of pulse streamer
        else:
            self.ip = ip
        self.ps = PulseStreamer(self.ip)


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





    




