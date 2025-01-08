import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QRadioButton, QHBoxLayout, QVBoxLayout\
, QPushButton, QGroupBox, QCheckBox, QLineEdit, QComboBox, QLabel
from PyQt5.QtGui import QPalette, QColor
import numpy as np
import time
from pulsestreamer import PulseStreamer, Sequence 

    

    
class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("My App")
        
        self.widget = QWidget()
        self.layout = QVBoxLayout(self.widget)
        self.widget_button = QWidget()
        self.layout.addWidget(self.widget_button)
        self.widget_dataset = QWidget()
        self.layout.addWidget(self.widget_dataset)
        self.layout_button = QHBoxLayout(self.widget_button)
        self.layout_dataset = QHBoxLayout(self.widget_dataset)
        
        self.btn1 = QPushButton('Off Pulse')
        self.btn1.setFixedSize(150,100)
        self.btn1.clicked.connect(self.off_pulse)
        self.layout_button.addWidget(self.btn1)
        
        self.btn1 = QPushButton('On Pulse')
        self.btn1.setFixedSize(150,100)
        self.btn1.clicked.connect(self.on_pulse)
        self.layout_button.addWidget(self.btn1)
        
        self.btn2 = QPushButton('Remove Row')
        self.btn2.setFixedSize(150,100)
        self.btn2.clicked.connect(self.remove_row)
        self.layout_button.addWidget(self.btn2)
        
        self.btn3 = QPushButton('Add Row')
        self.btn3.setFixedSize(150,100)
        self.btn3.clicked.connect(self.add_row)
        self.layout_button.addWidget(self.btn3)
        
        self.setCentralWidget(self.widget)
        self.add_delay()
        self.add_row()
        self.add_row()
        self.add_row()

        ip = '169.254.8.2'
        self.ps = PulseStreamer(ip)
        self.PulseStreamer = PulseStreamer
        self.Sequence = Sequence
        self.delay_array = None


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


        
    def print_index(self):
        count = self.layout_dataset.count()
        for i in range(count):
            item = self.layout_dataset.itemAt(i)
            widget = item.widget()
            layout = widget.layout()
            for j in range(1):
                item_sub = layout.itemAt(j)
                layout_sub = item_sub.layout()
                duration_num = int(layout_sub.itemAt(1).widget().text())
                duration_unit = layout_sub.itemAt(2).widget().currentText()
                print(duration_num, duration_unit)
            for j in range(1,5):
                item = layout.itemAt(j)
                widget = item.widget()
                if(widget.isChecked()):
                    print((i,j))
                    
    def read_data(self):
        count = self.layout_dataset.count()-1  #number of pulses 
        data_matrix = [[0]*9 for _ in range(count)] #skip first delay layout

        for i in range(count):
            item = self.layout_dataset.itemAt(i+1)#first is delay
            widget = item.widget()
            layout = widget.layout()
            for j in range(1):
                item_sub = layout.itemAt(j)
                layout_sub = item_sub.layout()
                duration_num = int(layout_sub.itemAt(1).widget().text())
                duration_unit = layout_sub.itemAt(2).widget().currentText()
                
                if(duration_unit == 'ns'):
                    duration_num *= 1
                elif(duration_unit == 'us'):
                    duration_num *= 1000
                elif(duration_unit == 'ms'):
                    duration_num *= 1000000
                    
                data_matrix[i][j] = int(duration_num)
                
            for j in range(1,9):
                item = layout.itemAt(j)
                widget = item.widget()
                if(widget.isChecked()):
                    data_matrix[i][j] = 1
        
        self.read_delay()
        time_slices = []
        for channel in range(8):
            time_slice = [[period[0], period[channel+1]] for period in data_matrix]
            time_slice_delayed = self.delay(self.delay_array[channel], time_slice)
            time_slices.append(time_slice_delayed)
        
        return time_slices

    def read_delay(self):

        item = self.layout_dataset.itemAt(0)#first is delay
        widget = item.widget()
        layout = widget.layout()
        delay_array = [0, 0, 0, 0, 0, 0, 0, 0]
        for j in range(8):
            item_sub = layout.itemAt(j)
            layout_sub = item_sub.layout()
            duration_num = int(layout_sub.itemAt(1).widget().text())
            duration_unit = layout_sub.itemAt(2).widget().currentText()
            
            if(duration_unit == 'ns'):
                duration_num *= 1
            elif(duration_unit == 'us'):
                duration_num *= 1000
            elif(duration_unit == 'ms'):
                duration_num *= 1000000
                
            delay_array[j] = int(duration_num)

        self.delay_array = delay_array
    
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
    

        
    def remove_row(self):
        count = self.layout_dataset.count()
        #print(count)
        if(count>=3):
            item = self.layout_dataset.itemAt(count-1)
            widget = item.widget()
            widget.deleteLater()
            
    def add_row(self):
        count = self.layout_dataset.count()
        row = QGroupBox('Pulse%d'%(count))
        self.layout_dataset.addWidget(row)
        layout_data = QVBoxLayout(row)
        
        sublayout = QHBoxLayout()
        layout_data.addLayout(sublayout)
        btn = QLabel('duration:')
        sublayout.addWidget(btn)
        btn = QLineEdit('1')
        sublayout.addWidget(btn)
        btn = QComboBox()
        btn.addItems(['ns','us' , 'ms'])
        sublayout.addWidget(btn)
        
        for index in range(1, 9):
            btn = QCheckBox()
            btn.setText('ch%d'%(index-1))
            btn.setCheckable(True)
            layout_data.addWidget(btn)
        
        
    def add_delay(self):
        count = self.layout_dataset.count()
        row = QGroupBox('Delay')
        self.layout_dataset.addWidget(row)
        layout_data = QVBoxLayout(row)
        
        for index in range(1, 9):
            sublayout = QHBoxLayout()
            layout_data.addLayout(sublayout)
            btn = QLabel('ch%d delay:'%(index-1))
            sublayout.addWidget(btn)
            btn = QLineEdit('0')
            sublayout.addWidget(btn)
            btn = QComboBox()
            btn.addItems(['ns','us' , 'ms'])
            sublayout.addWidget(btn)
 
        

        
if __name__ == "__main__":

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    app.exec()