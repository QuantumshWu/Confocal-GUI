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


    def off_pulse(self):


        # Create a sequence object
        sequence = self.ps.createSequence()

        # Create sequence and assign pattern to digital channel 0
        pattern_off = [(1e3, 0), (1e3, 0)]
        pattern_on = [(1e3, 1), (1e3, 1)]
        for channel in range(0, 8):
            sequence.setDigital(channel, pattern_off)# couter gate
        for channel in range(0, 2):
            sequence.setAnalog(channel, pattern_on)
        # Stream the sequence and repeat it indefinitely
        n_runs = PulseStreamer.REPEAT_INFINITELY
        self.ps.stream(sequence, n_runs)
        
    def on_pulse(self):
        
        self.off_pulse()
        time.sleep(0.5)
        
        def check_chs(array): # return a bool(0, 1) list for channels
            return array[1:]
        
        data_matrix = self.read_data()
        count = len(data_matrix)
        sequence = self.ps.createSequence()
        pattern_on = [(2, 1), (2, 1)]
        
        
        #if(count == 1):
        #    start = pb_inst_pbonly64(check_chs(data_matrix[0])+disable, Inst.CONTINUE, 0, data_matrix[0][0])
        #    pb_inst_pbonly64(check_chs(data_matrix[-1])+disable, Inst.BRANCH, start, data_matrix[-1][0])

        #else:
        #    start = pb_inst_pbonly64(check_chs(data_matrix[0])+disable, Inst.CONTINUE, 0, data_matrix[0][0])
        #    for i in range(count-2):
        #        pb_inst_pbonly64(check_chs(data_matrix[i+1])+disable, Inst.CONTINUE, 0, data_matrix[i+1][0])
         #   pb_inst_pbonly64(check_chs(data_matrix[-1])+disable, Inst.BRANCH, start, data_matrix[-1][0])

        for channel in range(0, 8):
            pattern = []
            pattern.append((data_matrix[0][0], check_chs(data_matrix[0])[channel]))
            for i in range(count-2):
                pattern.append((data_matrix[i+1][0], check_chs(data_matrix[i+1])[channel]))
            pattern.append((data_matrix[-1][0], check_chs(data_matrix[-1])[channel]))
            #print(channel, pattern)
            sequence.setDigital(channel, pattern)

        for channel in range(0, 2):
            sequence.setAnalog(channel, pattern_on)
        # Stream the sequence and repeat it indefinitely
        n_runs = PulseStreamer.REPEAT_INFINITELY
        #print(sequence.getData())
        #print('du', sequence.getDuration())
        self.ps.stream(sequence, n_runs)


        
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
        data_matrix = np.zeros((count, 9))  #skip first delay layout
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
                    
                data_matrix[i][j] = duration_num
                
            for j in range(1,9):
                item = layout.itemAt(j)
                widget = item.widget()
                if(widget.isChecked()):
                    data_matrix[i][j] = 1
        
        data_delay_matrix = self.delay(data_matrix)
        
        return data_delay_matrix
    
    def delay(self, data_matrix):
        # add delay, separate by all channels' time slices
        
        def extract_time_slice(data_matrix):
            # extract data_matrix[:,i+1]'s time slice in format [(time_i, enable_i), ...] such that enable_i for time_i-1 to time_i
            time_slice_array = [[(0, 0)] for i in range(len(data_matrix[0]) - 1)]
            cur_time = 0
            for ii, pulse in enumerate(data_matrix):
                #print(pulse)
                cur_time += pulse[0]
                for channel in range(len(data_matrix[0]) - 1):
                    time_slice_array[channel].append((cur_time, pulse[channel+1]))

            return time_slice_array
    
        def combine_time_slice(time_slice_array):
            time_all = []
            for i, time_slice in enumerate(time_slice_array):
                for i, time_label in enumerate(time_slice):
                    if(time_label[0] not in time_all):
                        time_all.append(time_label[0])

            data_matrix = np.zeros((len(time_all), len(time_slice_array)+1))
            data_matrix[:, 0] = np.sort(time_all)

            time_all = np.sort(time_all)

            for i, time_slice in enumerate(time_slice_array):
                cur_ref_index = 0 #time_slice
                cur_status = 0
                for j in range(len(time_all)):
                    cur_status = time_slice[cur_ref_index + 1][1]
                    data_matrix[j, i+1] += cur_status
                    #print(time_slice, time_all, cur_status, i, j)
                    if(time_all[j]>=time_slice[cur_ref_index + 1][0]):
                        cur_ref_index += 1

            cur = 0
            last = 0
            for pulse in data_matrix[1:]:
                cur = pulse[0]
                pulse[0] = pulse[0] - last
                last = cur
            return np.array(data_matrix[1:], dtype=int)

        def delay_channel(time_slice_array, channel_i, delay_time):
            # channel_i from 0 to n
            total_time = time_slice_array[0][-1][0] # first channel, last time stamp, time
            delay_time = delay_time%total_time
            if delay_time==0:
                return time_slice_array
            time_slice = time_slice_array[channel_i]
            time_slice_delayed = []
            is_boundary = 0
            is_delay_at_boundary = 0
            for i, time_stamp in enumerate(time_slice[1:]):# skip (0,0) since the (total_time, i) works
                if not is_boundary and (time_stamp[0]+delay_time) >= total_time:
                    is_boundary = 1
                    boundary_i = i
                time_stamp_delayed = ((time_stamp[0]+delay_time)%total_time, time_stamp[1])
                if time_stamp_delayed[0]==0:
                    is_delay_at_boundary = 1
                time_slice_delayed.append(time_stamp_delayed)

            #print(boundary_i, time_slice_delayed)
            if is_delay_at_boundary:
                time_slice_delayed = [(0,0)] + time_slice_delayed[boundary_i:][1:] + time_slice_delayed[:boundary_i] \
                                    + [(total_time, time_slice_delayed[boundary_i][1])]
            else:
                time_slice_delayed = [(0,0)] + time_slice_delayed[boundary_i:] + time_slice_delayed[:boundary_i] \
                                    + [(total_time, time_slice_delayed[boundary_i][1])]
            time_slice_array[channel_i] = time_slice_delayed

            return time_slice_array

        def delay_sequence(data_matrix, channel_i, delay_time):
            time_slice_array = extract_time_slice(data_matrix)
            time_slice_array = delay_channel(time_slice_array, channel_i, delay_time)
            data_matrix = combine_time_slice(time_slice_array)
            return data_matrix
        
        item = self.layout_dataset.itemAt(0)#first is delay
        widget = item.widget()
        layout = widget.layout()
        delay_matrix = np.zeros(8)
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

            delay_matrix[j] = duration_num
        
        for j in range(8):
            data_matrix =  delay_sequence(data_matrix, j, delay_matrix[j])
            
        return data_matrix
    

        
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