import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import matplotlib.ticker as mticker
from IPython import get_ipython
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.widgets import RectangleSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import threading
from decimal import Decimal
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QHBoxLayout, QWidget, QPushButton, QLabel, QLineEdit, QDoubleSpinBox
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.figure import Figure
from threading import Event
import io
from PIL import Image as PILImage
from IPython.display import display, Image as IPImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget
from PyQt5 import uic
from scipy.optimize import curve_fit
import glob
from scipy.signal import find_peaks
import os

from .logic import *




_new_black = '#373737'
params_inline = {
    'axes.labelsize': 13,
    'legend.fontsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'figure.figsize': [3.5, 2.625],
    'lines.linewidth': 2,
    'scatter.edgecolors': 'black',
    'legend.numpoints': 2,
    'lines.markersize': 4,
    'ytick.major.size': 6,  # major tick size in points
    'ytick.major.width': 0.8,  # major tick width in points
    'xtick.major.size': 6,  # major tick size in points
    'xtick.major.width': 0.8,  # major tick width in points
    'axes.linewidth': 0.8,
    'figure.subplot.left': 0,
    'figure.subplot.right': 1,
    'figure.subplot.bottom': 0,
    'figure.subplot.top': 1,
    'grid.linestyle': '--',
    'axes.grid': False,
    'text.usetex': False,
    'font.family' : 'sans-serif',
    'font.sans-serif' : 'Arial',
    'axes.unicode_minus' : False,
    'text.latex.preamble' : r'\usepackage{libertine} \usepackage[libertine]{newtxmath} \usepackage{sfmath}',
    "xtick.direction": "in",
    "ytick.direction": "in",
    'legend.frameon': False,
    'savefig.bbox' : 'tight',
    'savefig.pad_inches' : 0.05,
    'figure.dpi' : 150,
    
    'text.color': _new_black,
    'patch.edgecolor': _new_black,
    'patch.force_edgecolor': False, # Seaborn turns on edgecolors for histograms by default and I don't like it
    'hatch.color': _new_black,
    'axes.edgecolor': _new_black,
    'axes.titlecolor': _new_black, # should fallback to text.color
    'axes.labelcolor': _new_black,
    'xtick.color': _new_black,
    'ytick.color': _new_black,

}


params_nbagg = {
    'axes.labelsize': 13,
    'legend.fontsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'figure.figsize': [3.5*1.2, 2.625*1.2],
    'lines.linewidth': 2,
    'scatter.edgecolors':  _new_black,
    'legend.numpoints': 2,
    'lines.markersize': 4,
    'ytick.major.size': 6,  # major tick size in points
    'ytick.major.width': 0.8,  # major tick width in points
    'xtick.major.size': 6,  # major tick size in points
    'xtick.major.width': 0.8,  # major tick width in points
    'axes.linewidth': 0.8,
    'figure.subplot.left': 0,
    'figure.subplot.right': 1,
    'figure.subplot.bottom': 0,
    'figure.subplot.top': 1,
    'grid.linestyle': '--',
    'axes.grid': False,
    'text.usetex': False,
    'font.family' : 'sans-serif',
    'font.sans-serif' : 'Arial',
    'text.latex.preamble' : r'\usepackage{libertine} \usepackage[libertine]{newtxmath} \usepackage{sfmath}',
    "xtick.direction": "in",
    "ytick.direction": "in",
    'legend.frameon': False,
    'savefig.bbox' : 'tight',
    'savefig.pad_inches' : 0.05,
    'figure.dpi' : 150,
    
    'text.color': _new_black,
    'patch.edgecolor': _new_black,
    'patch.force_edgecolor': False, # Seaborn turns on edgecolors for histograms by default and I don't like it
    'hatch.color': _new_black,
    'axes.edgecolor': _new_black,
    'axes.titlecolor': _new_black, # should fallback to text.color
    'axes.labelcolor': _new_black,
    'xtick.color': _new_black,
    'ytick.color': _new_black,

}

def scale_params(params, scale):
    scaled_params = params.copy()
    scaled_params['figure.figsize'] = [dim * scale for dim in scaled_params['figure.figsize']]
    for key in ['axes.labelsize', 'legend.fontsize', 'xtick.labelsize', 
                'ytick.labelsize', 'lines.linewidth', 'lines.markersize', 
                'ytick.major.size', 'ytick.major.width', 'xtick.major.size', 
                'xtick.major.width', 'axes.linewidth']:
        if key in scaled_params:
            scaled_params[key] = scaled_params[key] * scale
    return scaled_params


def change_to_inline(params_type, scale):
    get_ipython().run_line_magic('matplotlib', 'inline')
    if params_type == 'inline':
        scaled_params = scale_params(params_inline, scale)
        matplotlib.rcParams.update(scaled_params)
    elif params_type == 'nbagg':
        scaled_params = scale_params(params_nbagg, scale)
        matplotlib.rcParams.update(scaled_params)
    else:
        print('wrong params_type')

def change_to_nbagg(params_type, scale):
    get_ipython().run_line_magic('matplotlib', 'widget')
    if params_type == 'inline':
        scaled_params = scale_params(params_inline, scale)
        matplotlib.rcParams.update(scaled_params)
    elif params_type == 'nbagg':
        scaled_params = scale_params(params_nbagg, scale)
        matplotlib.rcParams.update(scaled_params)
    else:
        print('wrong params_type')

def hide_elements():
    css_code = """
    <style>
    .output_wrapper button.btn.btn-default,
    .output_wrapper .ui-dialog-titlebar{
      display: none;
    }
    </style>
    """
    display(HTML(css_code))

def enable_long_output():
    css_code = """
    <style>
    .output_scroll {
        height: auto !important;
        max-height: none !important;
    }
    </style>
    """
    display(HTML(css_code))

def display_immediately(fig):
    # https://github.com/matplotlib/ipympl/issues/290
    # a fix for widget backend but is not needed for nbagg
    canvas = fig.canvas
    display(canvas)
    canvas._handle_message(canvas, {'type': 'send_image_mode'}, [])
    canvas._handle_message(canvas, {'type':'refresh'}, [])
    canvas._handle_message(canvas,{'type': 'initialized'},[])
    canvas._handle_message(canvas,{'type': 'draw'},[])

        
        
class LivePlotGUI():
    """
    if mode is 'PLE' or 'Live'
    labels = [xlabel, ylabel]
    data = [data_x, data_y]
    
    if mode is 'PL'
    labels = [xlabel, ylabel, zlabel]
    data = [data_x, data_y, data_z] # data_z n*m array, data_z[x, y] has coordinates (x, y)
    """
    
    def __init__(self, labels, update_time, data_generator, data, fig=None, config_instances=None, relim_mode='normal'):

        self.labels = labels
        
        if len(data) == 3: #PL
            self.xlabel = labels[0]
            self.ylabel = labels[1]
            self.zlabel = labels[2]
            self.data_x = data[0]
            self.data_y = data[1]
            self.data_z = data[2]
            self.points_total = len(self.data_z.flatten())
            
        else:# 'PLE' or 'Live'
            self.xlabel = labels[0]
            self.ylabel = labels[1] + ' x1'
            self.data_x = data[0]
            self.data_y = data[1]
            self.points_total = len(self.data_y.flatten())
        
        self.update_time = update_time
        self.data_generator = data_generator
        self.ylim_max = 100
        self.ylim_min = 0
        self.fig = fig
        if fig is None:
            self.have_init_fig = False
        else:
            self.have_init_fig = True

        self.points_done = 0
        # track how many data points have done
        self.selector = None
        # assign value by self.choose_selector()
        if config_instances is None:
            self.config_instances = {'display_scale':1}
        else:
            self.config_instances = config_instances
        self.scale = self.config_instances['display_scale']
        self.relim_mode = relim_mode
        
    def init_figure_and_data(self):
        change_to_nbagg(params_type = 'nbagg', scale=self.scale)
        #hide_elements()
        plt.ion()
        # make sure environment enables interactive then updating figure
        
        if self.fig is None:
            with plt.ioff():
                # avoid double display from display_immediately
                self.fig = plt.figure()

            self.fig.canvas.toolbar_visible = False
            self.fig.canvas.header_visible = False
            self.fig.canvas.footer_visible = False
            self.fig.canvas.resizable = False
            self.fig.canvas.capture_scroll = True
            display_immediately(self.fig)
            self.axes = self.fig.add_subplot(111)
        else:
            self.axes = self.fig.axes[0]
            
        self.clear_all() #makes sure no residual artist
        self.axes.set_autoscale_on(True)
        self.init_core()         

        
        
        #formatter = mticker.ScalarFormatter(useMathText=True)
        #self.axes.xaxis.set_major_formatter(formatter)
        #self.axes.xaxis.offsetText.set_visible(False)
        #ticks_offset = self.axes.xaxis.get_major_formatter().get_offset()
        # code to move offset of ticks to label

        self.axes.set_ylabel(self.ylabel)
        self.axes.set_xlabel(self.xlabel)
        
        if not self.have_init_fig:
            self.fig.tight_layout()
        self.fig.canvas.draw()
        self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # store bg

        
        
        self.data_generator.start()
        
    def update_figure(self):
        self.update_core()
            

        self.axes.draw_artist(self.line)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()
        self.points_done = self.data_generator.points_done
        
        
    def init_core(self):
        raise NotImplementedError("Subclasses should implement this method.")
        
    def update_core(self):
        raise NotImplementedError("Subclasses should implement this method.")
        
    def choose_selector(self):
        raise NotImplementedError("Subclasses should implement this method.")
        
    def plot(self):
        self.init_figure_and_data()
        
        try:
            while not self.data_generator.is_done:
                if (self.data_generator.points_done == self.points_done):
                    # if no new data then no update
                    continue
                self.update_figure()
                time.sleep(self.update_time)
            else:
                self.update_figure()
                
        except BaseException as e:
            print(e)
            pass
        
        self.data_generator.stop()
        
        
        self.line.set_animated(False)        
        self.axes.set_autoscale_on(False)
        self.choose_selector()

        return self.fig, self.selector

    
    def stop(self):
        if self.data_generator.is_alive():
            self.data_generator.stop()
            
    def clear_all(self):
        # code to remove selectors' plot
        for ax in self.fig.axes:
            lines_to_remove = [line for line in ax.lines]
            patches_to_remove = [patch for patch in ax.patches]
            texts_to_remove = [text for text in ax.texts]


            for line in lines_to_remove:
                line.remove()
            for patch in patches_to_remove:
                patch.remove()
            for text in texts_to_remove:
                text.remove()

        self.fig.canvas.draw()


    def relim(self):
        # return 1 if need redraw
        # accept relim mode 'tight' or 'normal'
        # 'tight' will relim to fit upper and lower bound
        # 'normal' will relim to fit 0 and upper bound
        
        if 0<self.points_done<self.points_total:
            max_data_y = np.max(self.data_y[:self.points_done])
            min_data_y = np.min(self.data_y[:self.points_done])

        else:
            max_data_y = np.max(self.data_y)
            min_data_y = np.min(self.data_y)

        if max_data_y == 0:
            return 0
        # no new data

        if self.relim_mode == 'normal':
            data_range = max_data_y - 0
        elif self.relim_mode == 'tight':
            data_range = max_data_y - min_data_y

        if self.relim_mode == 'normal':

            if 0<=(self.ylim_max-max_data_y)<=0.5*data_range:
                return 0


            self.ylim_min = 0
            self.ylim_max = max_data_y*1.2

            self.axes.set_ylim(self.ylim_min, self.ylim_max)

            return 1

        elif self.relim_mode == 'tight':

            if 0<=(self.ylim_max - max_data_y)<=0.25*data_range and 0<=(min_data_y - self.ylim_min)<=0.25*data_range:
                return 0

            self.ylim_min = min_data_y - 0.1*data_range
            self.ylim_max = max_data_y + 0.1*data_range

            if self.ylim_min!=self.ylim_max:
                self.axes.set_ylim(self.ylim_min, self.ylim_max)

            return 1




            
            
class PLELive(LivePlotGUI):
    
    def init_core(self):
        self.line, = self.axes.plot(self.data_x, self.data_y, animated=True, color='grey', alpha=1)
        self.axes.set_xlim(np.min(self.data_x), np.max(self.data_x))
        self.axes.set_ylim(self.ylim_min, self.ylim_max)
        
    def update_core(self):
        if len(self.data_y) == 0:
            max_data_y = 0
        else:
            max_data_y = np.max(self.data_y)
        

        if (self.points_done%self.points_total)==0:

            self.ylabel = self.labels[1] + f' x{self.points_done//self.points_total + 1}'
            self.axes.set_ylabel(self.ylabel)


            is_redraw = self.relim()
            self.fig.canvas.draw()
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # update ylim so need redraw
        else:

            is_redraw = self.relim()
            if is_redraw:
                self.fig.canvas.draw()
                self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # update ylim so need redraw

            
        self.fig.canvas.restore_region(self.bg_fig)


        self.line.set_data(self.data_x[:self.data_generator.points_done], self.data_y[:self.data_generator.points_done])
        
    def choose_selector(self):
        self.area = AreaSelector(self.fig.axes[0])
        self.cross = CrossSelector(self.fig.axes[0])
        self.zoom = ZoomPan(self.fig.axes[0])
        
        self.selector = [self.area, self.cross, self.zoom]
        

class PLLive(LivePlotGUI):
    
    def init_core(self):
        
        cmap_ = matplotlib.cm.get_cmap('inferno')
        cmap = cmap_.copy()
        cmap.set_under(cmap_(0))
        half_step_x = 0.5*(self.data_x[-1] - self.data_x[0])/len(self.data_x)
        half_step_y = 0.5*(self.data_y[-1] - self.data_y[0])/len(self.data_y)
        extents = [self.data_x[0]-half_step_x, self.data_x[-1]+half_step_x, \
                   self.data_y[-1]+half_step_y, self.data_y[0]-half_step_y] #left, right, bottom, up
        self.line = self.axes.imshow(self.data_z, animated=True, alpha=1, cmap=cmap, extent=extents)
        cbar = self.fig.colorbar(self.line)
        cbar.set_label(self.zlabel)

        self.vmin = 0
        self.vmax = 0
            
        
    def update_core(self):
        
        if np.max(self.data_z)==0:
            vmin = 0
        else:
            vmin = np.min(self.data_z[self.data_z!=0])

        vmax = np.max(self.data_z)

        if vmax == vmin:
                vmax = vmin + 1

        if vmin*0.8 < self.vmin or self.vmax < vmax*1.2:
            self.vmin = vmin*0.8
            self.vmax = vmax*1.2
        # filter out zero data

            self.line.set_clim(vmin=vmin, vmax=np.max(self.data_z))
            self.fig.canvas.draw()
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)


        self.fig.canvas.restore_region(self.bg_fig)
        self.line.set_array(self.data_z)
        
    def choose_selector(self):
        self.area = AreaSelector(self.fig.axes[0])
        self.cross = CrossSelector(self.fig.axes[0])
        self.zoom = ZoomPan(self.fig.axes[0])
        
        self.selector = [self.area, self.cross, self.zoom]

class PLDisLive(LivePlotGUI):

    def init_core(self):
        width, height = self.fig.get_size_inches()
        self.fig.set_size_inches(width, height*1.25)
        
        cmap_ = matplotlib.cm.get_cmap('inferno')
        cmap = cmap_.copy()
        cmap.set_under(cmap_(0))
        half_step_x = 0.5*(self.data_x[-1] - self.data_x[0])/len(self.data_x)
        half_step_y = 0.5*(self.data_y[-1] - self.data_y[0])/len(self.data_y)
        extents = [self.data_x[0]-half_step_x, self.data_x[-1]+half_step_x, \
                   self.data_y[-1]+half_step_y, self.data_y[0]-half_step_y] #left, right, bottom, up
        self.line = self.axes.imshow(self.data_z, animated=True, alpha=1, cmap=cmap, extent=extents)

        self.vmin = 0
        self.vmax = 0
        self.counts_max = 5
        
        
        divider = make_axes_locatable(self.axes)
        self.axright = divider.append_axes("top", size="20%", pad=0.25)
        
        if np.max(self.data_z.flatten()) == 0:
            hist_data = [0, 0]
        else:
            hist_data = [i for i in self.data_z.flatten() if i != 0]
        self.n, self.bins, self.patches = self.axright.hist(hist_data, orientation='vertical', bins=30, color='grey')
        self.axright.set_ylim(0, self.counts_max)
        self.cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = self.fig.colorbar(self.line, cax = self.cax)
        cbar.set_label(self.zlabel)

        
    def update_core(self):
        
        if np.max(self.data_z.flatten()) == 0:
            self.hist_data = [0, 0]
            vmin = 0
            vmax = np.max(self.data_z) + 1
        else:
            vmin = np.min(self.data_z[self.data_z!=0])
            vmax = np.max(self.data_z)
            if vmax == vmin:
                vmax = vmin + 1
            self.hist_data = [i for i in self.data_z.flatten() if i != 0]

        self.n, _ = np.histogram(self.hist_data, bins=self.bins)

        
        if vmin*0.8 < self.vmin or self.vmax < vmax*1.2:
            self.vmin = vmin*0.8
            self.vmax = vmax*1.2
        # filter out zero data

            self.line.set_clim(vmin=vmin, vmax=vmax)
            
            self.n, self.bins, self.patches = self.axright.hist(self.hist_data, orientation='vertical', 
                                                                bins=30, color='grey')
            self.counts_max = np.max(self.n) + 5
            self.axright.set_ylim(0, self.counts_max)
            self.axright.set_xlim(vmin - (vmax-vmin)*0.1, vmax + (vmax-vmin)*0.1)
            self.fig.canvas.draw()
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            
        elif np.max(self.n) >= self.counts_max:
            self.counts_max = np.max(self.n) + 5
            self.axright.set_ylim(0, self.counts_max)
            self.fig.canvas.draw()
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)


        self.fig.canvas.restore_region(self.bg_fig)
        self.line.set_array(self.data_z)
        
        for count, patch in zip(self.n, self.patches):
            patch.set_height(count)
            #patch.set_width(0.5)
        
        self.axright.draw_artist(self.axright.patch)
        for patch in self.patches:
            self.axright.draw_artist(patch)
            
        
    def choose_selector(self):
        self.area = AreaSelector(self.fig.axes[0])
        self.cross = CrossSelector(self.fig.axes[0])
        self.zoom = ZoomPan(self.fig.axes[0])
        
        cmap = self.axes.images[0].colorbar.mappable.get_cmap()
        self.line_l = self.axright.axvline(np.min(self.hist_data), color=cmap(0))
        self.line_h = self.axright.axvline(np.max(self.hist_data), color=cmap(0.95))
        self.drag_line = DragVLine(self.line_l, self.line_h, self.update_clim, self.axright)
        
        self.selector = [self.area, self.cross, self.zoom, self.drag_line]
    
    def update_clim(self):
        vmin = self.line_l.get_xdata()[0]
        vmax = self.line_h.get_xdata()[0]
        self.line.set_clim(vmin, vmax)

class PLGUILive(LivePlotGUI):
    
    def init_core(self):
    
        self.cbar = self.fig.axes[0].images[0].colorbar
        self.axright = self.fig.axes[1]
        self.cax = self.fig.axes[2]
        
        cmap_ = matplotlib.cm.get_cmap('inferno')
        cmap = cmap_.copy()
        cmap.set_under(cmap_(0))
        half_step_x = 0.5*(self.data_x[-1] - self.data_x[0])/len(self.data_x)
        half_step_y = 0.5*(self.data_y[-1] - self.data_y[0])/len(self.data_y)
        extents = [self.data_x[0]-half_step_x, self.data_x[-1]+half_step_x, \
                   self.data_y[-1]+half_step_y, self.data_y[0]-half_step_y] #left, right, bottom, up

        self.line = self.fig.axes[0].images[0]
        self.line.set(cmap = cmap, extent=extents)
        self.line.set_array(self.data_z)

        self.cbar.update_normal(self.line)
        self.cbar.set_label(self.zlabel)
        
        self.counts_max = 5
        self.vmin = 0
        self.vmax = 0
        # filter out zero data

        
        if np.max(self.data_z.flatten()) == 0:
            hist_data = [0, 0]
        else:
            hist_data = [i for i in self.data_z.flatten() if i != 0]
        self.n, self.bins, self.patches = self.axright.hist(hist_data, orientation='vertical', bins=30, color='grey')
        self.axright.set_ylim(0, self.counts_max)

        
        
    def update_core(self):
        if np.max(self.data_z.flatten()) == 0:
            self.hist_data = [0, 0]
            vmin = 0
            vmax = np.max(self.data_z) + 1
        else:
            vmin = np.min(self.data_z[self.data_z!=0])
            vmax = np.max(self.data_z)
            if vmax == vmin:
                vmax = vmin + 1
            self.hist_data = [i for i in self.data_z.flatten() if i != 0]

        self.n, _ = np.histogram(self.hist_data, bins=self.bins)

        
        if vmin*0.8 < self.vmin or self.vmax < vmax*1.2:
            #print('v')
            self.vmin = vmin*0.8
            self.vmax = vmax*1.2
        # filter out zero data

            self.line.set_clim(vmin=vmin, vmax=vmax)
            
            self.n, self.bins, self.patches = self.axright.hist(self.hist_data, orientation='vertical', 
                                                                bins=30, color='grey')
            self.counts_max = np.max(self.n) + 5
            self.axright.set_ylim(0, self.counts_max)
            self.axright.set_xlim(vmin - (vmax-vmin)*0.1, vmax + (vmax-vmin)*0.1)
            self.fig.canvas.draw()
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            
        elif np.max(self.n) >= self.counts_max:
            self.counts_max = np.max(self.n) + 5
            self.axright.set_ylim(0, self.counts_max)
            self.fig.canvas.draw()
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)


        self.fig.canvas.restore_region(self.bg_fig)
        self.line.set_array(self.data_z)
        
        for count, patch in zip(self.n, self.patches):
            patch.set_height(count)
            #patch.set_width(0.5)
        
        self.axright.draw_artist(self.axright.patch)
        for patch in self.patches:
            self.axright.draw_artist(patch)
        
    def choose_selector(self):
        self.area = AreaSelector(self.fig.axes[0])
        self.cross = CrossSelector(self.fig.axes[0])
        self.zoom = ZoomPan(self.fig.axes[0])
        
        cmap = self.axes.images[0].colorbar.mappable.get_cmap()
        self.line_l = self.axright.axvline(np.min(self.hist_data), color=cmap(0))
        self.line_h = self.axright.axvline(np.max(self.hist_data), color=cmap(0.95))
        self.drag_line = DragVLine(self.line_l, self.line_h, self.update_clim, self.axright)
        
        self.selector = [self.area, self.cross, self.zoom, self.drag_line]
    
    def update_clim(self):
        vmin = self.line_l.get_xdata()[0]
        vmax = self.line_h.get_xdata()[0]
        self.line.set_clim(vmin, vmax)
        
                      
class AreaSelector():
    def __init__(self, ax):
        self.ax = ax
        self.text = None
        self.range = [None, None, None, None]
        artist = ax.get_children()[0]
        if isinstance(artist, matplotlib.image.AxesImage):
            cmap = ax.images[0].colorbar.mappable.get_cmap()
            self.color = cmap(0.95)
        else:
            self.color = 'grey'
            
        self.selector = RectangleSelector(ax, self.onselect, interactive=True, useblit=False, button=[1], 
                                          props=dict(alpha=0.8, fill=False, 
                                                     linestyle='-', color=self.color)) 
        #set blit=True has weird bug, or implement RectangleSelector myself
        
    
        
    def get_decimal_places(self, value):
        non_zero = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}
        value_str = str(Decimal(value))

        if '.' in value_str:
            int_str, float_str = value_str.split('.')

            if any(num in non_zero for num in int_str):
                return 0

            decimal = 0
            for num_str in float_str:
                decimal += 1
                if num_str in non_zero:
                    return decimal
        else:
            return 0

        
    def onselect(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        if x1 == x2 or y1 == y2:
            self.range = [None, None, None, None]
            if self.text is not None:
                self.text.remove()
                self.text = None
            self.ax.figure.canvas.draw()
            return
        
        self.range = [min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)]
        
        
        x_data = self.ax.lines[0].get_xdata()
        y_data = self.ax.lines[0].get_ydata()
        self.gap_x = np.abs(np.max(x_data) - np.min(x_data))/1000
        self.gap_y = np.abs(np.max(y_data) - np.min(y_data))/1000
        
        
        decimal_x = self.get_decimal_places(self.gap_x)
        decimal_y = self.get_decimal_places(self.gap_y)
        
        format_str = f'{{:.{decimal_x}f}}, {{:.{decimal_y}f}}'
        
        if self.text is None:
            self.text = self.ax.text(0.025, 0.975, f'({format_str.format(x1, y1)})\n({format_str.format(x2, y2)})',
                    transform=self.ax.transAxes,
                    color=self.color, ha = 'left', va = 'top'
                    )
        else:
            self.text.set_text(f'({format_str.format(x1, y1)})\n({format_str.format(x2, y2)})')

        self.ax.figure.canvas.draw()
        
    def set_active(self, active):
        if not active:
            self.selector.set_active(False)
            


class CrossSelector():
    
    def __init__(self, ax):
        self.point = None
        self.ax = ax
        self.last_click_time = None
        self.wavelength = None # x of cross selector for PLE
        self.xy = None #xy of PL
        
        artist = ax.get_children()[0]
        if isinstance(artist, matplotlib.image.AxesImage):
            cmap = ax.images[0].colorbar.mappable.get_cmap()
            self.color = cmap(0.95)
        else:
            self.color = 'grey'
            
        x_data = self.ax.lines[0].get_xdata()
        y_data = self.ax.lines[0].get_ydata()
        self.gap_x = np.abs(np.max(x_data) - np.min(x_data)) / 1000 if (len(x_data)>0) else 0.01
        self.gap_y = np.abs(np.max(y_data) - np.min(y_data)) / 1000 if (len(y_data)>0) else 0.01
        
        self.cid_press = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)

    def on_press(self, event):
        if event.inaxes == self.ax:                
            if event.button == 3:  # mouse right key
                current_time = time.time()
                if self.last_click_time is None or (current_time - self.last_click_time) > 0.3:
                    self.last_click_time = current_time
                else:
                    self.last_click_time = None
                    self.remove_point()
                    self.ax.figure.canvas.draw()
                    return
                    

                decimal_x = self.get_decimal_places(self.gap_x)
                decimal_y = self.get_decimal_places(self.gap_y)

                format_str = f'{{:.{decimal_x}f}}, {{:.{decimal_y}f}}'
        
                x, y = event.xdata, event.ydata
                self.wavelength = x
                self.xy = [x, y]
                if self.point is None:
                    self.vline = self.ax.axvline(x, color=self.color, linestyle='-', alpha=0.8)
                    self.hline = self.ax.axhline(y, color=self.color, linestyle='-', alpha=0.8)
                    self.text = self.ax.text(0.975, 0.975, f'({format_str.format(x, y)})', ha='right', va='top', 
                                             transform=self.ax.transAxes, color=self.color)
                    self.point, = self.ax.plot(x, y, 'o', alpha=0.8, color=self.color)
                else:
                    self.vline.set_xdata(x)
                    self.hline.set_ydata(y)
                    self.point.set_xdata(x)
                    self.point.set_ydata(y)
                    self.text.set_text(f'({format_str.format(x, y)})')
                
                self.ax.figure.canvas.draw()
    
    def remove_point(self):
        if self.point is not None:
            self.vline.remove()
            self.hline.remove()
            self.point.remove()
            self.text.remove()
            self.point = None
            self.wavelength = None
            self.xy = None
    
    def get_decimal_places(self, value):
        non_zero = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}
        value_str = str(Decimal(value))

        if '.' in value_str:
            int_str, float_str = value_str.split('.')

            if any(num in non_zero for num in int_str):
                return 0

            decimal = 0
            for num_str in float_str:
                decimal += 1
                if num_str in non_zero:
                    return decimal
        else:
            return 0
        
    def set_active(self, active):
        if not active:
            self.ax.figure.canvas.mpl_disconnect(self.cid_press)

        
        
class ZoomPan():
    def __init__(self, ax):
        self.ax = ax
        self.cid_scroll = self.ax.figure.canvas.mpl_connect('scroll_event', self.on_scroll)
        artist = ax.get_children()[0]
        if isinstance(artist, matplotlib.image.AxesImage):
            self.image_type = 'imshow'
            cmap = ax.images[0].colorbar.mappable.get_cmap()
            self.color = cmap(0)
            self.ax.set_facecolor(self.color)
        else:
            self.image_type = 'plot'
            
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.x_center = (xlim[0] + xlim[1])/2
        self.y_center = (ylim[0] + ylim[1])/2


    def on_scroll(self, event):
        if event.inaxes == self.ax:
            xlim_min = self.ax.get_xlim()[0]
            ylim_min = self.ax.get_ylim()[0]

            scale_factor = 1.1 if event.button == 'up' else (1/1.1)

            xlim = [scale_factor*(xlim_min - self.x_center) + self.x_center\
                    , self.x_center - scale_factor*(xlim_min - self.x_center)]
            ylim = [scale_factor*(ylim_min - self.y_center) + self.y_center\
                    , self.y_center - scale_factor*(ylim_min - self.y_center)]
            
            if self.image_type == 'imshow':
                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)
            else:
                self.ax.set_xlim(xlim)
            self.ax.figure.canvas.draw()
            
    def set_active(self, active):
        if not active:
            self.ax.figure.canvas.mpl_disconnect(self.cid_scroll)
    
            
class DragVLine():
    def __init__(self, line_l, line_h, update_func, ax):
        self.line_l = line_l
        self.line_h = line_h
        self.ax = ax
        self.press = None
        self.update_func = update_func
        self.line_l.set_animated(True)
        self.line_h.set_animated(True)
        self.is_on_l = False
        self.is_on_h = False
        self.useblit = True
        self.background = None
        self.cid_press = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_draw = self.ax.figure.canvas.mpl_connect('draw_event', self.on_draw)
        self.last_update_time = time.time()

    def on_draw(self, event):
        self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
        for line in self.ax.lines:
            self.ax.draw_artist(line)


    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        contains_l, _ = self.line_l.contains(event)
        contains_h, _ = self.line_h.contains(event)
        if not (contains_l or contains_h):
            return
        
        if contains_l:
            self.is_on_l = True
            self.press = self.line_l.get_xdata(), event.xdata
        else:
            self.is_on_h = True
            self.press = self.line_h.get_xdata(), event.xdata

        self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)

    def on_motion(self, event):
        if not self.press:
            return
        if event.inaxes != self.ax:
            return
        current_time = time.time()
        if current_time - self.last_update_time < 0.03:
            return
        xpress, xdata = self.press
        dx = event.xdata - xdata
        new_xdata = [x + dx for x in xpress]
        if self.is_on_l:
            self.line_l.set_xdata(new_xdata)
        if self.is_on_h:
            self.line_h.set_xdata(new_xdata)
        self.update_func()


        self.ax.figure.canvas.restore_region(self.background)
        for line in self.ax.lines:
            self.ax.draw_artist(line)
        self.ax.figure.canvas.blit(self.ax.bbox)


        self.last_update_time = current_time

    def on_release(self, event):
        self.press = None
        self.is_on_l = False
        self.is_on_h = False
        self.update_func()
        self.ax.figure.canvas.draw()
        
    def set_active(self, active):
        if not active:
            self.ax.figure.canvas.mpl_disconnect(self.cid_press)
            self.ax.figure.canvas.mpl_disconnect(self.cid_release)
            self.ax.figure.canvas.mpl_disconnect(self.cid_motion)
            self.ax.figure.canvas.mpl_disconnect(self.cid_draw)
            
            
class DataFigure():
    """
    The class contains all data of the figure, enables more operations
    such as curve fit or save data
    
    Parameters
    ----------
    live_plot :instance of class LivePlot
    
    
    Examples
    --------
    >>> data_figure = DataFigure(live_plot)
    
    >>> data_x, data_y = data_figure.data
    
    >>> data_figure.save('my_figure')
    'save to my_figure_{time}.jpg and my_figure_{time}.txt'

    >>> data_figure.lorent(p0 = None)
    'figure with lorent curve fit'
    
    >>> data_figure.clear()
    'remove lorent fit and text'
    """
    def __init__(self, live_plot, address=None):

        if address is None:
            self.fig = live_plot.fig
            self.selector = live_plot.selector
            self.info = live_plot.data_generator.info
            self.live_plot = live_plot
            # load all necessary info defined in device.info for device in config_instances
        else:
            if '*' in address:
                files_all = glob.glob(address)# includes .jpg, .npz etc.
                files = []
                for file in files_all:
                    if '.npz' in file:
                        files.append(file)

                if len(files) > 1:
                    print(files)
                data_generator = LoadAcquire(address=files[0])
            else:
                data_generator = LoadAcquire(address=address)

            exposure = data_generator.info.get('exposure')
            if data_generator.type == 'PLE':
                data_x = data_generator.data_x
                data_y = data_generator.data_y
                _live_plot = PLELive(labels=['Wavelength (nm)', f'Counts/{exposure}s'], \
                                    update_time=0.1, data_generator=data_generator, data=[data_x, data_y], config_instances = None)
            elif data_generator.type == 'CPT':
                data_x = data_generator.data_x
                data_y = data_generator.data_y
                _live_plot = PLELive(labels=['frequency', f'Counts/{exposure}s'], \
                                    update_time=0.1, data_generator=data_generator, data=[data_x, data_y], config_instances = None)

            else:
                data_x = data_generator.data_x
                data_y = data_generator.data_y
                data_z = data_generator.data_z
                _live_plot = PLDisLive(labels=['X', 'Y', f'Counts/{exposure}s'], \
                        update_time=1, data_generator=data_generator, data=[data_x, data_y, data_z], config_instances = None)

            fig, selector = _live_plot.plot()
            self.fig = _live_plot.fig
            self.selector = _live_plot.selector
            self.info = _live_plot.data_generator.info
            # load all necessary info defined in device.info for device in config_instances



        self.p0 = None
        self.fit = None
        self.text = None
        self.log_info = '' # information for log output
        self.unit = 'nm'
        
        artist = self.fig.axes[0].get_children()[0]
        if isinstance(artist, matplotlib.image.AxesImage):
            self.mode = 'PL'
        else:
            self.mode = 'PLE'
        
        if self.mode == 'PLE':
            self.data_x = self.fig.axes[0].lines[0].get_xdata()
            self.data_y = self.fig.axes[0].lines[0].get_ydata()
            self._data = [self.data_x, self.data_y]
        else:#PL
            #self.data_array = self.fig.axes[0].images[0].get_array()
            #self.extent = self.fig.axes[0].images[0].get_extent()
            if address is None:
                self.data_x = self.live_plot.data_x
                self.data_y = self.live_plot.data_y
                self.data_z = self.live_plot.data_z
            else:
                self.data_x = data_x
                self.data_y = data_y
                self.data_z = data_z
            #self._data = [self.data_array, self.extent]
            self._data = [self.data_x, self.data_y, self.data_z]

    
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, data_x_data_y):
        print('can not assign data')

    def xlim(self, x_min, x_max):
        self.fig.axes[0].set_xlim(x_min, x_max)

    def ylim(self, y_min, y_max):
        self.fig.axes[0].set_ylim(y_min, y_max)


        
    def save(self, addr='', extra_info=None):
        current_time = time.localtime()
        current_date = time.strftime("%Y-%m-%d", current_time)
        current_time_formatted = time.strftime("%H:%M:%S", current_time)
        time_str = current_date.replace('-', '_') + '_' + current_time_formatted.replace(':', '_')

        if addr=='' or ('/' not in addr):
            pass
        else:
            directory = os.path.dirname(addr)
            if not os.path.exists(directory):
                os.makedirs(directory)


        self.fig.savefig(addr + self.mode + time_str + '.jpg', dpi=300)

        if extra_info is None:
            extra_info = {}
        
        if self.mode == 'PLE':
            np.savez(addr + self.mode + time_str + '.npz', data_x = self.data_x, data_y = self.data_y, info = {**self.info, **extra_info})
        else:
            #np.savez(addr + self.mode + time_str + '.npz', extent = self.extent, array = self.data_array, info = {**self.info, **extra_info})
            np.savez(addr + self.mode + time_str + '.npz', data_x = self.data_x, data_y = self.data_y, data_z = self.data_z, info = {**self.info, **extra_info})

        print(f'saved fig as {addr}{self.mode}{time_str}.npz')
        
        
    def lorent(self, p0=None, is_print=True, is_save=False, fit=True):
        if self.mode == 'PL':
            return 
        spl = 299792458
        
        def _lorent(x, center, full_width, height, bg):
            return height*((full_width/2)**2)/((x - center)**2 + (full_width/2)**2) + bg
        
        if p0 is None:# no input
            self.p0 = [self.data_x[np.argmax(self.data_y)], 
                           (self.data_x[-1] - self.data_x[0])/4, np.max(self.data_y), 0]
        else:
            self.p0 = p0
        
        if fit:
            popt, pcov = curve_fit(_lorent, self.data_x, self.data_y, p0=self.p0)
        else:
            popt, pcov = self.p0, None
        
        if is_print:
            pass
            #print(f'popt = {popt}, pcov = {pcov}')
        
        if self.fit is None:
            self.fit = self.fig.axes[0].plot(self.data_x, _lorent(self.data_x, *popt), color='orange', linestyle='--')
        else:
            self.fit[0].set_ydata(_lorent(self.data_x, *popt))
        self.fig.canvas.draw()
        
        if is_print:
            

            full_width_GHz = np.abs(spl/(popt[0]-0.5*popt[1]) - spl/(popt[0]+0.5*popt[1]))
                
            _popt = np.insert(popt, 2, full_width_GHz)
            
            if self.unit == 'nm':
                pass
            else:
                _popt[1], _popt[2] = _popt[2], _popt[1]
            
            popt_str = ['center', 'FWHM', 'in GHz', 'height', 'bg']
            formatted_popt = [f'{x:.5f}'.rstrip('0') for x in _popt]
            result_list = [f'{name} = {value}' for name, value in zip(popt_str, formatted_popt)]
            formatted_popt_str = '\n'.join(result_list)
            result = f'{formatted_popt_str}'
            self.log_info = result
            # format popt to display as text
                
            
            if self.text is None:
                self.text = self.fig.axes[0].text(0.025, 0.5, 
                                                  result, transform=self.fig.axes[0].transAxes, 
                                                  color='red', ha='left', va='center')
            else:
                self.text.set_text(result)
                
            self.fig.canvas.draw()
            
        if is_save:
            self.save(addr='')

        return [popt_str, pcov], _popt


    def lorent_zeeman(self, p0=None, is_print=True, is_save=False, func=None, bounds = None, fit=True):
        #fit of PLE under B field
        if self.mode == 'PL':
            return 
        spl = 299792458
        
        def _lorent_zeeman(x, center, full_width, height, bg, factor, split):
            return height*((full_width/2)**2)/((x - center - split/2)**2 + (full_width/2)**2) \
                + factor*height*((full_width/2)**2)/((x - center + split/2)**2 + (full_width/2)**2) + bg




        data_y = self.data_y
        data_x = self.data_x
        guess_height = (np.max(data_y)-np.min(data_y))
        peaks, properties = find_peaks(data_y, width=3, prominence=guess_height/8) # width about 100MHz
        peaks_largest2 = peaks[np.argsort(data_y[peaks])[::-1]][:2]
        guess_center = data_x[int(np.mean(peaks_largest2))]
        guess_width = 0.0007
        guess_spl = np.abs((data_x[peaks_largest2[0]] - guess_center)*2) + 0.0001
        data_width = np.abs(data_x[-1] - data_x[0])

        
        if p0 is None:# no input
            self.p0 = [guess_center, guess_width, guess_height, np.min(data_y), 1, guess_spl]
        else:
            self.p0 = p0
        
        if func is None:
            _func = _lorent_zeeman
        else:
            _func = func

        if bounds is None:
            self.bounds = ([np.min(data_x), guess_width/8, guess_height/2,-np.inf, 0.99, 0.00001], \
                              [np.max(data_x), guess_width*4, guess_height*1.1, np.inf, 1, guess_spl*5])
        else:
            self.bounds = bounds

        if fit:
            popt, pcov = curve_fit(_func, self.data_x, self.data_y, p0=self.p0, bounds = self.bounds)
        else:
            popt, pcov = self.p0, None
        
        if is_print:
            pass
            #print(f'popt = {popt}, pcov = {pcov}')
        
        if self.fit is None:
            self.fit = self.fig.axes[0].plot(self.data_x, _func(self.data_x, *popt), color='orange', linestyle='--')
        else:
            self.fit[0].set_ydata(_func(self.data_x, *popt))
        self.fig.canvas.draw()
        
        if is_print:
            

            full_width_GHz = np.abs(spl/(popt[0]-0.5*popt[1]) - spl/(popt[0]+0.5*popt[1]))
            splitting_GHz = np.abs(spl/(popt[0]-0.5*popt[-1]) - spl/(popt[0]+0.5*popt[-1]))

            _popt = np.insert(popt, 2, full_width_GHz)
            _popt = np.insert(_popt[:-1], len(_popt[:-1]), splitting_GHz)
            
            if self.unit == 'nm':
                pass
            else:
                _popt[1], _popt[2] = _popt[2], _popt[1]
            
            popt_str = ['center', 'FWHM', 'in GHz', 'height', 'bg', 'factor', 'split (GHz)']
            formatted_popt = [f'{x:.5f}'.rstrip('0') for x in _popt]
            result_list = [f'{name} = {value}' for name, value in zip(popt_str, formatted_popt)]
            formatted_popt_str = '\n'.join(result_list)
            result = f'{formatted_popt_str}'
            self.log_info = result
            # format popt to display as text
                
            
            if self.text is None:
                self.text = self.fig.axes[0].text(0.025, 0.5, 
                                                  result, transform=self.fig.axes[0].transAxes, 
                                                  color='red', ha='left', va='center')
            else:
                self.text.set_text(result)
                
            self.fig.canvas.draw()
            
        if is_save:
            self.save(addr='')

        return [popt_str, pcov], _popt


    def cpt(self, p0=None, is_print=True, is_save=False, fit=True):
        if self.mode == 'PL':
            return 
        spl = 299792458
        
        def _cpt(x, center, full_width, height, bg_c, bg_k):
            return height*((full_width/2)**2)/((x - center)**2 + (full_width/2)**2) + bg_k*x + bg_c
        
        if p0 is None:# no input
            guess_k = (self.data_y[-1] - self.data_y[0])/(self.data_x[-1] - self.data_x[0])#
            guess_c = self.data_y[0] - guess_k*self.data_x[0]
            self.p0 = [np.mean(self.data_x), 
                           (self.data_x[-1] - self.data_x[0])/5, -(np.max(self.data_y) - np.min(self.data_y)), guess_c, guess_k]
        else:
            self.p0 = p0
        
        if fit:
            popt, pcov = curve_fit(_cpt, self.data_x, self.data_y, p0=self.p0)
        else:
            popt, pcov = self.p0, None
        
        if is_print:
            pass
            #print(f'popt = {popt}, pcov = {pcov}')
        
        if self.fit is None:
            self.fit = self.fig.axes[0].plot(self.data_x, _cpt(self.data_x, *popt), color='orange', linestyle='--')
        else:
            self.fit[0].set_ydata(_cpt(self.data_x, *popt))
        self.fig.canvas.draw()
        
        if is_print:
            

            full_width_GHz = np.abs(spl/(popt[0]-0.5*popt[1]) - spl/(popt[0]+0.5*popt[1]))
                
            _popt = np.insert(popt, 2, full_width_GHz)
            
            if self.unit == 'nm':
                pass
            else:
                _popt[1], _popt[2] = _popt[2], _popt[1]
            
            popt_str = ['center', 'FWHM', 'in GHz', 'height', 'bg']
            formatted_popt = [f'{x:.5f}'.rstrip('0') for x in _popt[:-1]]
            result_list = [f'{name} = {value}' for name, value in zip(popt_str, formatted_popt)]
            formatted_popt_str = '\n'.join(result_list)
            result = f'{formatted_popt_str}'
            self.log_info = result
            # format popt to display as text
                
            
            if self.text is None:
                self.text = self.fig.axes[0].text(0.025, 0.5, 
                                                  result, transform=self.fig.axes[0].transAxes, 
                                                  color='red', ha='left', va='center')
            else:
                self.text.set_text(result)
                
            self.fig.canvas.draw()
            
        if is_save:
            self.save(addr='')

        return [popt_str, pcov], _popt

    def T1spin_inte(self, t_array=None, fine_tune_pos=True):
        # integral the beginning x ns for overcoming AOM rise/fall time and better SNR
        # t array [t of first peak, t of second peak, gap after peak to inte] in ns
        if self.mode == 'PL':
            return 
        spl = 299792458

        t_gap = int(np.abs(self.data_x[1] - self.data_x[0]))

        if (t_array is None) or (len(t_array) != 3):
            return 

        mean_counts = np.mean(self.data_y)
        bg_at_plateau = np.mean(self.data_y[self.data_y > mean_counts])
        print(bg_at_plateau)

        init_region = (self.data_x>t_array[0]) & (self.data_x<(t_array[0]+t_array[2]))
        ro_region = (self.data_x>t_array[1]) & (self.data_x<(t_array[1]+t_array[2]))
        if fine_tune_pos:
            for ii in range(t_array[0]//t_gap, t_array[0]//t_gap+1000):
                if self.data_y[ii] >= bg_at_plateau:
                    init_first = ii
                    print(ii)
                    break
            for ii in range(t_array[1]//t_gap, t_array[0]//t_gap+1000):
                if self.data_y[ii] >= bg_at_plateau:
                    ro_first = ii
                    print('ro',ii)
                    break

        popt_str = ['counts_init', 'counts_ro']
        if fine_tune_pos:
            counts_init = np.sum(self.data_y[init_first:(init_first+t_array[2]//t_gap)])
            counts_ro = np.sum(self.data_y[ro_first:(ro_first+t_array[2]//t_gap)])
        else:
            counts_init = np.sum(self.data_y[init_region])
            counts_ro = np.sum(self.data_y[ro_region])
        formatted_popt = [counts_init, counts_ro]
        result_list = [f'{name} = {value}' for name, value in zip(popt_str, formatted_popt)]
        formatted_popt_str = '\n'.join(result_list)
        result = f'{formatted_popt_str}'

        if self.text is None:
            self.fill1 = self.fig.axes[0].fill_between(self.data_x, self.data_y, where=init_region , color='red', alpha=0.3)
            self.fill2 = self.fig.axes[0].fill_between(self.data_x, self.data_y, where=ro_region, color='red', alpha=0.3)
        else:
            self.fill1.remove()
            self.fill2.remove()
            self.fill1 = self.fig.axes[0].fill_between(self.data_x, self.data_y, where=init_region , color='red', alpha=0.3)
            self.fill2 = self.fig.axes[0].fill_between(self.data_x, self.data_y, where=ro_region, color='red', alpha=0.3)

        if self.text is None:
            self.text = self.fig.axes[0].text(0.025, 0.975, 
                                                  result, transform=self.fig.axes[0].transAxes, 
                                                  color='red', ha='left', va='top')
        else:
            self.text.set_text(result)



    def T1spin(self, t_array=None, p0=None, is_print=True, is_save=False, fit=True):
        # t array [t of first peak, t of second peak, gap after peak to fit]
        if self.mode == 'PL':
            return 
        spl = 299792458

        if (t_array is None) or (len(t_array) != 3):
            return 
        
        def _T1spin(x, bg, height_init, height_ro, decay):

            if isinstance(x, np.ndarray):  
                return np.array([_T1spin(xi, bg, height_init, height_ro, decay) for xi in x])
            else:
                if t_array[0]<=x<=(t_array[0]+t_array[2]):
                    return bg + height_init*np.exp(-(x-t_array[0])/decay)
                elif t_array[1]<=x<=(t_array[1]+t_array[2]):
                    return bg + height_ro*np.exp(-(x-t_array[1])/decay)
                else:
                    return self.data_y[np.where(self.data_x == x)[0][0]]
                """
                if x<=gap_array[0] or gap_array[1]<=x<=gap_array[2]:
                    return bg
                elif gap_array[0]<x<gap_array[1]:
                    return bg + height_init*np.exp(-(x-gap_array[0])/decay) + height_thermal
                else:
                    return bg + height_ro*np.exp(-(x-gap_array[2])/decay) + height_thermal
                """
        if p0 is None:# no input
            guess_bg = self.data_y[-1]
            guess_height_init = np.max(self.data_y) - guess_bg
            guess_height_ro = guess_height_init/2
            guess_decay = 100
            self.p0 = [guess_bg, guess_height_init, guess_height_ro, guess_decay]

            self.bounds = ([guess_bg/3, guess_height_init/3, 0.1, guess_decay/3], \
                              [guess_bg*3, guess_height_init*3, guess_height_init*3, guess_decay*3])
            """
            guess_bg = np.min(self.data_y)
            guess_height_thermal = self.data_y[-1] - guess_bg
            guess_height_init = np.max(self.data_y) - guess_height_thermal
            guess_height_ro = guess_height_init/2
            guess_decay = 100
            self.p0 = [guess_bg, guess_height_thermal, guess_height_init, guess_height_ro, guess_decay]
            """

        else:
            self.p0 = p0
            guess_bg = p0[0]
            guess_height_init = p0[1]
            guess_height_ro = p0[2]
            guess_decay = p0[3]

            self.bounds = ([guess_bg/5, guess_height_init/5, 0.1, 1], \
                              [guess_bg*5, guess_height_init*5, guess_height_init*5, guess_decay*5])
        
        if fit:
            data_x = self.data_x
            partial_data_index = np.where(((t_array[0]<=data_x) & (data_x<=(t_array[0]+t_array[2]))) \
                                | ((t_array[1]<=data_x) & (data_x<=(t_array[1]+t_array[2])))) # index for part of data contains peaks 
            popt, pcov = curve_fit(_T1spin, self.data_x[partial_data_index], self.data_y[partial_data_index], p0=self.p0, bounds = self.bounds)
        else:
            popt, pcov = self.p0, None
        
        if is_print:
            pass
            #print(f'popt = {popt}, pcov = {pcov}')
        
        if self.fit is None:
            self.fit = self.fig.axes[0].plot(self.data_x, _T1spin(self.data_x, *popt), color='orange', linestyle='--')
        else:
            self.fit[0].set_ydata(_T1spin(self.data_x, *popt))
        self.fig.canvas.draw()
        
        if is_print:
            

            
            popt_str = ['bg', 'h_init', 'h_ro', 'decay']
            formatted_popt = [f'{x:.5f}'.rstrip('0') for x in popt]
            result_list = [f'{name} = {value}' for name, value in zip(popt_str, formatted_popt)]
            formatted_popt_str = '\n'.join(result_list)
            result = f'{formatted_popt_str}'
            self.log_info = result
            # format popt to display as text
                
            
            if self.text is None:
                self.text = self.fig.axes[0].text(0.025, 0.975, 
                                                  result, transform=self.fig.axes[0].transAxes, 
                                                  color='red', ha='left', va='top')
            else:
                self.text.set_text(result)
                
            self.fig.canvas.draw()
            
        if is_save:
            self.save(addr='')

        return [popt_str, pcov], popt


    def T1spin_v2(self, t_array=None, p0=None, is_print=True, fit=True):
        # t array [t before first peak, t before second peak, gap after t to fit, gap]
        # t_array assign using which part of data for fitting
        if self.mode == 'PL':
            return 
        spl = 299792458

        t_gap = int(np.abs(self.data_x[1] - self.data_x[0]))

        if (t_array is None) or (len(t_array) != 4):
            return 
        
        def _T1spin(x, bg0, bg1, height_init, height_ro, decay_ex, decay_op, peak_init):
            # height is peak counts - background1
            peak_ro = peak_init + t_array[3]
            height_init_abs = height_init + bg1
            height_ro_abs = height_ro + bg1

            if isinstance(x, np.ndarray):  
                return np.array([_T1spin(xi, bg0, bg1, height_init, height_ro, decay_ex, decay_op, peak_init) for xi in x])
            else:
                if t_array[0]<=x<=(t_array[0]+t_array[2]):
                    if x<=peak_init:
                        return bg0 + (height_init_abs - bg0)*(np.exp(-(peak_init-x)/decay_ex))
                    else:
                        return bg1 + (height_init_abs - bg1)*(np.exp(-(x-peak_init)/decay_op))

                elif t_array[1]<=x<=(t_array[1]+t_array[2]):
                    if x<=peak_ro:
                        return bg0 + (height_ro_abs - bg0)*(np.exp(-(peak_ro-x)/decay_ex))
                    else:
                        return bg1 + (height_ro_abs - bg1)*(np.exp(-(x-peak_ro)/decay_op))
                else:
                    return self.data_y[np.where(self.data_x == x)[0][0]]
  
        if p0 is None:# no input


            guess_bg0 = np.min(self.data_y)
            guess_bg1 = self.data_y[int((t_array[1] + t_array[2])//t_gap)]
            guess_height_init = np.max(self.data_y) - guess_bg1
            guess_height_ro = guess_height_init/2
            guess_decay_ex = 2
            guess_decay_op = 20
            guess_peak_init = int(self.data_x[np.argmax(self.data_y[t_array[0]//t_gap:(t_array[0]+t_array[2])//t_gap]) + (t_array[0]//t_gap)])

            self.p0 = [guess_bg0, guess_bg1, guess_height_init, guess_height_ro, guess_decay_ex, guess_decay_op, guess_peak_init]

            self.bounds = ([guess_bg0/5, guess_bg1/5, 0, 0, guess_decay_ex/5, guess_decay_op/5, guess_peak_init-t_array[2]/2], \
                              [guess_bg0*5, guess_bg1*5, guess_height_init*5, guess_height_ro*5, guess_decay_ex*5, \
                              guess_decay_op*5, guess_peak_init+t_array[2]/2])


        else:
            self.p0 = p0
            guess_bg0 = p0[0]
            guess_bg1 = p0[1]
            guess_height_init = p0[2]
            guess_height_ro = p0[3]
            guess_decay_ex = p0[4]
            guess_decay_op = p0[5]
            guess_peak_init = p0[6]

            self.bounds = ([guess_bg0/5, guess_bg1/5, 0, 0, guess_decay_ex/5, guess_decay_op/5, guess_peak_init-t_array[2]/2], \
                              [guess_bg0*5, guess_bg1*5, guess_height_init*5, guess_height_ro*5, guess_decay_ex*5, \
                              guess_decay_op*5, guess_peak_init+t_array[2]/2])
        
        
        if fit:
            data_x = self.data_x
            partial_data_index = np.where(((t_array[0]<=data_x) & (data_x<=(t_array[0]+t_array[2]))) \
                                | ((t_array[1]<=data_x) & (data_x<=(t_array[1]+t_array[2])))) # index for part of data contains peaks 
            popt, pcov = curve_fit(_T1spin, self.data_x[partial_data_index], self.data_y[partial_data_index], p0=self.p0, bounds = self.bounds)
        else:
            popt, pcov = self.p0, None


        if self.fit is None:
            self.fit = self.fig.axes[0].plot(self.data_x, _T1spin(self.data_x, *popt), color='orange', linestyle='--')
        else:
            self.fit[0].set_ydata(_T1spin(self.data_x, *popt))
        self.fig.canvas.draw()
        
        if is_print:

            
            popt_str = ['bg0', 'bg1', 'height_init', 'height_ro', 'decay_ex', 'decay_op', 'peak_ro']
            formatted_popt = [f'{x:.5f}'.rstrip('0') for x in popt]
            result_list = [f'{name} = {value}' for name, value in zip(popt_str, formatted_popt)]
            formatted_popt_str = '\n'.join(result_list)
            result = f'{formatted_popt_str}'
            self.log_info = result
            # format popt to display as text
                
            
            if self.text is None:
                self.text = self.fig.axes[0].text(0.025, 0.975, 
                                                  result, transform=self.fig.axes[0].transAxes, 
                                                  color='red', ha='left', va='top')
            else:
                self.text.set_text(result)
                
            self.fig.canvas.draw()

        return [popt_str, pcov], popt
            
    def clear(self):
        if (self.text is None) and (self.fit is None):
            return
        if self.text is not None:
            self.text.remove()
        if self.fit is not None:
            self.fit[0].remove()
        self.fig.canvas.draw()
        self.fit = None
        self.text = None
        
    def _change_unit(self):
        spl = 299792458
        xlim = self.fig.axes[0].get_xlim()
        
        for line in self.fig.axes[0].lines:
            data_x = np.array(line.get_xdata())
            if np.array_equal(data_x, np.array([0, 1])):
                line.set_xdata(data_x)
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    new_xdata = np.where(data_x != 0, spl / data_x, np.inf)
                    line.set_xdata(new_xdata)
            
        for patch in self.fig.axes[0].patches:# move rectangle
            x, y = patch.get_xy()
            width = patch.get_width()
            if x==0:
                x=np.inf
            if width==0:
                width=np.inf
            patch.set_x(spl/x)
            patch.set_width(spl/(x + width) - spl/(x))
            
        # set Zoom pan center, and moves all selectors accordingly
        if self.selector == []:
            pass
        else:
            zoom_pan_handle = self.selector[2]
            zoom_pan_handle.x_center = spl/zoom_pan_handle.x_center
            
            area_handle = self.selector[0]
            if area_handle.range[0] is not None:
                area_handle.range = spl/np.array(area_handle.range)
                
            cross_handle = self.selector[1]
            if cross_handle.wavelength is not None:
                cross_handle.wavelength = spl/cross_handle.wavelength
            
        self.data_x = self.fig.axes[0].lines[0].get_xdata()
        self.data_y = self.fig.axes[0].lines[0].get_ydata()
        self._data = [self.data_x, self.data_y]
            
        self.fig.axes[0].set_xlim(spl/xlim[0], spl/xlim[-1])
            
            
    def to_GHz(self):
        if self.unit == 'GHz':
            return
        if self.mode == 'PL':
            return 
        spl = 299792458
        self.unit = 'GHz'
        
        self._change_unit()
        
        xlabel = self.fig.axes[0].get_xlabel()
        self.fig.axes[0].set_xlabel(xlabel[:-4] + '(GHz)')
        self.fig.canvas.draw()
        
        
    def to_nm(self):
        if self.unit == 'nm':
            return
        if self.mode == 'PL':
            return 
        
        spl = 299792458
        self.unit = 'nm'
        
        self._change_unit()
        
        xlabel = self.fig.axes[0].get_xlabel()
        self.fig.axes[0].set_xlabel(xlabel[:-5] + '(nm)')
        self.fig.canvas.draw()
            
            
def ple(wavelength_array, exposure, config_instances, repeat=1):
                
    data_x = wavelength_array
    data_y = np.zeros(len(data_x))
    data_generator = PLEAcquire(exposure = exposure, data_x=data_x, data_y=data_y, config_instances = config_instances, repeat=repeat)
    liveplot = PLELive(labels=['Wavelength (nm)', f'Counts/{exposure}s'], \
                        update_time=0.1, data_generator=data_generator, data=[data_x, data_y], config_instances = config_instances)
    fig, selector = liveplot.plot()
    data_figure = DataFigure(liveplot)
    return fig, data_figure


def odmr(frequency_array, exposure, power, config_instances, repeat=1):
                
    data_x = frequency_array
    data_y = np.zeros(len(data_x))
    data_generator = ODMRAcquire(exposure = exposure, data_x=data_x, data_y=data_y, power = power, config_instances = config_instances, repeat=repeat)
    liveplot = PLELive(labels=['Frequency (Hz)', f'Counts/{exposure}s'], \
                        update_time=0.1, data_generator=data_generator, data=[data_x, data_y], config_instances = config_instances, relim_mode='tight')
    fig, selector = liveplot.plot()
    data_figure = DataFigure(liveplot)
    return fig, data_figure

def pl(center, coordinates_x, coordinates_y, exposure, config_instances, is_dis = False, wavelength=None):
    """
    example
    
    >>> pl(center=[0, 0], coordinates_x=np.linspace(-5,5,10), \
        coordinates_y=np.linspace(-5,5,10), exposure=0.2)
    """
    
    data_x = np.array(coordinates_x) + center[0]
    data_y = np.array(coordinates_y) + center[1]
    data_z = np.zeros((len(data_y), len(data_x)))
    # reverse for compensate x,y order of imshow
    data_generator = PLAcquire(exposure = exposure, data_x = data_x, data_y = data_y, \
                               data_z = data_z, config_instances = config_instances, wavelength=wavelength)
    if is_dis:
        liveplot = PLDisLive(labels=['X', 'Y', f'Counts/{exposure}s'], \
                        update_time=1, data_generator=data_generator, data=[data_x, data_y, data_z], config_instances = config_instances)
    else:
        liveplot = PLLive(labels=['X', 'Y', f'Counts/{exposure}s'], \
                            update_time=1, data_generator=data_generator, data=[data_x, data_y, data_z], config_instances = config_instances)
    fig, selector = liveplot.plot()
    data_figure = DataFigure(liveplot)
    return fig, data_figure


def live(data_array, exposure, config_instances, wavelength=None, is_finite=False):
                
    data_x = data_array
    data_y = np.zeros(len(data_x))
    data_generator = LiveAcquire(exposure = exposure, data_x=data_x, data_y=data_y, \
                                 config_instances = config_instances, wavelength=wavelength, is_finite=is_finite)
    liveplot = PLELive(labels=['Data', f'Counts/{exposure}s'], \
                        update_time=0.01, data_generator=data_generator, data=[data_x, data_y], config_instances = config_instances)
    fig, selector = liveplot.plot()
    data_figure = DataFigure(liveplot)
    return fig, data_figure

def pl_gui(center, coordinates_x, coordinates_y, exposure, config_instances, fig = None, wavelength=None):
    """
    example
    
    >>> pl(center=[0, 0], coordinates_x=np.linspace(-5,5,10), \
        coordinates_y=np.linspace(-5,5,10), exposure=0.2)
    """
    
    data_x = np.array(coordinates_x) + center[0]
    data_y = np.array(coordinates_y) + center[1]
    data_z = np.zeros((len(data_y), len(data_x)))
    # reverse for compensate x,y order of imshow
    data_generator = PLAcquire(exposure = exposure, data_x = data_x, data_y = data_y, \
                               data_z = data_z, config_instances = config_instances, wavelength=wavelength)
    liveplot = PLGUILive(labels=['X', 'Y', f'Counts/{exposure}s'], \
                        update_time=1, data_generator=data_generator, data=[data_x, data_y, data_z], fig=fig, config_instances = config_instances)
    fig, selector = liveplot.plot()
    data_figure = DataFigure(liveplot)
    return fig, data_figure

def area(wavelength_array, exposure, coordinates_x, coordinates_y, config_instances, mode = 'PLE'):
                
    data_x = wavelength_array
    if mode == 'PLE':
        data_y = np.zeros(len(data_x))
    elif mode == 'PLE':
        data_y = np.zeros((len(coordinates_y), len(coordinates_x)))

    data_generator = AreaAcquire(exposure = exposure, data_x=data_x, data_y=data_y, 
                                    data_x_area = coordinates_x, data_y_area = coordinates_y,
                                     config_instances = config_instances, mode = mode)
    if mode == 'PLE':
        liveplot = PLELive(labels=['Wavelength (nm)', f'Counts/{exposure}s'], \
                            update_time=0.1, data_generator=data_generator, data=[data_x, data_y], config_instances = config_instances)
    elif mode == 'PL':
        liveplot = PLDisLive(labels=['X', 'Y', f'Counts/{exposure}s'], \
                        update_time=1, data_generator=data_generator, data=[coordinates_x, coordinates_y, data_y], config_instances = config_instances)

    fig, selector = liveplot.plot()
    data_figure = DataFigure(liveplot)
    return fig, data_figure


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










from matplotlib.animation import FuncAnimation



class LivePlotWithAnimation:
    """
    使用 matplotlib.animation.FuncAnimation 实现实时更新的基础类。
    """
    def __init__(self, labels, update_time, data_generator, data, fig=None, config_instances=None, relim_mode='normal'):
        self.labels = labels
        self.data_x = data[0]
        self.data_y = data[1]
        self.update_time = update_time
        self.data_generator = data_generator
        self.fig = plt.figure()
        self.relim_mode = relim_mode
        self.ylim_max = 100
        self.ylim_min = 0
        self.line = None
        self.axes = None
        self.bg_fig = None
        self.anim = None

    def init_figure_and_data(self):
        #change_to_nbagg(params_type = 'nbagg', scale=1)
        plt.ion()
        self.axes = self.fig.add_subplot(111)
        #self.clear_all()
        self.axes.set_ylabel(self.labels[1])
        self.axes.set_xlabel(self.labels[0])
        self.init_core()


        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.resizable = False
        self.fig.canvas.capture_scroll = True
        #display_immediately(self.fig)

            
        #self.axes.set_autoscale_on(True)
        self.fig.tight_layout()



        self.data_generator.start()

    def init_core(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def update_core(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def plot(self):
        self.init_figure_and_data()

        def update(frame):
            print('hello')
            if not self.data_generator.is_done:
                self.update_figure()
            else:
                self.anim.event_source.stop()

        self.anim = FuncAnimation(
            self.fig, update, interval=self.update_time * 1000, blit=True, repeat=False
        )
        plt.show()
        #print('hi', self.anim)

        return self.fig, self.anim

    def update_figure(self):
        self.update_core()

    def clear_all(self):
        if self.axes:
            self.axes.clear()


class PLELiveWithAnimation(LivePlotWithAnimation):
    """
    实现具体绘图逻辑的子类。
    """
    def init_core(self):
        self.line, = self.axes.plot(self.data_x, self.data_y, color='grey', alpha=1)
        self.axes.set_xlim(np.min(self.data_x), np.max(self.data_x))

    def update_core(self):
        max_data_y = np.max(self.data_y)
        if self.relim_mode == 'normal':
            if self.data_generator.points_done % len(self.data_x) == 0:
                self.ylim_max = max_data_y + 500
                self.axes.set_ylim(0, self.ylim_max)

            elif not 100 < (self.ylim_max - max_data_y) < 1000:
                self.ylim_max = max_data_y + 500
                self.axes.set_ylim(0, self.ylim_max)

        self.line.set_data(self.data_x[:self.data_generator.points_done],
                           self.data_y[:self.data_generator.points_done])
        #self.fig.canvas.draw_idle()



def ple_with_animation(wavelength_array, exposure, config_instances, repeat=1):
                
    data_x = wavelength_array
    data_y = np.zeros(len(data_x))
    data_generator = PLEAcquire(exposure = exposure, data_x=data_x, data_y=data_y, config_instances = config_instances, repeat=repeat)
    liveplot = PLELiveWithAnimation(labels=['Wavelength (nm)', f'Counts/{exposure}s'], \
                        update_time=0.1, data_generator=data_generator, data=[data_x, data_y], config_instances = config_instances)
    fig, anim = liveplot.plot()
    #data_figure = DataFigure(liveplot)
    return fig, anim

def live_with_animation(data_array, exposure, config_instances, wavelength=None, is_finite=False):
                
    data_x = data_array
    data_y = np.zeros(len(data_x))
    data_generator = LiveAcquire(exposure = exposure, data_x=data_x, data_y=data_y, \
                                 config_instances = config_instances, wavelength=wavelength, is_finite=is_finite)
    liveplot = PLELiveWithAnimation(labels=['Data', f'Counts/{exposure}s'], \
                        update_time=0.01, data_generator=data_generator, data=[data_x, data_y], config_instances = config_instances)
    fig, anim = liveplot.plot()
    #data_figure = DataFigure(liveplot)
    return fig, anim