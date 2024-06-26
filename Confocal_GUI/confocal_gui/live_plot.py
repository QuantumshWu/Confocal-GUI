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

def change_to_inline(params_type):
    get_ipython().run_line_magic('matplotlib', 'inline')
    if params_type == 'inline':
        matplotlib.rcParams.update(params_inline)
    elif params_type == 'nbagg':
        matplotlib.rcParams.update(params_nbagg)
    else:
        print('wrong params_type')

def change_to_nbagg(params_type):
    get_ipython().run_line_magic('matplotlib', 'nbagg')
    if params_type == 'inline':
        matplotlib.rcParams.update(params_inline)
    elif params_type == 'nbagg':
        matplotlib.rcParams.update(params_nbagg)
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
        
        
class LivePlotGUI():
    """
    if mode is 'PLE' or 'Live'
    labels = [xlabel, ylabel]
    data = [data_x, data_y]
    
    if mode is 'PL'
    labels = [xlabel, ylabel, zlabel]
    data = [data_x, data_y, data_z] # data_z n*m array, data_z[x, y] has coordinates (x, y)
    """
    
    def __init__(self, labels, update_time, data_generator, data, fig=None):
        
        if len(data) == 3: #PL
            self.xlabel = labels[0]
            self.ylabel = labels[1]
            self.zlabel = labels[2]
            self.data_x = data[0]
            self.data_y = data[1]
            self.data_z = data[2]
            
        else:# 'PLE' or 'Live'
            self.xlabel = labels[0]
            self.ylabel = labels[1]
            self.data_x = data[0]
            self.data_y = data[1]
        
        self.update_time = update_time
        self.data_generator = data_generator
        self.ylim_max = 100
        self.fig = fig
        if fig is None:
            self.have_init_fig = False
        else:
            self.have_init_fig = True
        
    def init_figure_and_data(self):
        change_to_nbagg(params_type = 'nbagg')
        hide_elements()
        # make sure environment enables interactive then updating figure
        
        if self.fig is None:
            self.fig = plt.figure()  # figsize
            self.axes = self.fig.add_subplot(111)
        else:
            self.axes = self.fig.axes[0]
            
        self.clear_all() #makes sure no residual artist
        self.axes.set_autoscale_on(True)
        
        self.init_core()            
        self.ylim_max = 100
        
        
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
        

        self.selector = self.choose_selector()

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
            
            
            
class PLELive(LivePlotGUI):
    
    def init_core(self):
        
        self.line, = self.axes.plot(self.data_x, self.data_y, animated=True, color='grey', alpha=0.7)
        self.axes.set_xlim(np.min(self.data_x), np.max(self.data_x))
        
    def update_core(self):
        if len(self.data_y) == 0:
            max_data_y = 0
        else:
            max_data_y = np.max(self.data_y)
        
        if not 100 < (self.ylim_max - max_data_y) < 1000:
            self.ylim_max = max_data_y + 500
            self.axes.set_ylim(0, self.ylim_max)

            self.fig.canvas.draw()
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # update ylim so need redraw

        self.fig.canvas.restore_region(self.bg_fig)
        self.line.set_ydata(self.data_y)
        
    def choose_selector(self):
        self.area = AreaSelector(self.fig.axes[0])
        self.cross = CrossSelector(self.fig.axes[0])
        self.zoom = ZoomPan(self.fig.axes[0])
        
        return [self.area, self.cross, self.zoom]
        

class PLLive(LivePlotGUI):
    
    def init_core(self):
        
        cmap_ = matplotlib.cm.get_cmap('inferno')
        cmap = cmap_.copy()
        cmap.set_under(cmap_(0))
        half_step_x = 0.5*(self.data_x[-1] - self.data_x[0])/len(self.data_x)
        half_step_y = 0.5*(self.data_y[-1] - self.data_y[0])/len(self.data_y)
        extents = [self.data_x[0]-half_step_x, self.data_x[-1]+half_step_x, \
                   self.data_y[-1]+half_step_y, self.data_y[0]-half_step_y] #left, right, bottom, up
        self.line = self.axes.imshow(self.data_z, animated=True, alpha=0.7, cmap=cmap, extent=extents)
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
        
        return [self.area, self.cross, self.zoom]

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
        self.line = self.axes.imshow(self.data_z, animated=True, alpha=0.7, cmap=cmap, extent=extents)

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
        
        return [self.area, self.cross, self.zoom, self.drag_line]
    
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
        
        return [self.area, self.cross, self.zoom, self.drag_line]
    
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
        self.gap_x = np.abs(np.max(x_data) - np.min(x_data)) / 1000
        self.gap_y = np.abs(np.max(y_data) - np.min(y_data)) / 1000
        
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
    fig : matplotlib fig object
    
    
    Examples
    --------
    >>> data_figure = DataFigure(fig)
    
    >>> data_x, data_y = data_figure.data
    
    >>> data_figure.save('my_figure')
    'save to my_figure_{time}.jpg and my_figure_{time}.txt'

    >>> data_figure.lorent(p0 = None)
    'figure with lorent curve fit'
    
    >>> data_figure.clear()
    'remove lorent fit and text'
    """
    def __init__(self, fig, selector=None):
        self.fig = fig
        self.p0 = None
        self.fit = None
        self.text = None
        self.log_info = '' # information for log output
        if selector is None:
            self.selector = []
        else:
            self.selector = selector
        self.unit = 'nm'
        
        artist = fig.axes[0].get_children()[0]
        if isinstance(artist, matplotlib.image.AxesImage):
            self.mode = 'PL'
        else:
            self.mode = 'PLE'
        
        if self.mode == 'PLE':
            self.data_x = fig.axes[0].lines[0].get_xdata()
            self.data_y = fig.axes[0].lines[0].get_ydata()
            self._data = [self.data_x, self.data_y]
        else:#PL
            self.data_array = fig.axes[0].images[0].get_array()
            self.extent = fig.axes[0].images[0].get_extent()
            self._data = [self.data_array, self.extent]
    
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, data_x_data_y):
        print('can not assign data')
        
    def save(self, addr=''):
        current_time = time.localtime()
        current_date = time.strftime("%Y-%m-%d", current_time)
        current_time_formatted = time.strftime("%H:%M:%S", current_time)
        time_str = current_date.replace('-', '_') + '_' + current_time_formatted.replace(':', '_')

        if addr=='':
            pass
        elif not os.path.exists(addr):
            os.makedirs(addr)


        self.fig.savefig(addr + self.mode + time_str + '.jpg', dpi=300)
        
        if self.mode == 'PLE':
            np.savez(addr + self.mode + time_str + '.npz', data_x = self.data_x, data_y = self.data_y)
        else:
            np.savez(addr + self.mode + time_str + '.npz', array = self.data_array, extent = self.extent)
        
        
    def lorent(self, p0=None, is_print=True, is_save=False):
        if self.mode == 'PL':
            return 
        spl = 299792458
        
        def _lorent(x, center, full_width, height, bg):
            return height*((full_width/2)**2)/((x - center)**2 + (full_width/2)**2) + bg
        
        if p0 is None:# no input
            self.p0 = [self.data_x[np.argmax(self.data_y)], 
                           (self.data_x[-1] - self.data_x[0])/4, np.max(self.data_y), 0]
        
        
        popt, pcov = curve_fit(_lorent, self.data_x, self.data_y, p0=self.p0)
        
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
            
            
def ple(wavelength_array, exposure, config_instances):
                
    data_x = wavelength_array
    data_y = np.zeros(len(data_x))
    data_generator = PLEAcquire(exposure = exposure, data_x=data_x, data_y=data_y, config_instances = config_instances)
    liveplot = PLELive(labels=['Wavelength (nm)', f'Counts/{exposure}s'], \
                        update_time=0.1, data_generator=data_generator, data=[data_x, data_y])
    fig, selector = liveplot.plot()
    data_figure = DataFigure(fig, selector)
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
                        update_time=1, data_generator=data_generator, data=[data_x, data_y, data_z])
    else:
        liveplot = PLLive(labels=['X', 'Y', f'Counts/{exposure}s'], \
                            update_time=1, data_generator=data_generator, data=[data_x, data_y, data_z])
    fig, selector = liveplot.plot()
    data_figure = DataFigure(fig, selector)
    return fig, data_figure


def live(data_array, exposure, config_instances, wavelength=None, is_finite=False):
                
    data_x = data_array
    data_y = np.zeros(len(data_x))
    data_generator = LiveAcquire(exposure = exposure, data_x=data_x, data_y=data_y, \
                                 config_instances = config_instances, wavelength=wavelength, is_finite=is_finite)
    liveplot = PLELive(labels=['Data', f'Counts/{exposure}s'], \
                        update_time=0.1, data_generator=data_generator, data=[data_x, data_y])
    fig, selector = liveplot.plot()
    data_figure = DataFigure(fig, selector)
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
                        update_time=1, data_generator=data_generator, data=[data_x, data_y, data_z], fig=fig)
    fig, selector = liveplot.plot()
    data_figure = DataFigure(fig, selector)
    return fig, data_figure