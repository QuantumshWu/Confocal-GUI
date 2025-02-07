import io
import os
import sys
import glob
import time
import threading
from decimal import Decimal
from threading import Event
import numbers
import itertools
import re
from abc import ABC, abstractmethod

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoLocator, ScalarFormatter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseEvent
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from PIL import Image as PILImage
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import warnings
from scipy.optimize import OptimizeWarning
from IPython import get_ipython
from IPython.display import display, HTML, clear_output, Javascript, Image as IPImage

import atexit


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


params_widget = {
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


def change_to_inline():
    get_ipython().run_line_magic('matplotlib', 'inline')
    matplotlib.rcParams.update(params_inline)

def change_to_widget():
    get_ipython().run_line_magic('matplotlib', 'widget')
    matplotlib.rcParams.update(params_widget)


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

def display_immediately(fig, display_id):
    # https://github.com/matplotlib/ipympl/issues/290
    # a fix for widget backend but is not needed for nbagg
    canvas = fig.canvas
    display_handle = display(canvas, display_id = display_id)
    canvas._handle_message(canvas, {'type': 'send_image_mode'}, [])
    canvas._handle_message(canvas, {'type':'refresh'}, [])
    canvas._handle_message(canvas,{'type': 'initialized'},[])
    canvas._handle_message(canvas,{'type': 'draw'},[])
    return display_handle

class LoadAcquire(threading.Thread):
    """
    class for load existing .npz data
    """
    def __init__(self, address, config_instances=None):
        super().__init__()


        loaded = np.load(address, allow_pickle=True)
        keys = loaded.files
        print("Keys in npz file:", keys)
        print(loaded['info'].item())


        self.info = loaded['info'].item()
        # important information to be saved with figures 
        self.data_x = loaded['data_x']
        self.data_y = loaded['data_y']
        if isinstance(self.data_x[0], numbers.Number):
            self.plot_type = '1D'
        else:
            self.plot_type = '2D'
            self.x_array = []
            self.y_array = []
            for x in self.data_x:
                if x[0] not in self.x_array:
                    self.x_array.append(x[0])
                if x[1] not in self.y_array:
                    self.y_array.append(x[1])    

        self.exposure = loaded['info'].item()['exposure']
        self.measurement_name = f'load_from_{address[:-4]}_'

        self.daemon = True
        self.is_running = True
        self.is_done = False
        self.points_done = len([data for data in self.data_y.flatten() if not np.isnan(data)]) 
        #how many data points have done, will control display, filter out all np.nan which should be points not done
        self.repeat_done = 0
        
    
    def run(self):
        
        self.is_done = True
        #finish all data
 
        
    def stop(self):
        if self.is_alive():
            self.is_running = False
            self.join()


class SmartOffsetFormatter(ticker.ScalarFormatter):
    # formatter to set offset and scale of y to not exceed border of fig
    # rewrite ScalarFormatter to implement this
    #
    #        def set_locs(self, locs):
    #        # docstring inherited
    #           self.locs = locs
    #           if len(self.locs) > 0:
    #               if self._useOffset:
    #                   self._compute_offset()
    #               self._set_order_of_magnitude()
    #               self._set_format()

    def __init__(self):
        super().__init__()
        self._offset_threshold = 1
        self.set_powerlimits((-2, 2))
        self._useOffset = True
        
    def _compute_offset(self):
        locs = self.locs
        # Restrict to visible ticks.
        vmin, vmax = sorted(self.axis.get_view_interval())
        locs = np.asarray(locs)
        self.locs = locs[(vmin <= locs) & (locs <= vmax)]
        self.abs_step = np.abs(self.locs[0] - self.locs[1])
        locs_step = [0.1, 0.2, 0.5, 2.5]
        for step in locs_step:
            oom = np.log10(self.abs_step/step)
            if np.abs(round(oom)-oom)<0.001:
                self.oom = round(oom)
        # autolocator gives locs with step in [0.1, 0.2, 0.5, 2.5]
        # calculate oom based on last figure

        if not len(self.locs):
            self.offset = 0
            return
        lmin, lmax = self.locs.min(), self.locs.max()
        # Only use offset if there are at least two ticks and every tick has
        # the same sign.
        if lmin == lmax or lmin <= 0 <= lmax:
            self.offset = 0
            return
        # min, max comparing absolute values (we want division to round towards
        # zero so we work on absolute values).
        abs_min, abs_max = sorted([abs(float(lmin)), abs(float(lmax))])
        sign = np.copysign(1, lmin)
        abs_mean = (abs_min + abs_max)/2

        n = self._offset_threshold # if abs_mean <= 10**(self.oom+n) then no offset
        self.offset = (sign * (abs_mean // 10 ** (self.oom+n)) * 10 ** (self.oom+n)
                       if abs_mean // 10 ** self.oom >= 10**(n-0.01)
                       else 0)

    def _set_order_of_magnitude(self):
        # if scientific notation is to be used, find the appropriate exponent
        # if using a numerical offset, find the exponent after applying the
        # offset. When lower power limit = upper <> 0, use provided exponent.
        if not self._scientific:
            self.orderOfMagnitude = 0
            return
        if self._powerlimits[0] == self._powerlimits[1] != 0:
            # fixed scaling when lower power limit = upper <> 0.
            self.orderOfMagnitude = self._powerlimits[0]
            return

        if not len(self.locs):
            self.orderOfMagnitude = 0
            return


        step_str = f'{self.abs_step:.1e}'
        step_oom = int(step_str.split('e')[-1])

        if step_oom <= self._powerlimits[0]:
            self.orderOfMagnitude = self.oom
        elif step_oom >= self._powerlimits[1]:
            self.orderOfMagnitude = self.oom
        else:
            self.orderOfMagnitude = 0

    
        
class LivePlotGUI(ABC):
    """
    if mode is 'PLE' or 'Live'
    labels = [xlabel, ylabel]
    data = [data_x, data_y]
    
    if mode is 'PL'
    labels = [xlabel, ylabel, zlabel]
    data = [data_x, data_y, data_z] # data_z n*m array, data_z[x, y] has coordinates (x, y)
    """

    display_handle = None
    old_fig = None
    display_handle_id = 0
    # convert the old fig from widget to static otherwise reopen notebook will lose these figs
    # display_id is important to keep tracking of all output areas
    
    def __init__(self, labels, update_time, data_generator, data, fig=None, config_instances=None, relim_mode='normal'):

        self.labels = labels
        

        self.xlabel = labels[0]
        self.ylabel = labels[1]
        self.data_x = data[0]
        self.data_y = data[1]
        self.points_total = len(self.data_x.flatten())
        
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
        self.repeat_done = 0
        self.selector = None
        # assign value by self.choose_selector()
        self.config_instances = config_instances
        self.relim_mode = relim_mode
        self.valid_relim_mode = ['normal', 'tight']

        self.blit_axes = []
        self.blit_artists = []
        # array contains axe and artist to be updated using blit
        self.line_colors = ['grey', 'skyblue', 'pink']

    def convert_widget_to_fig(self):
        # convert the old fig from widget to static otherwise reopen notebook will lose these figs
        if LivePlotGUI.display_handle is not None:
            # convert the old fig from widget to static otherwise reopen notebook will lose these figs
            LivePlotGUI.display_handle_id += 1
            LivePlotGUI.display_handle.update(LivePlotGUI.old_fig)
        
    def init_figure_and_data(self):
        change_to_widget()
        # make sure environment enables interactive then updating figure
        plt.ion()

        if not self.have_init_fig:

            self.convert_widget_to_fig()

            with plt.ioff():
                # avoid double display from display_immediately
                self.fig = plt.figure()

            self.fig.canvas.toolbar_visible = False
            self.fig.canvas.header_visible = False
            self.fig.canvas.footer_visible = False
            self.fig.canvas.resizable = False
            self.fig.canvas.capture_scroll = True
            LivePlotGUI.display_handle = display_immediately(self.fig, f'{LivePlotGUI.display_handle_id}')
            self.fig.canvas.layout.display = 'none'
            # set to invisble to skip the inti fig display
            self.axes = self.fig.add_subplot(111)
        else:
            self.axes = self.fig.axes[0]
            
        self.clear_all() # make sure no residual artist
        self.axes.yaxis.set_major_formatter(SmartOffsetFormatter())
        # make sure no long ticks induce cut off of label


        self.init_core() 
        self.fig.tight_layout()
        self.fig.canvas.draw()

        if not self.have_init_fig:      
            self.fig.canvas.layout.display = 'initial'
            # set fit to visible 

        self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # store bg

        
        self.data_generator.start()
        
    def update_figure(self):
        self.points_done = self.data_generator.points_done
        # update points done so new plot
        self.update_core()

        for axe, artist in zip(self.blit_axes, self.blit_artists):
            axe.draw_artist(artist)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()
        
    @abstractmethod    
    def init_core(self):
        pass
    @abstractmethod     
    def update_core(self):
        pass
    @abstractmethod     
    def choose_selector(self):
        pass
    @abstractmethod 
    def set_ylim(self):
        pass
        
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

        self.after_plot()
        LivePlotGUI.old_fig = self.fig

        return self.fig, self.selector

    def after_plot(self):
        for axe, artist in zip(self.blit_axes, self.blit_artists):
            artist.set_animated(False)

        self.axes.set_autoscale_on(False)
        self.choose_selector()

    def stop(self):
        if self.data_generator.thread.is_alive():
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
            ax.set_title('')
            
        self.fig.tight_layout()

        self.fig.canvas.draw()


    def relim(self):
        # return 1 if need redraw
        # accept relim mode 'tight' or 'normal'
        # 'tight' will relim to fit upper and lower bound
        # 'normal' will relim to fit 0 and upper bound
        
        if 0<self.points_done<self.points_total:
            max_data_y = np.nanmax(self.data_y[:self.points_done])
            min_data_y = np.nanmin(self.data_y[:self.points_done])
        else:
            max_data_y = np.nanmax(self.data_y)
            min_data_y = np.nanmin(self.data_y)

        if max_data_y == 0:
            return False
        # no new data

        if min_data_y < 0:
            self.relim_mode = 'tight'
            # change relim mode if not able to keep 'normal' relim

        if self.relim_mode == 'normal':
            data_range = max_data_y - 0
        elif self.relim_mode == 'tight':
            data_range = max_data_y - min_data_y

        if self.relim_mode == 'normal':

            if 0<=(self.ylim_max-max_data_y)<=0.3*data_range:
                return False

            self.ylim_min = 0
            self.ylim_max = max_data_y*1.2


            self.set_ylim()

            return True

        elif self.relim_mode == 'tight':

            if 0<=(self.ylim_max - max_data_y)<=0.2*data_range and 0<=(min_data_y - self.ylim_min)<=0.2*data_range:
                return False

            self.ylim_min = min_data_y - 0.1*data_range
            self.ylim_max = max_data_y + 0.1*data_range

            if self.ylim_min!=self.ylim_max:
                self.set_ylim()

            return True


           
            
class PLELive(LivePlotGUI):
    
    def init_core(self):
        self.lines = self.axes.plot(self.data_x, self.data_y, animated=True, alpha=1)
        for i, line in enumerate(self.lines):
            line.set_color(self.line_colors[i])
            self.blit_axes.append(self.axes)
            self.blit_artists.append(line)

        self.axes.set_xlim(np.nanmin(self.data_x), np.nanmax(self.data_x))
        self.axes.set_ylim(self.ylim_min, self.ylim_max)

        self.axes.set_ylabel(self.ylabel + ' x1')
        self.axes.set_xlabel(self.xlabel)
        
    def update_core(self):
        
        if (self.repeat_done!=self.data_generator.repeat_done):
            self.ylabel = self.labels[1] + f' x{self.data_generator.repeat_done + 1}'
            self.repeat_done = self.data_generator.repeat_done
            self.axes.set_ylabel(self.ylabel)

            self.relim()
            self.fig.canvas.draw()
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # update ylim so need redraw

        else:

            if self.relim():
                self.fig.canvas.draw()
                self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # update ylim so need redraw

            
        self.fig.canvas.restore_region(self.bg_fig)
        for i, line in enumerate(self.lines):
            line.set_data(self.data_x, self.data_y[:, i])

    def set_ylim(self):
        self.axes.set_ylim(self.ylim_min, self.ylim_max)
        
    def choose_selector(self):
        self.area = AreaSelector(self.fig.axes[0])
        self.cross = CrossSelector(self.fig.axes[0])
        self.zoom = ZoomPan(self.fig.axes[0])
        
        self.selector = [self.area, self.cross, self.zoom]


class LiveAndDisLive(LivePlotGUI):
    # live_plot class for realizing live plot plus distribution of counts
    # default live_plot class for live() function
    
    def init_core(self):
        warnings.filterwarnings("ignore", category=OptimizeWarning)

        self.lines = self.axes.plot(self.data_x, self.data_y, animated=True, alpha=1)
        for i, line in enumerate(self.lines):
            line.set_color(self.line_colors[i])
            self.blit_axes.append(self.axes)
            self.blit_artists.append(line)

        self.axes.set_xlim(np.nanmin(self.data_x), np.nanmax(self.data_x))
        self.axes.set_ylim(self.ylim_min, self.ylim_max)

        self.axes.set_ylabel(self.ylabel + ' x1')
        self.axes.set_xlabel(self.xlabel)


        if self.have_init_fig:
            # such as in GUI

            self.axdis = self.fig.axes[1]
            self.axdis.xaxis.set_major_locator(AutoLocator())
            self.axdis.xaxis.set_major_formatter(ScalarFormatter())
            self.axdis.relim()
            self.axdis.autoscale_view()
            self.axdis.tick_params(axis='y', labelleft=False)
            self.axdis.tick_params(axis='both', which='both',bottom=False,top=False)

        else:

            divider = make_axes_locatable(self.axes)        
            self.axdis = divider.append_axes("right", size="20%", pad=0.1, sharey=self.axes)
            self.axdis.xaxis.set_major_locator(AutoLocator())
            self.axdis.xaxis.set_major_formatter(ScalarFormatter())
            self.axdis.relim()
            self.axdis.autoscale_view()
            self.axdis.tick_params(axis='y', labelleft=False)
            self.axdis.tick_params(axis='both', which='both',bottom=False,top=False)
            # reset axdis ticks, labels
            self.fig.tight_layout()
            self.fig.canvas.draw()


        self.counts_max = 10
        # filter out zero data
        self.n_bins = np.min((self.points_total//4, 100))
        self.n, self.bins, self.patches = self.axdis.hist(self.data_y[:self.points_done, 0], orientation='horizontal',\
             bins=self.n_bins, color='grey', range=(self.ylim_min, self.ylim_max))
        self.axdis.set_xlim(0, self.counts_max)

        for patch in self.patches:
            patch.set_animated(True)
            self.blit_axes.append(self.axdis)
            self.blit_artists.append(patch)


        self.poisson_fit_line, = self.axdis.plot(self.data_y[:, 0], [0 for data in self.data_y], color='orange', animated=True, alpha=1)
        self.blit_axes.append(self.axdis)
        self.blit_artists.append(self.poisson_fit_line)

        self.points_done_fits = self.points_done
        self.ylim_min_dis = self.ylim_min
        self.ylim_max_dis = self.ylim_max


    @staticmethod
    def _gauss_func(x, A, mu, sigma):
        return A * np.exp(- (x - mu)**2 / (2.0 * sigma**2))


    def update_fit(self):
        # update fitting

        if not (self.points_done - self.points_done_fits)>=10:
            return
        else:
            self.points_done_fits = self.points_done

        mask = self.n > 0
        bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        bin_centers_fit = bin_centers[mask]
        counts_fit = self.n[mask]

        
        if len(bin_centers_fit) == 0:
            popt = None
        else:
            try:
                popt, pcov = curve_fit(
                    self._gauss_func,
                    bin_centers_fit,
                    counts_fit,
                    p0=[np.max(counts_fit), np.mean(bin_centers_fit), (np.max(bin_centers_fit)-np.min(bin_centers_fit))/4],
                    bounds=([0, np.min(bin_centers_fit), (np.max(bin_centers_fit)-np.min(bin_centers_fit))/10], \
                            [np.max(counts_fit)*4, np.max(bin_centers_fit), (np.max(bin_centers_fit)-np.min(bin_centers_fit))*10])
                )
            except Exception as e:
                popt = None

        if popt is not None:
            x_fit = np.sort(np.append(np.linspace(self.ylim_min, self.ylim_max, 100), bin_centers))
            y_fit = self._gauss_func(x_fit, *popt)
            
            if hasattr(self, 'poisson_fit_line') and self.poisson_fit_line is not None:
                self.poisson_fit_line.set_data(y_fit, x_fit)

            if popt[1]<=0:
                ratio = 0
            else:
                ratio = popt[2]/np.sqrt(popt[1])

            result = f'$\\sigma$={ratio:.2f}$\\sqrt{{\\mu}}$'
            if not hasattr(self, 'text'):
                self.text = self.axdis.text(0.5, 1.01, 
                                                  result, transform=self.axdis.transAxes, 
                                                  color='orange', ha='center', va='bottom', animated=True)
                self.blit_artists.append(self.text)
                self.blit_axes.append(self.axdis)
            else:
                self.text.set_text(result)

        
    def update_core(self):

        new_counts, _ = np.histogram(self.data_y[:self.points_done, 0], bins=self.bins)
        bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        
        if self.relim() or (self.repeat_done!=self.data_generator.repeat_done) or (np.max(new_counts) > self.counts_max):

            self.ylabel = self.labels[1] + f' x{self.data_generator.repeat_done + 1}'
            self.repeat_done = self.data_generator.repeat_done
            self.axes.set_ylabel(self.ylabel)
            self.relim_dis()
            self.update_dis()
            self.counts_max = np.max((np.max(self.n) + 5, int(np.max(self.n)*1.5)))
            self.axdis.set_xlim(0, self.counts_max)

            self.fig.canvas.draw()
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # update ylim so need redraw

        elif self.relim_dis():

            self.update_dis()
            self.counts_max = np.max((np.max(self.n) + 5, int(np.max(self.n)*1.5)))
            self.axdis.set_xlim(0, self.counts_max)

            self.fig.canvas.draw()
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # update ylim so need redraw

            
        self.fig.canvas.restore_region(self.bg_fig)
        for i, line in enumerate(self.lines):
            line.set_data(self.data_x, self.data_y[:, i])
 
        self.update_dis()
        self.update_fit()

    def update_dis(self):
        counts, bins = np.histogram(self.data_y[:self.points_done, 0],
                                    bins=self.n_bins, range=(self.ylim_min_dis, self.ylim_max_dis))
        self.n = counts
        self.bins = bins

        for i, patch in enumerate(self.patches):
            # bins[i] ~ bins[i+1] y
            y0 = bins[i]
            dy = bins[i+1] - bins[i]

            # x 0 to counts[i]
            x0 = 0
            dx = counts[i]

            patch.set_x(x0)
            patch.set_width(dx)
            patch.set_y(y0)
            patch.set_height(dy)


    def relim_dis(self):
        # return 1 if need redraw, only calculate relim of main data (self.data_y[:, 0])
        
        if 0<self.points_done<self.points_total:
            max_data_y = np.nanmax(self.data_y[:self.points_done, 0])
            min_data_y = np.nanmin(self.data_y[:self.points_done, 0])
        else:
            max_data_y = np.nanmax(self.data_y[:, 0])
            min_data_y = np.nanmin(self.data_y[:, 0])

        if max_data_y == 0:
            return False
        # no new data

        data_range = max_data_y - min_data_y


        if 0<=(self.ylim_max_dis - max_data_y)<=0.2*data_range and 0<=(min_data_y - self.ylim_min_dis)<=0.2*data_range:
            return False

        self.ylim_min_dis = min_data_y - 0.1*data_range
        self.ylim_max_dis = max_data_y + 0.1*data_range

        return True



    def set_ylim(self):        
        self.axes.set_ylim(self.ylim_min, self.ylim_max)
        self.axdis.set_ylim(self.ylim_min, self.ylim_max)

        
    def choose_selector(self):
        self.area = AreaSelector(self.fig.axes[0])
        self.cross = CrossSelector(self.fig.axes[0])
        self.zoom = ZoomPan(self.fig.axes[0])
        
        self.selector = [self.area, self.cross, self.zoom]
        

class PLLive(LivePlotGUI):
    
    def init_core(self):

        self.axes.yaxis.set_major_formatter(ScalarFormatter())
        self.axes.xaxis.set_major_formatter(ScalarFormatter())

        try:
            self.x_array = self.data_generator.x_array
            self.y_array = self.data_generator.y_array
        except:
            raise KeyError('Data generator has no x_array, y_array')
        self.data_shape = (len(self.y_array), len(self.x_array))

        cmap_ = matplotlib.cm.get_cmap('inferno')
        cmap = cmap_.copy()
        cmap.set_bad(cmap_(0))
        half_step_x = 0.5*(self.x_array[-1] - self.x_array[0])/len(self.x_array)
        half_step_y = 0.5*(self.y_array[-1] - self.y_array[0])/len(self.y_array)
        extents = [self.x_array[0]-half_step_x, self.x_array[-1]+half_step_x, \
                   self.y_array[-1]+half_step_y, self.y_array[0]-half_step_y] #left, right, bottom, up


        self.line = self.axes.imshow(self.data_y.reshape(self.data_shape), animated=True, alpha=1, cmap=cmap, extent=extents)
        divider = make_axes_locatable(self.axes)
        self.cax = divider.append_axes("right", size="5%", pad=0.15)
        self.cbar = self.fig.colorbar(self.line, cax = self.cax)

        self.fig.axes[0].set_xlim((extents[0], extents[1]))
        self.fig.axes[0].set_ylim((extents[2], extents[3]))


        self.cbar.set_label(self.ylabel + ' x1')
        self.axes.set_ylabel(self.xlabel[1])
        self.axes.set_xlabel(self.xlabel[0])

        self.blit_axes.append(self.axes)
        self.blit_artists.append(self.line)


        
    def update_core(self):

        if self.relim() or (self.repeat_done!=self.data_generator.repeat_done):

            self.ylabel = self.labels[1] + f' x{self.data_generator.repeat_done + 1}'
            self.repeat_done = self.data_generator.repeat_done
            self.cbar.set_label(self.ylabel)
            self.fig.canvas.draw() 
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        self.fig.canvas.restore_region(self.bg_fig)

        self.line.set_array(self.data_y.reshape(self.data_shape))
        # other data just np.nan and controlled by set_bad
        

    def set_ylim(self):
        self.line.set_clim(vmin=self.ylim_min, vmax=self.ylim_max)
        
    def choose_selector(self):

        self.area = AreaSelector(self.fig.axes[0])
        self.cross = CrossSelector(self.fig.axes[0])
        self.zoom = ZoomPan(self.fig.axes[0])
        


        self.fig.canvas.draw()        
        self.selector = [self.area, self.cross, self.zoom]



class PLDisLive(LivePlotGUI):

    def init_core(self):

        self.axes.yaxis.set_major_formatter(ScalarFormatter())
        self.axes.xaxis.set_major_formatter(ScalarFormatter())

        try:
            self.x_array = self.data_generator.x_array
            self.y_array = self.data_generator.y_array
        except:
            raise KeyError('Data generator has no x_array, y_array')
        self.data_shape = (len(self.y_array), len(self.x_array))

        cmap_ = matplotlib.cm.get_cmap('inferno')
        cmap = cmap_.copy()
        cmap.set_bad(cmap_(0))
        half_step_x = 0.5*(self.x_array[-1] - self.x_array[0])/len(self.x_array)
        half_step_y = 0.5*(self.y_array[-1] - self.y_array[0])/len(self.y_array)
        extents = [self.x_array[0]-half_step_x, self.x_array[-1]+half_step_x, \
                   self.y_array[-1]+half_step_y, self.y_array[0]-half_step_y] #left, right, bottom, up

        if self.have_init_fig:
            # such as in GUI
            self.cbar = self.fig.axes[0].images[0].colorbar
            self.axright = self.fig.axes[1]
            self.cax = self.fig.axes[2]

            self.line = self.fig.axes[0].images[0]
            self.line.set(cmap = cmap, extent=extents)
            self.line.set_array(self.data_y.reshape(self.data_shape))
            self.line.set_animated(True)
            self.cbar.update_normal(self.line)

            self.fig.axes[0].set_xlim((extents[0], extents[1]))
            self.fig.axes[0].set_ylim((extents[2], extents[3]))

        else:

            width, height = self.fig.get_size_inches()
            divider = make_axes_locatable(self.axes)        
            self.cax = divider.append_axes("right", size="5%", pad=0.15)
            self.axright = divider.append_axes("top", size="20%", pad=0.25)
            self.fig.set_size_inches(width, height*1.25)
            self.line = self.axes.imshow(np.zeros(self.data_shape), animated=True, alpha=1, cmap=cmap, extent=extents)
            self.fig.tight_layout()
            self.fig.canvas.draw()

            self.cbar = self.fig.colorbar(self.line, cax = self.cax)
            self.fig.axes[0].set_xlim((extents[0], extents[1]))
            self.fig.axes[0].set_ylim((extents[2], extents[3]))

        self.counts_max = 10
        # filter out zero data
        self.axright.xaxis.set_major_locator(AutoLocator())
        self.axright.xaxis.set_major_formatter(ScalarFormatter())
        self.axright.relim()
        self.axright.autoscale_view()
        # reset axright ticks, labels
        self.n_bins = np.min((self.points_total//4, 100))
        self.n, self.bins, self.patches = self.axright.hist(self.data_y[:self.points_done], orientation='vertical',\
             bins=self.n_bins, color='grey', range=(self.ylim_min, self.ylim_max))
        self.axright.set_ylim(0, self.counts_max)

        self.blit_axes.append(self.axes)
        self.blit_artists.append(self.line)
        for patch in self.patches:
            patch.set_animated(True)
            self.blit_axes.append(self.axright)
            self.blit_artists.append(patch)

        self.cbar.set_label(self.ylabel + ' x1')
        self.axes.set_ylabel(self.xlabel[1])
        self.axes.set_xlabel(self.xlabel[0])

        self.axright.tick_params(axis='both', which='both',bottom=False,top=False)

        
    def update_core(self):
        new_counts, _ = np.histogram(self.data_y[:self.points_done], bins=self.bins)
        self.n[:] = new_counts 

        if self.relim() or (np.max(self.n) > self.counts_max) or (self.repeat_done!=self.data_generator.repeat_done):
            self.counts_max = np.max((np.max(self.n) + 5, int(np.max(self.n)*1.5)))
            self.axright.set_ylim(0, self.counts_max)

            self.ylabel = self.labels[1] + f' x{self.data_generator.repeat_done + 1}'
            self.repeat_done = self.data_generator.repeat_done
            self.cbar.set_label(self.ylabel)

            self.fig.canvas.draw() 
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        self.fig.canvas.restore_region(self.bg_fig)

        self.line.set_array(self.data_y.reshape(self.data_shape))
        # other data just np.nan and controlled by set_bad
   
        for count, patch in zip(self.n, self.patches):
            patch.set_height(count)        

    def set_ylim(self):
        self.line.set_clim(vmin=self.ylim_min, vmax=self.ylim_max)
        self.n, self.bins, self.patches = self.axright.hist(self.data_y[:self.points_done], orientation='vertical',\
             bins=self.n_bins, color='grey', range=(self.ylim_min, self.ylim_max))
        self.axright.set_xlim(self.ylim_min, self.ylim_max)

        self.blit_artists[:] = self.blit_artists[:-self.n_bins]
        for patch in self.patches:
            patch.set_animated(True) 
            self.blit_artists.append(patch)

        
    def choose_selector(self):

        self.area = AreaSelector(self.fig.axes[0])
        self.cross = CrossSelector(self.fig.axes[0])
        self.zoom = ZoomPan(self.fig.axes[0])
        
        cmap = self.axes.images[0].colorbar.mappable.get_cmap()

        y_min = np.nanmin(self.data_y[:self.points_done])
        y_max = np.nanmax(self.data_y[:self.points_done])
        self.line_min = self.axright.axvline(y_min, color='red', linewidth=6, alpha=0.3)
        self.line_max = self.axright.axvline(y_max, color='red', linewidth=6, alpha=0.3)

        self.line_l = self.axright.axvline(self.ylim_min, color=cmap(0), linewidth=6)
        self.line_h = self.axright.axvline(self.ylim_max, color=cmap(0.95), linewidth=6)

        self.axright.set_xticks([y_min, y_max])
        self.axright.set_xticklabels([f'{xtick:.0f}' for xtick in [y_min, y_max]])

        self.drag_line = DragVLine(self.line_l, self.line_h, self.update_clim, self.axright)
        self.fig.canvas.draw()
        # must be here to display self.line_l etc. after plot done, don't know why?
        
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
        x1, x2, y1, y2 = self.selector.extents
        # changed by rectangleselector
        
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
        
        self.cid_press = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)

    def on_press(self, event):
        if event.inaxes == self.ax:                
            if event.button == 3:  # mouse right key
                current_time = time.time()
                if (self.last_click_time is None) or (current_time - self.last_click_time) > 0.3:
                    self.last_click_time = current_time
                else:
                    self.last_click_time = None
                    self.remove_point()
                    self.ax.figure.canvas.draw()
                    return
                    
                x_data = self.ax.lines[0].get_xdata()
                y_data = self.ax.lines[0].get_ydata()
                self.gap_x = np.abs(np.max(x_data) - np.min(x_data)) / 1000 if (len(x_data)>0) else 0.01
                self.gap_y = np.abs(np.max(y_data) - np.min(y_data)) / 1000 if (len(y_data)>0) else 0.01

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
                    self.vline.set_xdata([x, x])
                    self.hline.set_ydata([y, y])
                    self.point.set_xdata([x, ])
                    self.point.set_ydata([y, ])
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
        self.ax.draw_artist(self.line_l)
        self.ax.draw_artist(self.line_h)
        #self.ax.figure.canvas.blit(self.ax.bbox)

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


def dummy_cross(ax, x, y):

    def data_to_display(ax, xdata, ydata):
        return ax.transData.transform((xdata, ydata))

    x_disp, y_disp = data_to_display(ax, x, y)

    press_event = MouseEvent(name='button_press_event',
                         canvas=ax.figure.canvas,
                         x=x_disp, y=y_disp,
                         button=3)  
    press_event.inaxes = ax  
    ax.figure.canvas.callbacks.process('button_press_event', press_event)

    time.sleep(0.31)

    ax.figure.canvas.callbacks.process('button_press_event', press_event)



def dummy_area(ax, x1, y1, x2, y2):

    x1_disp, y1_disp = ax.transData.transform((x1, y1))
    x2_disp, y2_disp = ax.transData.transform((x2, y2))

    press_event = MouseEvent('button_press_event', ax.figure.canvas, 0, 0, button=1)
    press_event.inaxes = ax
    ax.figure.canvas.callbacks.process('button_press_event', press_event)
    release_event = MouseEvent('button_release_event', ax.figure.canvas, 0, 0, button=1)
    release_event.inaxes = ax
    ax.figure.canvas.callbacks.process('button_release_event', release_event)
    # close existing rectangle, otherwise bug
    press_event = MouseEvent('button_press_event', ax.figure.canvas, x1_disp, y1_disp, button=1)
    press_event.inaxes = ax
    ax.figure.canvas.callbacks.process('button_press_event', press_event)
    motion_event = MouseEvent('motion_notify_event', ax.figure.canvas, x2_disp, y2_disp, button=1)
    motion_event.inaxes = ax
    ax.figure.canvas.callbacks.process('motion_notify_event', motion_event)
    release_event = MouseEvent('button_release_event', ax.figure.canvas, x2_disp, y2_disp, button=1)
    release_event.inaxes = ax
    ax.figure.canvas.callbacks.process('button_release_event', release_event)

            


valid_fit_func = ['lorent', 'decay', 'rabi']         
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
    def __init__(self, live_plot, address=None, fig=None, relim_mode='tight'):

        if address is None:
            self.fig = live_plot.fig
            self.selector = live_plot.selector
            self.info = live_plot.data_generator.info
            self.live_plot = live_plot
            # load all necessary info defined in device.info for device in config_instances\
            self.plot_type = live_plot.data_generator.plot_type
            self.measurement_name = live_plot.data_generator.measurement_name


            self.data_x = live_plot.data_generator.data_x
            self.data_y = live_plot.data_generator.data_y
            # only get data acquired from data_generator

        else:
            if '*' in address:
                self.is_error = False
                files_all = glob.glob(address)# includes .jpg, .npz etc.
                files = []
                for file in files_all:
                    if '.npz' in file:
                        files.append(file)
                if len(files) > 1:
                    print(files)
                if len(files) == 0:
                    self.is_error = True
                    return

                data_generator = LoadAcquire(address=files[0])
            else:
                data_generator = LoadAcquire(address=address)
            exposure = data_generator.info.get('exposure')
            self.measurement_name = data_generator.measurement_name
            self.plot_type = data_generator.plot_type
            if self.plot_type == '1D':
                self.data_x = data_generator.data_x
                self.data_y = data_generator.data_y
                x_label = data_generator.info.get('x_label', 'Data (1)')
                y_label = data_generator.info.get('y_label', f'Counts/{exposure}s x1')

                _live_plot = PLELive(labels=[x_label, y_label[:-3]], \
                                    update_time=0.1, data_generator=data_generator, data=[self.data_x, self.data_y], \
                                    config_instances = None, relim_mode = relim_mode, fig=fig)

            else:
                self.data_x = data_generator.data_x
                self.data_y = data_generator.data_y
                x_label = data_generator.info.get('x_label', ['X', 'Y'])
                y_label = data_generator.info.get('y_label', f'Counts/{exposure}s x1')

                _live_plot = PLDisLive(labels=[x_label, y_label[:-3]], \
                        update_time=1, data_generator=data_generator, data=[self.data_x, self.data_y], \
                        config_instances = None, relim_mode = relim_mode, fig=fig)

            fig, selector = _live_plot.plot()
            self.fig = _live_plot.fig
            self.selector = _live_plot.selector
            self.info = _live_plot.data_generator.info
            self.live_plot = _live_plot
            # load all necessary info defined in device.info for device in config_instances


        self.p0 = None
        self.fit = None
        self.fit_func = None
        self.text = None
        self.log_info = '' # information for log output
        if self.plot_type == '1D':
            x_label = self.live_plot.xlabel
            pattern = r'\((.+)\)$'
            match = re.search(pattern, x_label)
            if match:
                self.unit = match.group(1)
            else:
                self.unit = '1'
        else:
            self.unit = '1'

        warnings.filterwarnings("ignore", category=OptimizeWarning)
        

    def xlim(self, x_min, x_max):
        self.fig.axes[0].set_xlim(x_min, x_max)

    def ylim(self, y_min, y_max):
        self.fig.axes[0].set_ylim(y_min, y_max)

    def close_selector(self):
        for selector in self.selector:
            selector.set_active(False)

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


        self.fig.savefig(addr + self.measurement_name + time_str + '.jpg', dpi=300)

        if extra_info is None:
            extra_info = {}

        
        if self.plot_type == '1D':
            x_label = self.fig.axes[0].get_xlabel()
            y_label = self.fig.axes[0].get_ylabel()
            np.savez(addr + self.measurement_name + time_str + '.npz', data_x = self.data_x, data_y = self.data_y, \
                info = {**self.info, **extra_info, **{'x_label':x_label, 'y_label':y_label}})
        else:
            x_label = self.fig.axes[0].get_xlabel()
            y_label = self.fig.axes[0].get_ylabel()
            z_label = self.fig.axes[0].images[0].colorbar.ax.yaxis.label.get_text()
            np.savez(addr + self.measurement_name + time_str + '.npz', data_x = self.data_x, data_y = self.data_y, \
                info = {**self.info, **extra_info, **{'x_label':[x_label, y_label], 'y_label':z_label}})

        print(f'saved fig as {addr}{self.measurement_name}{time_str}.npz')


    def _display_popt(self, popt, popt_str, popt_pos):
        # popt_str = ['amplitude', 'offset', 'omega', 'decay', 'phi'], popt_pos = 'lower left' etc
        valid_pos = ['upper left', 'upper right', 'lower left', 'lower right']
        if popt_pos not in valid_pos:
            print(f'wrong popt pos, can only be {valid_pos}')
        _popt = popt
        formatted_popt = [f'{x:.5f}'.rstrip('0') for x in _popt]
        result_list = [f'{name} = {value}' for name, value in zip(popt_str, formatted_popt)]
        formatted_popt_str = '\n'.join(result_list)
        result = f'{formatted_popt_str}'
        self.log_info = result
        # format popt to display as text
                    
        if self.text is None:
            if popt_pos=='upper left':
                self.text = self.fig.axes[0].text(0.025, 1-0.025, 
                                                  result, transform=self.fig.axes[0].transAxes, 
                                                  color='red', ha='left', va='top')
            elif popt_pos == 'upper right':
                self.text = self.fig.axes[0].text(1-0.025, 1-0.025, 
                                                  result, transform=self.fig.axes[0].transAxes, 
                                                  color='red', ha='right', va='top')

            elif popt_pos == 'lower left':
                self.text = self.fig.axes[0].text(0.025, 0.025, 
                                                  result, transform=self.fig.axes[0].transAxes, 
                                                  color='red', ha='left', va='bottom')

            elif popt_pos == 'lower right':
                self.text = self.fig.axes[0].text(1-0.025, 0.025, 
                                                  result, transform=self.fig.axes[0].transAxes, 
                                                  color='red', ha='right', va='bottom')

        else:
            self.text.set_text(result)
            
        self.fig.canvas.draw()

    def _select_fit(self, min_num=2):
        # return data in the area selector, and only return first set if there are multiple sets of data (only data not data_ref)
        valid_index = [i for i, data in enumerate(self.data_y) if not np.isnan(data[0])]
        # index of none np.nan data
        if self.selector[0] is None:
            return self.data_x[valid_index], self.data_y[valid_index, 0]
        else:
            xl, xh, yl, yh = self.selector[0].range
            if (xl is None) or (xh is None):
                return self.data_x[valid_index], self.data_y[valid_index, 0]
            if (xl - xh)==0:
                return self.data_x[valid_index], self.data_y[valid_index, 0]

            index_l = np.argmin(np.abs(self.data_x[valid_index] - xl))
            index_h = np.argmin(np.abs(self.data_x[valid_index] - xh))
            index_l, index_h = np.sort([index_l, index_h])
            # in order to handle data_x from max to min (e.g. GHz unit)
            if np.abs(index_l - index_h)<=min_num:
                return self.data_x[valid_index], self.data_y[valid_index, 0]
            return self.data_x[valid_index][index_l:index_h], self.data_y[valid_index, 0][index_l:index_h]


    def lorent(self, p0=None, is_display=True, is_fit=True):
        if self.plot_type == '2D':
            return 
        self.data_x_p, self.data_y_p = self._select_fit(min_num=4)
        # use the area selector results for fitting , min_num should at least be number of fitting parameters
        spl = 299792458
        
        def _lorent(x, center, full_width, height, bg):
            return height*((full_width/2)**2)/((x - center)**2 + (full_width/2)**2) + bg
        
        if p0 is None:# no input

            # likely be positive height
            guess_center = self.data_x_p[np.argmax(self.data_y_p)]
            guess_height = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))
            guess_bg = np.min(self.data_y_p)
            guess_full_width = np.abs(self.data_x_p[0]-self.data_x_p[-1])/4
            self.p0_pos = [guess_center, guess_full_width, guess_height, guess_bg]

            # likely be negtive height
            guess_center = self.data_x_p[np.argmin(self.data_y_p)]
            guess_height = -np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))
            guess_bg = np.max(self.data_y_p)
            guess_full_width = np.abs(self.data_x_p[0]-self.data_x_p[-1])/4
            self.p0_neg = [guess_center, guess_full_width, guess_height, guess_bg]

        else:
            self.p0 = p0
            guess_center = self.p0[0]
            guess_full_width = self.p0[1]
            guess_height = self.p0[2]
            guess_bg = self.p0[3]

        data_x_range = np.abs(self.data_x_p[-1] - self.data_x_p[0])
        data_y_range = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))
        self.bounds = ([guess_center - data_x_range, guess_full_width/10, -10*data_y_range, -10*data_y_range], \
        [guess_center + data_x_range, guess_full_width*10, 10*data_y_range, 10*data_y_range])
        
        if is_fit and (p0 is None):
            try:
                popt_pos, pcov_pos = curve_fit(_lorent, self.data_x_p, self.data_y_p, p0=self.p0_pos)
                popt_neg, pcov_neg = curve_fit(_lorent, self.data_x_p, self.data_y_p, p0=self.p0_neg)
            except:
                return
            loss_pos = np.sum((_lorent(self.data_x_p, *popt_pos) - self.data_y_p)**2)
            loss_neg = np.sum((_lorent(self.data_x_p, *popt_neg) - self.data_y_p)**2)

            if loss_pos < loss_neg:
                popt, pcov = popt_pos, pcov_pos
            else:
                popt, pcov = popt_neg, pcov_neg

        elif is_fit and (p0 is not None):
            try:
                popt, pcov = curve_fit(_lorent, self.data_x_p, self.data_y_p, p0=self.p0)
            except:
                return

        else:
            popt, pcov = self.p0, None
        
        
        if self.fit is None:
            self.fit = self.fig.axes[0].plot(self.data_x, _lorent(self.data_x, *popt), color='orange', linestyle='--')
        else:
            self.fit[0].set_ydata(_lorent(self.data_x, *popt))
        self.fig.canvas.draw()
        
        popt_str = ['center', 'FWHM', 'height', 'bg']
        if is_display:
            if popt[2] > 0:
                self._display_popt(popt, popt_str, 'upper right')
            else:
                self._display_popt(popt, popt_str, 'lower right')

        self.fit_func = 'lorent'
        return [popt_str, pcov], popt


    def lorent_zeeman(self, p0=None, is_display=True, func=None, bounds = None, is_fit=True):
        #fit of PLE under B field, will rewrite soon
        if self.plot_type == '2D':
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
            popt, pcov = curve_fit(_func, self.data_x_p, self.data_y_p, p0=self.p0, bounds = self.bounds)
        else:
            popt, pcov = self.p0, None
        
        
        if self.fit is None:
            self.fit = self.fig.axes[0].plot(self.data_x, _func(self.data_x, *popt), color='orange', linestyle='--')
        else:
            self.fit[0].set_ydata(_func(self.data_x, *popt))
        self.fig.canvas.draw()
        
        if is_display:
            

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
            

        return [popt_str, pcov], _popt


    def rabi(self, p0=None, is_display=True, is_fit=True):
        if self.plot_type == '2D':
            return 
        self.data_x_p, self.data_y_p = self._select_fit(min_num=5)
        
        def _rabi(x, amplitude, offset, omega, decay, phi):
            return amplitude*np.sin(2*np.pi*omega*x + phi)*np.exp(-x/decay) + offset
        
        if p0 is None:# no input
            guess_amplitude = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))/2
            guess_offset = np.mean(self.data_y_p)
            guess_omega = 0.5/np.abs(self.data_x_p[np.argmin(self.data_y_p)] - self.data_x_p[np.argmax(self.data_y_p)])
            guess_decay = np.abs(1/((1 - (np.abs(np.min(self.data_y_p) - guess_offset)/np.abs(np.max(self.data_y_p) - guess_offset)))*2*guess_omega))
            guess_phi = np.pi/2

            data_y_range = np.max(self.data_y_p) - np.min(self.data_y_p)

            self.p0 = [guess_amplitude, guess_offset, guess_omega, guess_decay, guess_phi]

        else:
            self.p0 = p0

            guess_amplitude = self.p0[0]
            guess_offset = self.p0[1]
            guess_omega = self.p0[2]
            guess_decay = self.p0[3]
            guess_phi = self.p0[4]

            data_y_range = np.max(self.data_y_p) - np.min(self.data_y_p)

        self.bounds = ([guess_amplitude/3, guess_offset - data_y_range, guess_omega*0.5, guess_decay/5, guess_phi - np.pi/10], \
            [guess_amplitude*3, guess_offset + data_y_range, guess_omega*1.1, guess_decay*5, guess_phi + np.pi/10])
        
        if is_fit:
            popt, pcov = curve_fit(_rabi, self.data_x_p, self.data_y_p, p0=self.p0, bounds = self.bounds)
        else:
            popt, pcov = self.p0, None
        
        if self.fit is None:
            self.fit = self.fig.axes[0].plot(self.data_x, _rabi(self.data_x, *popt), color='orange', linestyle='--')
        else:
            self.fit[0].set_ydata(_rabi(self.data_x, *popt))
        self.fig.canvas.draw()
        
        popt_str = ['amplitude', 'offset', 'omega', 'decay', 'phi']
        if is_display:
            self._display_popt(popt, popt_str, 'upper right')
            
        self.fit_func = 'rabi'
        return [popt_str, pcov], popt


    def decay(self, p0=None, is_display=True, is_fit=True):
        if self.plot_type == '2D':
            return 
        self.data_x_p, self.data_y_p = self._select_fit(min_num=3)
        
        def _exp_decay(x, amplitude, offset, decay):
            return amplitude*np.exp(-x/decay) + offset
        
        if p0 is None:# no input

            # if positive
            guess_amplitude = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))
            guess_offset = np.mean(self.data_y_p)
            guess_decay = np.abs(self.data_x_p[np.argmin(self.data_y_p)] - self.data_x_p[np.argmax(self.data_y_p)])/2
            data_y_range = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))

            self.p0_pos = [guess_amplitude, guess_offset, guess_decay]
            # if negtive
            self.p0_neg = [-guess_amplitude, guess_offset, guess_decay]

        else:
            self.p0 = p0

            guess_amplitude = self.p0[0]
            guess_offset = self.p0[1]
            guess_decay = self.p0[2]

            data_y_range = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))

            
        self.bounds = ([-4*data_y_range, guess_offset - data_y_range, guess_decay/10], \
            [4*data_y_range, guess_offset + data_y_range, guess_decay*10])
        
        if is_fit and (p0 is None):
            popt_pos, pcov_pos = curve_fit(_exp_decay, self.data_x_p, self.data_y_p, p0=self.p0_pos)
            popt_neg, pcov_neg = curve_fit(_exp_decay, self.data_x_p, self.data_y_p, p0=self.p0_neg)
            loss_pos = np.sum((_exp_decay(self.data_x_p, *popt_pos) - self.data_y_p)**2)
            loss_neg = np.sum((_exp_decay(self.data_x_p, *popt_neg) - self.data_y_p)**2)

            if loss_pos < loss_neg:
                popt, pcov = popt_pos, pcov_pos
            else:
                popt, pcov = popt_neg, pcov_neg

        elif is_fit and (p0 is not None):

            popt, pcov = curve_fit(_exp_decay, self.data_x_p, self.data_y_p, p0=self.p0)

        else:
            popt, pcov = self.p0, None
        
        
        if self.fit is None:
            self.fit = self.fig.axes[0].plot(self.data_x, _exp_decay(self.data_x, *popt), color='orange', linestyle='--')
        else:
            self.fit[0].set_ydata(_exp_decay(self.data_x, *popt))
        self.fig.canvas.draw()
        
        popt_str = ['amplitude', 'offset','decay']
        if is_display:
            if popt[0] > 0:
                self._display_popt(popt, popt_str, 'upper right')
            else:
                self._display_popt(popt, popt_str, 'lower right')
            
        self.fit_func = 'decay'
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

    def _update_unit(self, transform):

        for line in self.fig.axes[0].lines:
            data_x = np.array(line.get_xdata())
            if np.array_equal(data_x, np.array([0, 1])):
                line.set_xdata(data_x)
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    new_xdata = np.where(data_x != 0, transform(data_x), np.inf)
                    line.set_xdata(new_xdata)

        xlim = self.fig.axes[0].get_xlim()
        self.data_x = self.fig.axes[0].lines[0].get_xdata()
        self.fig.axes[0].set_xlim(transform(xlim[0]), transform(xlim[-1]))

        if self.selector == []:
            pass
        else:
            zoom_pan_handle = self.selector[2]
            zoom_pan_handle.x_center = transform(zoom_pan_handle.x_center)
            
            area_handle = self.selector[0]
            if area_handle.range[0] is not None:
                new_x1 = transform(self.selector[0].range[0])
                new_x2 = transform(self.selector[0].range[1])
                new_y1 = self.selector[0].range[2]
                new_y2 = self.selector[0].range[3]
                dummy_area(area_handle.ax, new_x1, new_y1, new_x2, new_y2)
            cross_handle = self.selector[1]
            if cross_handle.wavelength is not None:
                new_x = transform(cross_handle.xy[0])
                new_y = cross_handle.xy[1]
                dummy_cross(cross_handle.ax, new_x, new_y) 

        if self.fit is not None:
            self.clear()
            try:
                exec(f'self.{self.fit_func}()')
            except:
                pass

    def change_unit(self):
        if self.plot_type == '2D':
            return


        if self.unit in ['GHz', 'nm', 'MHz']:
            spl = 299792458  # m/s
            conversion_map = {
                'nm': ('GHz', lambda x: spl / x),
                'GHz': ('MHz', lambda x: x * 1e3),
                'MHz': ('nm', lambda x: spl / (x/1e3))
            }
        elif self.unit in ['ns', 'us', 'ms']:
            conversion_map = {
                'ms': ('ns', lambda x: x * 1e6),
                'ns': ('us', lambda x: x / 1e3),
                'us': ('ms', lambda x: x / 1e3)
            }
        else:
            return


        new_unit, conversion_func = conversion_map[self.unit]
        self._update_unit(conversion_func)

        ax = self.fig.axes[0]
        old_xlabel = ax.get_xlabel()
        new_xlabel = re.sub(r'\((.+)\)$', f'({new_unit})', old_xlabel)
        self.fig.axes[0].set_xlabel(new_xlabel)

        self.unit = new_unit
        self.fig.canvas.draw()

 



















