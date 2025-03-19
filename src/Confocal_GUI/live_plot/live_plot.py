import io
import os
import sys
import glob
import time
import threading
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
    def __init__(self, address):
        super().__init__()


        loaded = np.load(address, allow_pickle=True)
        keys = loaded.files
        try:
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
            short_addr = address[:-4].split('\\')[-1]
            self.measurement_name = f'load_from_{short_addr}_'

            self.daemon = True
            self.is_running = True
            self.is_done = False
            self.points_done = len([data for data in self.data_y.flatten() if not np.isnan(data)]) 
            #how many data points have done, will control display, filter out all np.nan which should be points not done
            self.repeat_done = 0
        except:
            raise KeyError('Not a valid data_figure file to load')
        
    
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
    
    def __init__(self, labels, update_time, data_generator, data, fig=None, relim_mode='normal'):

        self.labels = labels
        

        self.xlabel = labels[0]
        self.ylabel = labels[1]
        self.data_x = data[0]
        self.data_y = data[1]
        self.points_total = len(self.data_x)
        
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
            plt.close('all')
            # close all previous figures
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
        self.axes_formatter = SmartOffsetFormatter()
        self.axes.yaxis.set_major_formatter(self.axes_formatter)
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
                    time.sleep(self.update_time)
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
            ax.clear()
         
        self.fig.tight_layout()
        self.fig.canvas.draw()

    def update_verts(self, bins, counts, verts, mode='horizontal'):
        if mode=='horizontal':
            left = bins[:-1]
            right = bins[1:]
            counts = counts
            verts[:, 0, 0] = 0
            verts[:, 0, 1] = left
            verts[:, 1, 0] = counts
            verts[:, 1, 1] = left
            verts[:, 2, 0] = counts
            verts[:, 2, 1] = right
            verts[:, 3, 0] = 0
            verts[:, 3, 1] = right
        elif mode=='vertical':
            left = bins[:-1]
            right = bins[1:]
            counts = counts
            verts[:, 0, 0] = left
            verts[:, 0, 1] = 0
            verts[:, 1, 0] = left
            verts[:, 1, 1] = counts
            verts[:, 2, 0] = right
            verts[:, 2, 1] = counts
            verts[:, 3, 0] = right
            verts[:, 3, 1] = 0



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

        self.axes.set_xlim(self.data_x[0], self.data_x[-1]) # use index not min/max otherwise different orders under different units
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
        else:
            divider = make_axes_locatable(self.axes)        
            self.axdis = divider.append_axes("right", size="20%", pad=0.1, sharey=self.axes)

        self.axdis.xaxis.set_major_locator(AutoLocator())
        self.axdis.xaxis.set_major_formatter(ScalarFormatter())
        self.axdis.relim()
        self.axdis.autoscale_view()
        self.axdis.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        self.axdis.tick_params(axis='both', which='both',bottom=False,top=False)
        # reset axdis ticks, labels

        self.counts_max = 10
        # filter out zero data
        self.n_bins = np.min((self.points_total//4, 50))
        self.n, self.bins = np.histogram(self.data_y[:self.points_done, 0],
                        bins=self.n_bins, range=(self.ylim_min, self.ylim_max))

        self.verts = np.empty((self.n_bins, 4, 2))
        self.update_verts(self.bins, self.n, self.verts)
        self.poly = matplotlib.collections.PolyCollection(self.verts, facecolors='grey', animated=True)
        self.axdis.add_collection(self.poly)
        self.axdis.set_xlim(0, self.counts_max)
        self.blit_axes.append(self.axdis)
        self.blit_artists.append(self.poly)
        # use collection to manage hist patches



        self.poisson_fit_line, = self.axdis.plot(self.data_y[:, 0], [0 for data in self.data_y], color='orange', animated=True, alpha=1)
        self.blit_axes.append(self.axdis)
        self.blit_artists.append(self.poisson_fit_line)

        self.points_done_fits = self.points_done
        self.ylim_min_dis = self.ylim_min
        self.ylim_max_dis = self.ylim_max

        self.last_data_time = time.time()


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
                result = f'$\\sigma$={ratio:.2f}$\\sqrt{{\\mu}}$'
            else:
                ratio = popt[2]/np.sqrt(popt[1])
                if ratio <= 0.01: # ratio<1 means not a poisson distribution
                    ratio = popt[2]/popt[1]
                    result = f'$\\sigma$={ratio:.1e}$\\mu$'
                else:
                    result = f'$\\sigma$={ratio:.2f}$\\sqrt{{\\mu}}$'

            if not hasattr(self, 'fit_text'):
                self.fit_text = self.axdis.text(0.5, 1.01, 
                                                  result, transform=self.axdis.transAxes, 
                                                  color='orange', ha='center', va='bottom', animated=True)
                self.blit_artists.append(self.fit_text)
                self.blit_axes.append(self.axdis)
            else:
                self.fit_text.set_text(result)

        
    def update_core(self):
        
        if self.relim_dis() or self.relim() or (self.repeat_done!=self.data_generator.repeat_done):

            self.ylabel = self.labels[1] + f' x{self.data_generator.repeat_done + 1}'
            self.repeat_done = self.data_generator.repeat_done
            self.axes.set_ylabel(self.ylabel)
            self.n, self.bins = np.histogram(self.data_y[:self.points_done, 0],
                            bins=self.n_bins, range=(self.ylim_min_dis, self.ylim_max_dis))

            self.counts_max = np.max((np.max(self.n) + 5, int(np.max(self.n)*1.5)))
            self.axdis.set_xlim(0, self.counts_max)
            self.fig.canvas.draw()
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # update ylim so need redraw

        else: # no need to update bins positions
            self.n, _ = np.histogram(self.data_y[:self.points_done, 0],
                            bins=self.bins)
            if np.max(self.n) > self.counts_max:
                self.counts_max = np.max((np.max(self.n) + 5, int(np.max(self.n)*1.5)))
                self.axdis.set_xlim(0, self.counts_max)

                self.fig.canvas.draw()
                self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # update ylim so need redraw  

        self.fig.canvas.restore_region(self.bg_fig)
        for i, line in enumerate(self.lines):
            line.set_data(self.data_x, self.data_y[:, i])

        self.update_data_meter()
        self.update_dis()
        self.update_fit()

    def update_dis(self):
        self.update_verts(self.bins, self.n, self.verts)
        self.poly.set_verts(self.verts)

    def update_data_meter(self):
        if (time.time()-self.last_data_time) < 0.2:
            return
        self.last_data_time = time.time()
        newest_data = self.data_y[0, 0]
        if 1e-4<=np.abs(newest_data)<=1e4:
            oom = np.floor(np.log10(self.axes_formatter.abs_step))
            newest_data_str = f'{newest_data:.{0 if oom>=0 else -int(oom)}f}'
        else:
            newest_data_str = f'{newest_data:.1e}'
        if not hasattr(self, 'text'):
            self.text = self.axes.text(0.9, 1.005, 
                                              newest_data_str, transform=self.axes.transAxes, 
                                              color='grey', ha='right', va='bottom', animated=True, fontsize=12)
            self.blit_artists.append(self.text)
            self.blit_axes.append(self.axes)
        else:
            self.text.set_text(newest_data_str)


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


        self.lines = [self.axes.imshow(self.data_y.reshape(self.data_shape), animated=True, alpha=1, cmap=cmap, extent=extents),]
        divider = make_axes_locatable(self.axes)
        self.cax = divider.append_axes("right", size="5%", pad=0.15)
        self.cbar = self.fig.colorbar(self.lines[0], cax = self.cax)

        self.fig.axes[0].set_xlim((extents[0], extents[1]))
        self.fig.axes[0].set_ylim((extents[2], extents[3]))


        self.cbar.set_label(self.ylabel + ' x1')
        self.axes.set_ylabel(self.xlabel[1])
        self.axes.set_xlabel(self.xlabel[0])

        self.blit_axes.append(self.axes)
        self.blit_artists.append(self.lines[0])


        
    def update_core(self):

        if self.relim() or (self.repeat_done!=self.data_generator.repeat_done):

            self.ylabel = self.labels[1] + f' x{self.data_generator.repeat_done + 1}'
            self.repeat_done = self.data_generator.repeat_done
            self.cbar.set_label(self.ylabel)
            self.fig.canvas.draw() 
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        self.fig.canvas.restore_region(self.bg_fig)

        self.lines[0].set_array(self.data_y.reshape(self.data_shape))
        # other data just np.nan and controlled by set_bad
        

    def set_ylim(self):
        self.lines[0].set_clim(vmin=self.ylim_min, vmax=self.ylim_max)
        
    def choose_selector(self):

        self.area = AreaSelector(self.fig.axes[0])
        self.cross = CrossSelector(self.fig.axes[0])
        self.zoom = ZoomPan(self.fig.axes[0])

        self.ylim_min = np.nanmin(self.data_y[:self.points_done])
        self.ylim_max = np.nanmax(self.data_y[:self.points_done])
        self.lines[0].set_clim(vmin=self.ylim_min, vmax=self.ylim_max)
        # move colorbar to fit max
        


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
            self.axdis = self.fig.axes[1]
            self.cax = self.fig.axes[2]
            self.lines = [self.fig.axes[0].imshow(np.zeros(self.data_shape), animated=True, alpha=1, cmap=cmap, extent=extents),]
            self.fig.canvas.draw()
            self.cbar = self.fig.colorbar(self.lines[0], cax = self.cax)

            self.fig.axes[0].set_xlim((extents[0], extents[1]))
            self.fig.axes[0].set_ylim((extents[2], extents[3]))

        else:

            width, height = self.fig.get_size_inches()
            divider = make_axes_locatable(self.axes)        
            self.cax = divider.append_axes("right", size="5%", pad=0.15)
            self.axdis = divider.append_axes("top", size="20%", pad=0.25)
            self.fig.set_size_inches(width, height*1.25)
            self.lines = [self.axes.imshow(np.zeros(self.data_shape), animated=True, alpha=1, cmap=cmap, extent=extents),]
            self.fig.tight_layout()
            self.fig.canvas.draw()

            self.cbar = self.fig.colorbar(self.lines[0], cax = self.cax)
            self.fig.axes[0].set_xlim((extents[0], extents[1]))
            self.fig.axes[0].set_ylim((extents[2], extents[3]))

        self.counts_max = 10
        # filter out zero data
        self.axdis.xaxis.set_major_locator(AutoLocator())
        self.axdis.xaxis.set_major_formatter(ScalarFormatter())
        self.axdis.relim()
        self.axdis.autoscale_view()
        # reset axdis ticks, labels
        self.n_bins = np.min((self.points_total//4, 50))

        self.n, self.bins = np.histogram(self.data_y[:self.points_done, 0],
                        bins=self.n_bins, range=(self.ylim_min, self.ylim_max))

        self.verts = np.empty((self.n_bins, 4, 2))
        self.update_verts(self.bins, self.n, self.verts, mode='vertical')
        self.poly = matplotlib.collections.PolyCollection(self.verts, facecolors='grey', animated=True)
        self.axdis.add_collection(self.poly)
        self.axdis.set_xlim(0, self.counts_max)
        self.blit_axes.append(self.axdis)
        self.blit_artists.append(self.poly)

        self.blit_axes.append(self.axes)
        self.blit_artists.append(self.lines[0])


        self.cbar.set_label(self.ylabel + ' x1')
        self.axes.set_ylabel(self.xlabel[1])
        self.axes.set_xlabel(self.xlabel[0])

        self.axdis.tick_params(axis='both', which='both',bottom=False,top=False)

        
    def update_core(self):

        if self.relim() or (self.repeat_done!=self.data_generator.repeat_done):

            self.ylabel = self.labels[1] + f' x{self.data_generator.repeat_done + 1}'
            self.repeat_done = self.data_generator.repeat_done
            self.cbar.set_label(self.ylabel)

            self.n, self.bins = np.histogram(self.data_y[:self.points_done, 0],
                            bins=self.n_bins, range=(self.ylim_min, self.ylim_max))
            self.counts_max = np.max((np.max(self.n) + 5, int(np.max(self.n)*1.5)))
            self.axdis.set_ylim(0, self.counts_max)
            self.fig.canvas.draw()
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # update ylim so need redraw

        else: # no need to update bins positions
            self.n, _ = np.histogram(self.data_y[:self.points_done, 0],
                            bins=self.bins)
            if np.max(self.n) > self.counts_max:
                self.counts_max = np.max((np.max(self.n) + 5, int(np.max(self.n)*1.5)))
                self.axdis.set_ylim(0, self.counts_max)

                self.fig.canvas.draw()
                self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # update ylim so need redraw    


        self.fig.canvas.restore_region(self.bg_fig)

        self.lines[0].set_array(self.data_y.reshape(self.data_shape))
        # other data just np.nan and controlled by set_bad
   
        self.update_dis()    

    def update_dis(self):
        self.update_verts(self.bins, self.n, self.verts, mode='vertical')
        self.poly.set_verts(self.verts)  

    def set_ylim(self):
        self.lines[0].set_clim(vmin=self.ylim_min, vmax=self.ylim_max)
        self.axdis.set_xlim(self.ylim_min, self.ylim_max)
        
    def choose_selector(self):

        self.area = AreaSelector(self.fig.axes[0])
        self.cross = CrossSelector(self.fig.axes[0])
        self.zoom = ZoomPan(self.fig.axes[0])
        
        cmap = self.axes.images[0].colorbar.mappable.get_cmap()

        y_min = np.nanmin(self.data_y[:self.points_done])
        y_max = np.nanmax(self.data_y[:self.points_done])
        self.line_min = self.axdis.axvline(y_min, color='red', linewidth=6, alpha=0.3)
        self.line_max = self.axdis.axvline(y_max, color='red', linewidth=6, alpha=0.3)

        self.ylim_min = y_min
        self.ylim_max = y_max
        self.lines[0].set_clim(vmin=self.ylim_min, vmax=self.ylim_max)
        # move colorbar to fit max, min while not change axdis

        self.line_l = self.axdis.axvline(self.ylim_min, color=cmap(0), linewidth=6)
        self.line_h = self.axdis.axvline(self.ylim_max, color=cmap(0.95), linewidth=6)

        self.axdis.set_xticks([y_min, y_max])
        self.axdis.set_xticklabels([f'{xtick:.0f}' for xtick in [y_min, y_max]])

        self.drag_line = DragVLine(self.line_l, self.line_h, self.update_clim, self.axdis)
        self.fig.canvas.draw()
        # must be here to display self.line_l etc. after plot done, don't know why?
        
        self.selector = [self.area, self.cross, self.zoom, self.drag_line]
    
    def update_clim(self):
        vmin = self.line_l.get_xdata()[0]
        vmax = self.line_h.get_xdata()[0]
        self.lines[0].set_clim(vmin, vmax)

                      
class AreaSelector():
    def __init__(self, ax):
        self.ax = ax
        self.text = None
        self.range = [None, None, None, None]
        self.callback = None
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

        
    def on_callback(self):
        if self.callback is not None:
            self.callback()
        
    def onselect(self, eclick, erelease):
        x1, x2, y1, y2 = self.selector.extents
        # changed by rectangleselector
        
        if x1 == x2 or y1 == y2:
            self.range = [None, None, None, None]
            if self.text is not None:
                self.text.remove()
                self.text = None
            self.ax.figure.canvas.draw()
            self.on_callback()
            return
        
        self.range = [min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)]
        
        
        x_data = self.ax.get_xlim()
        y_data = self.ax.get_ylim()
        self.gap_x = np.abs(np.nanmax(x_data) - np.nanmin(x_data)) / 1000 if (len(x_data)>0) else 0.01
        self.gap_y = np.abs(np.nanmax(y_data) - np.nanmin(y_data)) / 1000 if (len(y_data)>0) else 0.01
        decimal_x = 0 if -int(np.ceil(np.log10(self.gap_x)))<0 else -int(np.ceil(np.log10(self.gap_x)))
        decimal_y = 0 if -int(np.ceil(np.log10(self.gap_y)))<0 else -int(np.ceil(np.log10(self.gap_y)))
        
        format_str = f'{{:.{decimal_x}f}}, {{:.{decimal_y}f}}'
        
        if self.text is None:
            self.text = self.ax.text(0.025, 0.975, f'({format_str.format(x1, y1)})\n({format_str.format(x2, y2)})',
                    transform=self.ax.transAxes,
                    color=self.color, ha = 'left', va = 'top'
                    )
        else:
            self.text.set_text(f'({format_str.format(x1, y1)})\n({format_str.format(x2, y2)})')

        self.ax.figure.canvas.draw()
        self.on_callback()
        
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
        self.callback = None
        artist = ax.get_children()[0]
        if isinstance(artist, matplotlib.image.AxesImage):
            cmap = ax.images[0].colorbar.mappable.get_cmap()
            self.color = cmap(0.95)
        else:
            self.color = 'grey'
        
        self.cid_press = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)

    def on_callback(self):
        if self.callback is not None:
            self.callback()

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
                    
                x_data = self.ax.get_xlim()
                y_data = self.ax.get_ylim()
                self.gap_x = np.abs(np.nanmax(x_data) - np.nanmin(x_data)) / 1000 if (len(x_data)>0) else 0.01
                self.gap_y = np.abs(np.nanmax(y_data) - np.nanmin(y_data)) / 1000 if (len(y_data)>0) else 0.01
                decimal_x = 0 if -int(np.ceil(np.log10(self.gap_x)))<0 else -int(np.ceil(np.log10(self.gap_x)))
                decimal_y = 0 if -int(np.ceil(np.log10(self.gap_y)))<0 else -int(np.ceil(np.log10(self.gap_y)))
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
                self.on_callback()
    
    def remove_point(self):
        if self.point is not None:
            self.vline.remove()
            self.hline.remove()
            self.point.remove()
            self.text.remove()
            self.point = None
            self.wavelength = None
            self.xy = None
        
    def set_active(self, active):
        if not active:
            self.ax.figure.canvas.mpl_disconnect(self.cid_press)

        
class ZoomPan():
    def __init__(self, ax):
        self.ax = ax
        self.cid_scroll = self.ax.figure.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.cid_press = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

        artist = ax.get_children()[0]
        if isinstance(artist, matplotlib.image.AxesImage):
            self.image_type = '2D'
            cmap = ax.images[0].colorbar.mappable.get_cmap()
            self.color = cmap(0)
            self.ax.set_facecolor(self.color)
            self.extents = artist.get_extent()

        else:
            self.image_type = '1D'
            
        self.dragging = False
        self.center_line = None
        self.callback = None
        self.data_figure = None

    def on_callback(self):
        if self.callback is not None:
            self.callback()


    def on_scroll(self, event):
        if event.inaxes == self.ax:

            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            self.x_center = (xlim[0] + xlim[1])/2
            self.y_center = (ylim[0] + ylim[1])/2

            xlim_min = xlim[0]
            ylim_min = ylim[0]

            scale_factor = 1.1 if event.button == 'up' else (1/1.1)

            xlim = [scale_factor*(xlim_min - self.x_center) + self.x_center\
                    , self.x_center - scale_factor*(xlim_min - self.x_center)]
            ylim = [scale_factor*(ylim_min - self.y_center) + self.y_center\
                    , self.y_center - scale_factor*(ylim_min - self.y_center)]
            
            if self.image_type == '2D':
                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)
            else:
                self.ax.set_xlim(xlim)
            self.ax.figure.canvas.draw()
            self.on_callback()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 2:
            if self.image_type == '1D':
                if event.dblclick:
                    if self.data_figure.selector[0].range[0] is not None:
                        # range of area_selector, self.data_figure is given in DataFigure
                        new_x_min = self.data_figure.selector[0].range[0]
                        new_x_max = self.data_figure.selector[0].range[1]
                        new_xlim = (new_x_min, new_x_max) if self.data_figure.data_x[0]<=self.data_figure.data_x[-1] else (new_x_max, new_x_min)
                        self.ax.set_xlim(new_xlim[0], new_xlim[1])
                        self.ax.figure.canvas.draw()
                    else:
                        self.ax.set_xlim(self.data_figure.data_x[0], self.data_figure.data_x[-1])
                        self.ax.figure.canvas.draw()
                    self.on_callback()
                    return

                self.dragging = True
                self.press_x_pixel = event.x
                self.xlim0 = self.ax.get_xlim()
                self.center_line = self.ax.axvline(np.mean(self.ax.get_xlim()),
                                                   color='red', linestyle='--', alpha=0.3)
                self.ax.figure.canvas.draw()
            else:
                if event.dblclick:
                    if self.data_figure.selector[0].range[0] is not None:
                        # range of area_selector, self.data_figure is given in DataFigure
                        range_array = self.data_figure.selector[0].range
                        self.ax.set_xlim(range_array[0], range_array[1])
                        self.ax.set_ylim(range_array[3], range_array[2])
                        self.ax.figure.canvas.draw()
                    else:

                        self.ax.set_xlim((self.extents[0], self.extents[1]))
                        self.ax.set_ylim((self.extents[2], self.extents[3]))
                        self.ax.figure.canvas.draw()
                    self.on_callback()
                    return

                self.dragging = True
                self.press_x_pixel = event.x
                self.press_y_pixel = event.y
                self.xlim0 = self.ax.get_xlim()
                self.ylim0 = self.ax.get_ylim()
                self.center_line = self.ax.scatter(np.mean(self.ax.get_xlim()), np.mean(self.ax.get_ylim()),
                                                   color='red', s=30, alpha=0.3)
                self.ax.figure.canvas.draw()

    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return
        if self.image_type == '1D':
            dx_pixels = event.x - self.press_x_pixel
            bbox = self.ax.get_window_extent()
            pixel_width = bbox.width
            data_width = self.xlim0[1] - self.xlim0[0]
            dx_data = dx_pixels * data_width / pixel_width
            new_xlim = (self.xlim0[0] - dx_data, self.xlim0[1] - dx_data)
            self.ax.set_xlim(new_xlim)
            mid = np.mean(new_xlim)
            self.center_line.set_xdata([mid, mid])
            self.ax.figure.canvas.draw_idle()
        else:
            dx_pixels = event.x - self.press_x_pixel
            dy_pixels = event.y - self.press_y_pixel
            bbox = self.ax.get_window_extent()
            pixel_width = bbox.width
            pixel_height = bbox.height
            data_width_x = self.xlim0[1] - self.xlim0[0]
            data_width_y = self.ylim0[1] - self.ylim0[0]
            dx_data = dx_pixels * data_width_x / pixel_width
            dy_data = dy_pixels * data_width_y / pixel_height
            new_xlim = (self.xlim0[0] - dx_data, self.xlim0[1] - dx_data)
            new_ylim = (self.ylim0[0] - dy_data, self.ylim0[1] - dy_data)
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            mid_x, mid_y = np.mean(new_xlim), np.mean(new_ylim)
            self.center_line.set_offsets([[mid_x, mid_y]])
            self.ax.figure.canvas.draw_idle()

    def on_release(self, event):
        if event.button == 2 and self.dragging:
            self.dragging = False
            if self.center_line is not None:
                self.center_line.remove()
                self.center_line = None
            self.ax.figure.canvas.draw()
            self.on_callback()

            
    def set_active(self, active):
        if not active:
            self.ax.figure.canvas.mpl_disconnect(self.cid_scroll)
            self.ax.figure.canvas.mpl_disconnect(self.cid_press)
            self.ax.figure.canvas.mpl_disconnect(self.cid_motion)
            self.ax.figure.canvas.mpl_disconnect(self.cid_release)
    
            
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

    press_event = MouseEvent('button_press_event', ax.figure.canvas, x1_disp, y1_disp, button=1)
    press_event.inaxes = ax
    ax.figure.canvas.callbacks.process('button_press_event', press_event)
    motion_event = MouseEvent('motion_notify_event', ax.figure.canvas, x2_disp, y2_disp, button=1)
    motion_event.inaxes = ax
    ax.figure.canvas.callbacks.process('motion_notify_event', motion_event)
    release_event = MouseEvent('button_release_event', ax.figure.canvas, x2_disp, y2_disp, button=1)
    release_event.inaxes = ax
    ax.figure.canvas.callbacks.process('button_release_event', release_event)
    # the first selection close existing rectangle, otherwise bug
    time.sleep(0.01)
    press_event = MouseEvent('button_press_event', ax.figure.canvas, x1_disp, y1_disp, button=1)
    press_event.inaxes = ax
    ax.figure.canvas.callbacks.process('button_press_event', press_event)
    motion_event = MouseEvent('motion_notify_event', ax.figure.canvas, x2_disp, y2_disp, button=1)
    motion_event.inaxes = ax
    ax.figure.canvas.callbacks.process('motion_notify_event', motion_event)
    release_event = MouseEvent('button_release_event', ax.figure.canvas, x2_disp, y2_disp, button=1)
    release_event.inaxes = ax
    ax.figure.canvas.callbacks.process('button_release_event', release_event)

            


valid_fit_func = ['lorent', 'decay', 'rabi', 'lorent_zeeman', 'center']         
class DataFigure():
    """
    The class contains all data of the figure, enables more operations
    such as curve fit or save data
    
    Parameters
    ----------
    live_plot :instance of class LivePlot
    
    
    Examples
    --------
    >>> data_figure = DataFigure(live_plot=live_plot)
    or
    >>> data_figure = DataFigure(is_GUI=True)
    
    >>> data_x, data_y = data_figure.data
    
    >>> data_figure.save('my_figure')
    'save to my_figure_{time}.jpg and my_figure_{time}.txt'

    >>> data_figure.lorent(p0 = None)
    'figure with lorent curve fit'
    
    >>> data_figure.clear()
    'remove lorent fit and text'
    """
    def __init__(self, live_plot=None, address=None, fig=None, relim_mode='tight', is_GUI=False):

        if is_GUI is True:
            from Confocal_GUI.gui import GUI_Load
            # load address using GUI
            address = GUI_Load()
            if address == '':
                return
            else:
                address = address.split('.')[-2] + '*'

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
                self.data_y = data_generator.data_y if len(data_generator.data_y.shape)==2 \
                    else data_generator.data_y.reshape(data_generator.data_y.shape[0], 1)
                # to read old data
                x_label = data_generator.info.get('x_label', 'Data (1)')
                y_label = data_generator.info.get('y_label', f'Counts/{exposure}s x1')

                _live_plot = PLELive(labels=[x_label, y_label[:-3]],
                                    update_time=0.1, data_generator=data_generator, data=[self.data_x, self.data_y]
                                    , relim_mode = relim_mode, fig=fig)

            else:
                self.data_x = data_generator.data_x
                self.data_y = data_generator.data_y
                x_label = data_generator.info.get('x_label', ['X', 'Y'])
                y_label = data_generator.info.get('y_label', f'Counts/{exposure}s x1')

                _live_plot = PLDisLive(labels=[x_label, y_label[:-3]],
                        update_time=1, data_generator=data_generator, data=[self.data_x, self.data_y],
                        relim_mode = relim_mode, fig=fig)

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
        self.selector[2].data_figure = self # give zoompan the handle to live_plot
        self._load_unit(True if (address is not None) else False)
        warnings.filterwarnings("ignore", category=OptimizeWarning)

    def _load_unit(self, is_load):
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

        if not is_load:
            self.unit_original = self.unit
        else:
            self.unit_original = self.info.get('unit_original', self.unit)
            # load from saved file and has unit_original

        if self.unit in ['GHz', 'nm', 'MHz']:
            spl = 299792458  # m/s
            self.conversion_map = {
                'nm': ('GHz', lambda x: spl / x),
                'GHz': ('MHz', lambda x: x * 1e3),
                'MHz': ('nm', lambda x: spl / (x/1e3))
            }
        elif self.unit in ['ns', 'us', 'ms']:
            self.conversion_map = {
                'ms': ('ns', lambda x: x * 1e6),
                'ns': ('us', lambda x: x / 1e3),
                'us': ('ms', lambda x: x / 1e3)
            }
        else:
            self.conversion_map = None

        self._update_transform_back()
        

    def xlim(self, x_min, x_max):
        self.fig.axes[0].set_xlim(x_min, x_max)

    def ylim(self, y_min, y_max):
        self.fig.axes[0].set_ylim(y_min, y_max)

    def close_selector(self):
        for selector in self.selector:
            selector.set_active(False)

    def _align_to_grid(self, x, type):
        # round to center of 2D grid
        # type='x' for x, type='y' for y
        if not self.plot_type == '2D':
            return
        if not hasattr(self, 'grid_center'):
            self.grid_center = self.data_x[0] # one of the center of grid, [x_center, y_center]
            self.step_x = np.abs(self.live_plot.data_generator.x_array[1] - self.live_plot.data_generator.x_array[0])
            self.step_y = np.abs(self.live_plot.data_generator.y_array[1] - self.live_plot.data_generator.y_array[0])

        if type == 'x':
            return round((x-self.grid_center[0])/self.step_x)*self.step_x + self.grid_center[0]
        if type == 'y':
            return round((x-self.grid_center[1])/self.step_y)*self.step_y + self.grid_center[1]

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
                info = {**self.info, **extra_info, **{'x_label':x_label, 'y_label':y_label, 'unit_original':self.unit_original}})
        else:
            x_label = self.fig.axes[0].get_xlabel()
            y_label = self.fig.axes[0].get_ylabel()
            z_label = self.fig.axes[0].images[0].colorbar.ax.yaxis.label.get_text()
            np.savez(addr + self.measurement_name + time_str + '.npz', data_x = self.data_x, data_y = self.data_y, \
                info = {**self.info, **extra_info, **{'x_label':[x_label, y_label], 'y_label':z_label, 'unit_original':self.unit_original}})

        print(f'saved fig as {addr}{self.measurement_name}{time_str}.npz')

    def _min_overlap(self, ax, text, candidates=None):
        """
        Given a list of candidate positions in normalized coordinates, each candidate being a tuple:
            (norm_x, norm_y, ha, va),
        this function positions the text (using ax.transAxes) for each candidate, forces a draw,
        and calculates the fraction of the line's total length that overlaps with the text's bounding box.
        For multi-line text (detected via '\n'), the function computes an overall bounding box
        from the rendered sizes of the first line and the remaining lines.

        The Liang-Barsky algorithm is used to compute the intersection length between each line segment
        (formed by consecutive data points) and the candidate text bounding box.

        The candidate whose overlapping fraction (overlap length / total line length) is minimal is chosen.

        Parameters:
            ax : matplotlib.axes.Axes
                The axes that contain the line and text.
            text : matplotlib.text.Text
                The text object to be positioned.
            candidates : list of tuples
                A list of candidate positions, each specified as 
                (normalized_x, normalized_y, ha, va), where normalized_x and normalized_y are in [0, 1]
                (axes coordinates), and ha, va are the horizontal and vertical alignment strings.
        """

        # Use default candidates if none provided.
        if candidates is None:
            candidates = [
                (0.025, 0.85, 'left', 'top'),
                (0.975, 0.85, 'right', 'top'),
                (0.025, 0.025, 'left', 'bottom'),
                (0.975, 0.025, 'right', 'bottom'),
                (0.025, 0.5, 'left', 'center'),
                (0.975, 0.5, 'right', 'center'),
                (0.5, 0.025, 'center', 'bottom'),
                (0.5, 0.85, 'center', 'top'),
                (0.5, 0.5, 'center', 'center'),
            ]

        canvas = ax.figure.canvas
        # Hide text during processing.
        text.set_alpha(0)
        orig_text = text.get_text()
        renderer = canvas.get_renderer()

        # ---------------------------------------------------------------------
        # Precompute polyline points in display coordinates.
        if self.plot_type == '1D':
            pts = np.column_stack([self.data_x_p, self.data_y_p])
            pts_disp = ax.transData.transform(pts)
            pts_full = np.column_stack([self.data_x, self.data_y[:, 0]])
            pts_disp_full = ax.transData.transform(pts_full)
        elif self.plot_type == '2D':
            pts = [[self.popt[-2], self.popt[-1]], [self.popt[-2]+1e-3, self.popt[-1]+1e-3]]
            pts_disp = ax.transData.transform(pts)
            pts_full = [[self.popt[-2], self.popt[-1]], [self.popt[-2]+1e-3, self.popt[-1]+1e-3]]
            pts_disp_full = ax.transData.transform(pts_full)
            # set center of fit as the line which text needed to avoid


        def total_length(pts_arr):
            """Compute the total length of a polyline given its display coordinates."""
            seg_lengths = np.hypot(np.diff(pts_arr[:, 0]), np.diff(pts_arr[:, 1]))
            return np.sum(seg_lengths)

        total_length_par = total_length(pts_disp)
        total_length_full = total_length(pts_disp_full)

        # ---------------------------------------------------------------------
        # Precompute overall text bounding box dimensions.
        # Render the text at the center (with center alignment) to obtain a consistent size.
        text.set_ha('center')
        text.set_va('center')
        text.set_position((0.5, 0.5))
        lines = orig_text.split("\n")

        # Render the first line and get its bounding box.
        text.set_text(lines[0])
        canvas.draw()
        first_bbox = text.get_window_extent(renderer)
        first_width = first_bbox.width
        first_height = first_bbox.height

        # If multi-line text, render the remaining lines and get their bounding box.
        if len(lines) > 1:
            text.set_text("\n".join(lines[1:]))
            canvas.draw()
            rest_bbox = text.get_window_extent(renderer)
            rest_width = rest_bbox.width
            rest_height = rest_bbox.height
        else:
            rest_bbox = None
            rest_width = 0
            rest_height = 0

        overall_width = first_width if rest_bbox is None else max(first_width, rest_width)
        overall_height = first_height if rest_bbox is None else (first_height + rest_height)

        # ---------------------------------------------------------------------
        # Helper: Compute candidate bounding box in display coordinates.
        def candidate_bbox(norm_x, norm_y, ha, va):
            """
            Compute the candidate text bounding box (xmin, ymin, xmax, ymax) in display coordinates,
            given normalized coordinates and alignment.
            """
            anchor_disp = ax.transAxes.transform((norm_x, norm_y))
            # Horizontal alignment.
            if ha == 'left':
                bbox_x0 = anchor_disp[0]
            elif ha == 'center':
                bbox_x0 = anchor_disp[0] - overall_width / 2
            elif ha == 'right':
                bbox_x0 = anchor_disp[0] - overall_width
            else:
                bbox_x0 = anchor_disp[0]

            # Vertical alignment.
            if va == 'top':
                bbox_y1 = anchor_disp[1]
                bbox_y0 = bbox_y1 - overall_height
            elif va == 'center':
                bbox_y0 = anchor_disp[1] - overall_height / 2
                bbox_y1 = anchor_disp[1] + overall_height / 2
            elif va == 'bottom':
                bbox_y0 = anchor_disp[1]
                bbox_y1 = bbox_y0 + overall_height
            else:
                bbox_y0 = anchor_disp[1] - overall_height / 2
                bbox_y1 = anchor_disp[1] + overall_height / 2

            return (bbox_x0, bbox_y0, bbox_x0 + overall_width, bbox_y0 + overall_height)

        # ---------------------------------------------------------------------
        # Liang-Barsky algorithm to compute overlapping length.
        def liang_barsky_clip_length(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
            """
            Compute the length of the portion of the line segment from (x0,y0) to (x1,y1)
            that lies within the rectangle [xmin, xmax] x [ymin, ymax] using the Liang-Barsky algorithm.
            """
            dx = x1 - x0
            dy = y1 - y0
            p = [-dx, dx, -dy, dy]
            q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]
            u1, u2 = 0.0, 1.0

            for pi, qi in zip(p, q):
                if pi == 0:
                    if qi < 0:
                        return 0.0  # Line is parallel and outside the boundary.
                else:
                    t = qi / float(pi)
                    if pi < 0:
                        u1 = max(u1, t)
                    else:
                        u2 = min(u2, t)
            if u1 > u2:
                return 0.0  # No valid intersection.
            seg_length = np.hypot(dx, dy)
            return (u2 - u1) * seg_length

        def compute_overlap_length(pts_arr, rect):
            """
            Compute the total overlapping length of the polyline (represented by pts_arr)
            with the given rectangular region.
            """
            xmin, ymin, xmax, ymax = rect
            overlap = 0.0
            for i in range(len(pts_arr) - 1):
                x0, y0 = pts_arr[i]
                x1, y1 = pts_arr[i+1]
                overlap += liang_barsky_clip_length(x0, y0, x1, y1, xmin, xmax, ymin, ymax)
            return overlap

        # ---------------------------------------------------------------------
        # Iterate over candidate positions and compute overlapping fractions.
        best_candidate_par = None
        min_fraction_par = np.inf
        best_candidate_full = None
        min_fraction_full = np.inf
        overlap_fraction_par_for_full_candidate = None

        for candidate in candidates:
            norm_x, norm_y, ha, va = candidate
            rect = candidate_bbox(norm_x, norm_y, ha, va)

            # (Optional) Update text properties for visualization.
            text.set_ha(ha)
            text.set_va(va)
            text.set_position((norm_x, norm_y))
            text.set_text(orig_text)
            canvas.draw()  # This draw() may be removed if not strictly necessary.

            overlap_par = compute_overlap_length(pts_disp, rect)
            overlap_full = compute_overlap_length(pts_disp_full, rect)

            fraction_par = overlap_par / total_length_par if total_length_par > 0 else 0
            fraction_full = overlap_full / total_length_full if total_length_full > 0 else 0

            if fraction_par < min_fraction_par:
                min_fraction_par = fraction_par
                best_candidate_par = candidate

            if fraction_full < min_fraction_full:
                min_fraction_full = fraction_full
                best_candidate_full = candidate
                overlap_fraction_par_for_full_candidate = fraction_par

        # Choose best candidate: if candidate from "full" set yields zero overlap in "par", prefer it.
        best_candidate = best_candidate_full if overlap_fraction_par_for_full_candidate == 0 else best_candidate_par

        # ---------------------------------------------------------------------
        # Update text object with the best candidate and make it visible.
        norm_x, norm_y, ha, va = best_candidate
        text.set_ha(ha)
        text.set_va(va)
        text.set_position((norm_x, norm_y))
        text.set_text(orig_text)
        text.set_alpha(1)
        canvas.draw()



    def _display_popt(self, popt, popt_str):
        # popt_str = ['amplitude', 'offset', 'omega', 'decay', 'phi'], popt_pos = 'lower left' etc

        _popt = popt
        formatted_popt = [f'{x:.5f}'.rstrip('0') for x in _popt]
        result_list = [f'{name}={value}' for name, value in zip(popt_str, formatted_popt)]
        formatted_popt_str = '\n'.join(result_list)
        result = f"{self.formula_str}\n{formatted_popt_str}"
        self.log_info = result
        # format popt to display as text
                    
        if self.text is None:
            self.text = self.fig.axes[0].text(0.5, 0.5, 
                                              result, transform=self.fig.axes[0].transAxes, 
                                              color='blue', ha='center', va='center', fontsize=10)

        else:
            self.text.set_text(result)


        self._min_overlap(self.fig.axes[0], self.text)
        for line in self.live_plot.lines:
            line.set_alpha(0.5)

        self.fig.canvas.draw()

    def _select_fit(self, min_num=2):
        # return data in the area selector, and only return first set if there are multiple sets of data (only data not data_ref)
        valid_index = [i for i, data in enumerate(self.data_y) if not np.isnan(data[0])]
        # index of none np.nan data
        if self.plot_type == '1D':
            if self.selector[0].range[0] is None:
                xlim = self.fig.axes[0].get_xlim()
                index_l = np.argmin(np.abs(self.data_x[valid_index] - xlim[0]))
                index_h = np.argmin(np.abs(self.data_x[valid_index] - xlim[1]))
                index_l, index_h = np.sort([index_l, index_h])
                # in order to handle data_x from max to min (e.g. GHz unit)
                if np.abs(index_l - index_h)<=min_num:
                    return self.data_x[valid_index], self.data_y[valid_index, 0]
                return self.data_x[valid_index][index_l:index_h], self.data_y[valid_index, 0][index_l:index_h]
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

        elif self.plot_type == '2D':
            if self.selector[0].range[0] is None:
                xl, xh = np.sort(self.fig.axes[0].get_xlim())
                yl, yh = np.sort(self.fig.axes[0].get_ylim())
                xl, xh = [self._align_to_grid(v, 'x') for v in (xl, xh)]
                yl, yh = [self._align_to_grid(v, 'y') for v in (yl, yh)]
                index_area = np.where(
                    (self.data_x[valid_index, 0] >= xl) & (self.data_x[valid_index, 0] <= xh) &
                    (self.data_x[valid_index, 1] >= yl) & (self.data_x[valid_index, 1] <= yh)
                )[0]
                if len(index_area)<=min_num:
                    return (self.data_x[valid_index][:, 0], self.data_x[valid_index][:, 1]), self.data_y[valid_index, 0]

                data_x_p = self.data_x[valid_index][index_area]
                return (data_x_p[:, 0], data_x_p[:, 1]), self.data_y[valid_index][index_area, 0]
            else:
                xl, xh, yl, yh = self.selector[0].range
                if (xl is None) or (xh is None):
                    return (self.data_x[valid_index][:, 0], self.data_x[valid_index][:, 1]), self.data_y[valid_index, 0]
                if (xl - xh)==0:
                    return (self.data_x[valid_index][:, 0], self.data_x[valid_index][:, 1]), self.data_y[valid_index, 0]

                xl, xh = [self._align_to_grid(v, 'x') for v in (xl, xh)]
                yl, yh = [self._align_to_grid(v, 'y') for v in (yl, yh)]
                index_area = np.where(
                    (self.data_x[valid_index, 0] >= xl) & (self.data_x[valid_index, 0] <= xh) &
                    (self.data_x[valid_index, 1] >= yl) & (self.data_x[valid_index, 1] <= yh)
                )[0]
                if len(index_area)<=min_num:
                    return (self.data_x[valid_index][:, 0], self.data_x[valid_index][:, 1]), self.data_y[valid_index, 0]

                data_x_p = self.data_x[valid_index][index_area]
                return (data_x_p[:, 0], data_x_p[:, 1]), self.data_y[valid_index][index_area, 0]
                # should return data_x_p as ([x0, x1, ...], [y0, y1, ...])

       

    def _fit_and_draw(self, is_fit, is_display, kwargs):
        # use self.p0_list and self.bounds, self.popt_str, self._fit_func
        for index, param in enumerate(self.popt_str):
            if kwargs.get(param, None) is not None:
                self.bounds[0][index], self.bounds[1][index] = np.sort([kwargs.get(param, None)*(1-1e-5), kwargs.get(param, None)*(1+1e-5)])
                for p0 in self.p0_list:
                    p0[index] = kwargs.get(param, None)

        if is_fit:
            try:
                loss_min = np.inf
                for p0 in self.p0_list:
                    popt_cur, pcov_cur = curve_fit(self._fit_func, self.data_x_p, self.data_y_p, p0=p0, bounds = self.bounds)
                    loss_cur = np.sum((self._fit_func(self.data_x_p, *popt_cur) - self.data_y_p)**2)
                    if loss_cur<loss_min:
                        loss_min = loss_cur
                        popt = popt_cur
                        pcov = pcov_cur
            except:
                return 'error', 'error'

        else:
            popt, pcov = self.p0_list[0], None
        self.popt = popt

        if is_display:
            self._display_popt(popt, self.popt_str)

        if self.plot_type == '1D':
            if self.fit is None:
                self.fit = self.fig.axes[0].plot(self.data_x, self._fit_func(self.data_x, *popt), color='orange', linestyle='--')
            else:
                self.fit[0].set_ydata(self._fit_func(self.data_x, *popt))
        elif self.plot_type == '2D':
            if self.fit is None:
                self.fit = [self.fig.axes[0].scatter(popt[-2], popt[-1], color='orange', s=50),]
                circle = matplotlib.patches.Circle((popt[-2], popt[-1]), radius=popt[-3], edgecolor='orange'
                    , facecolor='none', linewidth=2, alpha=0.5)
                self.fit.append(circle)
                self.fig.axes[0].add_patch(circle)
            else:
                self.fit[0].set_offsets((popt[-2], popt[-1]))
                self.fit[1].set_center((popt[-2], popt[-1]))
                self.fit[1].set_radius(popt[-3])

        self.fig.canvas.draw()

        return popt, pcov


    def lorent(self, p0=None, is_display=True, is_fit=True, **kwargs):
        if self.plot_type == '2D':
            return 
        self.data_x_p, self.data_y_p = self._select_fit(min_num=4)
        # use the area selector results for fitting , min_num should at least be number of fitting parameters
        self.formula_str = '$f(x)=H\\frac{(FWHM/2)^2}{(x-x_0)^2+(FWHM/2)^2}+B$'
        def _lorent(x, center, full_width, height, bg):
            return height*((full_width/2)**2)/((x - center)**2 + (full_width/2)**2) + bg
        self._fit_func = _lorent
        if p0 is None:# no input
            self.p0_list = []
            # likely be positive height
            guess_center = self.data_x_p[np.argmax(self.data_y_p)]
            guess_height = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))
            guess_bg = np.min(self.data_y_p)
            guess_full_width = np.abs(self.data_x_p[0]-self.data_x_p[-1])/4
            self.p0_list.append([guess_center, guess_full_width, guess_height, guess_bg])

            # likely be negtive height
            guess_center = self.data_x_p[np.argmin(self.data_y_p)]
            guess_height = -np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))
            guess_bg = np.max(self.data_y_p)
            guess_full_width = np.abs(self.data_x_p[0]-self.data_x_p[-1])/4
            self.p0_list.append([guess_center, guess_full_width, guess_height, guess_bg])


        else:
            self.p0_list = [p0, ]
            guess_center = self.p0[0]
            guess_full_width = self.p0[1]
            guess_height = self.p0[2]
            guess_bg = self.p0[3]

        data_x_range = np.abs(self.data_x_p[-1] - self.data_x_p[0])
        data_y_range = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))
        self.bounds = ([np.nanmin(self.data_x_p), guess_full_width/10, -10*data_y_range, np.nanmin(self.data_y_p)-10*data_y_range], \
        [np.nanmax(self.data_x_p), guess_full_width*10, 10*data_y_range, np.nanmax(self.data_y_p)+10*data_y_range])
        
        self.popt_str = ['$x_0$', 'FWHM', 'H', 'B']
        popt, pcov = self._fit_and_draw(is_fit, is_display, kwargs)
        self.fit_func = 'lorent'
        return [self.popt_str, pcov], popt


    def lorent_zeeman(self, p0=None, is_display=True, is_fit=True, **kwargs):
        #fit of PLE under B field, will rewrite soon
        if self.plot_type == '2D':
            return 
        
        self.data_x_p, self.data_y_p = self._select_fit(min_num=5)
        # use the area selector results for fitting , min_num should at least be number of fitting parameters
        self.formula_str = '$f(x)=H(L(\\delta/2)+L(-\\delta/2))+B$'
        def _lorent_zeeman(x, center, full_width, height, bg, split):
            return height*((full_width/2)**2)/((x - center - split/2)**2 + (full_width/2)**2) \
                + height*((full_width/2)**2)/((x - center + split/2)**2 + (full_width/2)**2) + bg
        self._fit_func = _lorent_zeeman
        if p0 is None:# no input
            self.p0_list = []
            try:
                guess_height = (np.max(self.data_y_p)-np.min(self.data_y_p))
                peaks, properties = find_peaks(self.data_y_p, width=1, prominence=guess_height/8) # width about 100MHz
                if len(peaks)==0:
                    return
                peaks_largest = peaks[np.argsort(self.data_y_p[peaks])[::-1]]
                for second_peak in peaks_largest:
                    guess_center = self.data_x_p[int(np.mean([peaks_largest[0], second_peak]))]
                    guess_full_width = properties['widths'][np.argsort(self.data_y_p[peaks])[-1]]*np.abs(self.data_x_p[1]-self.data_x_p[0])
                    guess_spl = np.abs((self.data_x_p[second_peak] - guess_center)*2)
                    if guess_spl<guess_full_width:
                        guess_height = guess_height/2
                    guess_bg = np.min(self.data_y_p)
                    self.p0_list.append([guess_center, guess_full_width, guess_height, guess_bg, guess_spl])

            except:
                return
        else:
            self.p0_list = [p0, ]
            guess_center = self.p0[0]
            guess_full_width = self.p0[1]
            guess_height = self.p0[2]
            guess_bg = self.p0[3]

        data_x_range = np.abs(self.data_x_p[-1] - self.data_x_p[0])
        data_y_range = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))
        self.bounds = ([np.nanmin(self.data_x_p), guess_full_width/10, -10*data_y_range, np.nanmin(self.data_y_p)-10*data_y_range, 0], \
        [np.nanmax(self.data_x_p), guess_full_width*10, 10*data_y_range, np.nanmax(self.data_y_p)+10*data_y_range, 2*data_x_range])
        
        self.popt_str = ['$x_0$', 'FWHM', 'H', 'B', '$\\delta$']
        popt, pcov = self._fit_and_draw(is_fit, is_display, kwargs)
        self.fit_func = 'lorent_zeeman'
        return [self.popt_str, pcov], popt


    def rabi(self, p0=None, is_display=True, is_fit=True, **kwargs):
        if self.plot_type == '2D':
            return 
        self.data_x_p, self.data_y_p = self._select_fit(min_num=5)

        self.formula_str = '$f(x)=A\\sin(2{\\pi}fx+\\varphi)e^{-x/\\tau}+B$'
        def _rabi(x, amplitude, offset, omega, decay, phi):
            return amplitude*np.sin(2*np.pi*omega*x + phi)*np.exp(-x/decay) + offset
        self._fit_func = _rabi
        if p0 is None:# no input
            guess_amplitude = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))/2
            guess_offset = np.mean(self.data_y_p)


            N = len(self.data_y_p)
            delta_x = self.data_x_p[1] - self.data_x_p[0]
            y_detrended = self.data_y_p - np.mean(self.data_y_p)
            fft_vals = np.fft.fft(y_detrended)
            fft_freq = np.fft.fftfreq(N, d=delta_x)
            mask = fft_freq > 0
            fft_vals = fft_vals[mask]
            fft_freq = fft_freq[mask]
            idx_peak = np.argmax(np.abs(fft_vals))
            guess_omega = fft_freq[idx_peak]
            # fft data to get frequency 

            delta_x_min_max = np.abs(self.data_x_p[np.argmin(self.data_y_p)] - self.data_x_p[np.argmax(self.data_y_p)])
            ratio_min_max = (np.abs(np.min(self.data_y_p) - guess_offset)/np.abs(np.max(self.data_y_p) - guess_offset))
            # amp_min = amp_max*exp(-delta_x_min_max/guess_decay)
            # guess_decay = -delta_x_min_max/ln(ratio_min_max)
            guess_decay = np.abs(-delta_x_min_max/np.log(ratio_min_max))
            guess_phi = np.pi/2


            self.p0_list = [[guess_amplitude, guess_offset, guess_omega, guess_decay, guess_phi],]

        else:
            self.p0_list = [p0, ]

            guess_amplitude = self.p0[0]
            guess_offset = self.p0[1]
            guess_omega = self.p0[2]
            guess_decay = self.p0[3]
            guess_phi = self.p0[4]

        data_y_range = np.max(self.data_y_p) - np.min(self.data_y_p)
        self.bounds = ([guess_amplitude/5, np.nanmin(self.data_y_p), guess_omega/5, guess_decay/5, guess_phi - np.pi/20], \
            [guess_amplitude*5, np.nanmax(self.data_y_p), guess_omega*5, guess_decay*5, guess_phi + np.pi/20])
        
        self.popt_str = ['A', 'B', 'f', '$\\tau$', '$\\varphi$']
        popt, pcov = self._fit_and_draw(is_fit, is_display, kwargs)            
        self.fit_func = 'rabi'
        return [self.popt_str, pcov], popt


    def decay(self, p0=None, is_display=True, is_fit=True, **kwargs):
        if self.plot_type == '2D':
            return 
        self.data_x_p, self.data_y_p = self._select_fit(min_num=3)
        self.formula_str = '$f(x)=Ae^{-x/\\tau}+B$'
        def _exp_decay(x, amplitude, offset, decay):
            return amplitude*np.exp(-x/decay) + offset
        self._fit_func = _exp_decay
        if p0 is None:# no input
            self.p0_list = []
            # if positive
            guess_amplitude = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))
            guess_offset = np.mean(self.data_y_p)
            guess_decay = np.abs(self.data_x_p[np.argmin(self.data_y_p)] - self.data_x_p[np.argmax(self.data_y_p)])/2
            self.p0_list.append([guess_amplitude, guess_offset, guess_decay])
            # if negtive
            self.p0_list.append([-guess_amplitude, guess_offset, guess_decay])

        else:
            self.p0_list = [p0, ]

            guess_amplitude = self.p0[0]
            guess_offset = self.p0[1]
            guess_decay = self.p0[2]
            
        data_y_range = np.max(self.data_y_p) - np.min(self.data_y_p)
        self.bounds = ([-4*data_y_range, guess_offset - data_y_range, guess_decay/10], \
            [4*data_y_range, guess_offset + data_y_range, guess_decay*10])
        
        self.popt_str = ['A', 'B','$\\tau$']
        popt, pcov = self._fit_and_draw(is_fit, is_display, kwargs)            
        self.fit_func = 'decay'
        return [self.popt_str, pcov], popt

    # 2D plot fit only display center, and x0, y0 must be last two parameters for fit_func
    def center(self, p0=None, is_display=True, is_fit=True, **kwargs):
        if self.plot_type == '1D':
            return 
        self.data_x_p, self.data_y_p = self._select_fit(min_num=5)
        # data_x_p is [[x0, y0], [x1, y1], ...]
        self.formula_str = '$f(r)=Ae^{-(r-(x0,y0))^2/R^2}+B$'
        def _center(coord, amplitude, offset, size, x0, y0):
            # coord is (x_array, y_array) or (x, y)
            # center is (x0, y0)
            x, y = coord
            x, y = np.array(x), np.array(y)
            x_dis = np.abs(x - x0)
            y_dis = np.abs(y - y0)
            return amplitude*np.exp(-(x_dis**2+y_dis**2)/size**2) + offset

        self._fit_func = _center
        if p0 is None:# no input
            self.p0_list = []
            guess_amplitude = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))
            guess_offset = np.min(self.data_y_p)

            max_5_points = np.argsort(self.data_y_p)[::-1][:5]
            x_range = np.ptp(self.data_x_p[0][max_5_points])
            y_range = np.ptp(self.data_x_p[1][max_5_points])
            guess_size = np.hypot(x_range, y_range)
            guess_x0 = np.mean(self.data_x_p[0][max_5_points])
            guess_y0 = np.mean(self.data_x_p[1][max_5_points])

            self.p0_list.append([guess_amplitude, guess_offset, guess_size, guess_x0, guess_y0])

        else:
            self.p0_list = [p0, ]

            guess_amplitude = self.p0[0]
            guess_offset = self.p0[1]
            guess_size = self.p0[2]
            guess_x0 = self.p0[3]
            guess_y0 = self.p0[4]


            
        self.bounds = ([guess_amplitude/5, 0, guess_size/10, np.min(self.data_x_p[0]), np.min(self.data_x_p[1])], \
            [guess_amplitude*5, guess_offset, guess_size*10, np.max(self.data_x_p[0]), np.max(self.data_x_p[1])])
        
        self.popt_str = ['A', 'B', 'R', 'x0', 'y0']
        popt, pcov = self._fit_and_draw(is_fit, is_display, kwargs)            
        self.fit_func = 'center'
        return [self.popt_str, pcov], popt


            
    def clear(self):
        if (self.text is None) and (self.fit is None):
            return
        if self.text is not None:
            self.text.remove()
        if self.fit is not None:
            for fit in self.fit:
                fit.remove()
        for line in self.live_plot.lines:
            line.set_alpha(1)
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
        if (self.plot_type == '2D') or (self.conversion_map is None):
            return

        new_unit, conversion_func = self.conversion_map[self.unit]
        self._update_unit(conversion_func)

        ax = self.fig.axes[0]
        old_xlabel = ax.get_xlabel()
        new_xlabel = re.sub(r'\((.+)\)$', f'({new_unit})', old_xlabel)
        self.fig.axes[0].set_xlabel(new_xlabel)

        self.unit = new_unit
        self._update_transform_back()
        self.fig.canvas.draw()

    def _update_transform_back(self):
        import functools
        transforms = []
        temp_unit = self.unit
        while (self.conversion_map is not None) and (temp_unit != self.unit_original):
            try:
                next_unit, conv_func = self.conversion_map[temp_unit]
            except KeyError:
                print(f'Unit {temp_unit} not in conversion_map')
                break
            transforms.append(conv_func)
            temp_unit = next_unit

        self.transform_back = (lambda x: functools.reduce(lambda a, f: f(a), transforms, x)) if transforms else lambda x: x


    def register_selector_callback(self, selector_i, func):
        # enable GUI passes func as the callback to track selector changes, for realtime update range
        # call after choose_selector() to reset selector.callback to func
        if self.selector is None:
            print('No selector to register callback')
            return
        self.selector[selector_i].callback = func



 



















