#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.colors as colors
import ipywidgets as ipw
import pandas as pd
import numpy as np
import numpy.ma as ma
import sesd
from datetime import datetime
from tqdm import tqdm
from IPython.display import clear_output
from halo import HaloNotebook as Halo # For spinner
import time

from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore") #UI can get fucked if any warnings come up.

# To prevent automatic figure display when execution of the cell ends
#%matplotlib notebook


# In[2]:


# Read data and some basic processing
# this will be replaced by a file picker and some library code in the non-demo version
data = pd.read_csv("https://raw.githubusercontent.com/epswartz/sample_forecasting_tool/master/sample_data.csv")
fmt = "%m/%d/%y %I:%M %p"
data.time = pd.to_datetime(data.time, format=fmt)

# Aggregate all 3 into monthly sums
year_month_start = 1 + (2017*12)
data['monthly-time'] = data.time.apply(lambda x: (x.year*12) + x.month - year_month_start)

util_series_names = data.drop(['time', 'monthly-time'], axis=1).columns


# In[3]:


# Declare all the widgets
util_picker = ipw.Dropdown(
    style={'description_width': 'initial'},
    options=[
        ('Steam', 'stm'),
        ('Chilled Water', 'chw'),
        ('Hot Water', 'hhw')
    ],
    value='stm',
    description="Utility Type:"
)

# Initially blank because the options are read in from file
building_picker = ipw.Dropdown(
    style={'description_width': 'initial'},
    options=[(n,n) for n in util_series_names],
    description="Utility/Building"
   #description="Building/Collection Point:"
)

anom_toggle = ipw.Checkbox(
    style={'description_width': 'initial'},
    value=False,
    description='Clean Data with Anomaly Detection',
    disabled=True,
    layout={'object_position':'left'}
)

aggregation_picker = ipw.Dropdown(
    style={'description_width': 'initial'},
    options=[
        ('Monthly', 'month'),
        ('Weekly', 'week'),
        ('Daily', 'day')
    ],
    value='month',
    description="Aggregation Level",
    disabled=True # For demo version only monthly
)

forecast_len_field = ipw.BoundedIntText(
    style={'description_width': 'initial'},
    value=12,
    min=1,
    max=60,
    step=1,
    description='Forecast Length (Months):',
    disabled=False
)

btn_layout = ipw.Layout(width='auto', height='40px', display="inline-flex")

display_btn = ipw.Button(
    layout=btn_layout,
    style={'button_color': 'darkgreen'},
    description="Display Forecast",
    disabled=False,
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='After choosing settings, click to display forecast time chart',
    #icon='fa-line-chart'
)

export_btn = ipw.Button(
    layout=btn_layout,
    style={'button_color': 'midnightblue'},
    description="Export Forecast CSV",
    disabled=False,
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip="After choosing settings, click to export data",
    #icon='fa-save'
)


clear_btn = ipw.Button(
    layout=btn_layout,
    style={'button_color': 'crimson'},
    description="Clear Plots",
    disabled=False,
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip="Deletes all currently shown plots."
    #icon='fa-save'
)

btn_box = ipw.HBox([display_btn, export_btn, clear_btn])

figure_display_area = ipw.Output()


# In[4]:


# Graphics/UI and direct handler functions
def render():
    """
    Becuase the content of some UI elements depends on the settings in others,
    the UI needs to be re-displayed whenever something is changed.
    
    This gets called once at program start, and then again after every setting change.
    """
    
    #display(util_picker) # No util picker for demo.
    
    # Get building options from loaded file
    building_picker.options = [(n,n) for n in util_series_names]
    display(building_picker)
    
    display(anom_toggle)
    
    display(aggregation_picker)
    
    display(forecast_len_field)
    
    # TODO build and display normalization options for external data
    
    display(btn_box)
    
    display(figure_display_area)

def on_change():
    """
    Called whenever any UI element is changed.
    
    Responsible for changing active_series, an identifier (file and column name) for the series of interest.
    Also responsible for calling render().
    """
    render()
    
# TODO split the data processing part of this out into something that gets called by this and
# the export handler.
def display_forecast_on_click(b):
    """
    Runs whenever the "Display Forecast" button is clicked.
    
    Does the forecasting if needed, and show the plot.
    """
    global current_forecast_key
    global current_forecast_ts
    global current_aggregated_data
    
    
    #spinner = Halo(text='Loading', spinner='dots')
    #spinner.start()
    
    # If we've already done the calculation, just do the plotting.
    # FIXME This needs to check other settings
    if current_forecast_key == building_picker.value:
        show_forecast_plot()
        return
    
    current_forecast_key = building_picker.value
    
    # Aggregate data into monthly and convert to numpy
    monthly_sum = data.groupby(['monthly-time'])[building_picker.value].agg('sum')
    ts = monthly_sum.to_numpy()
    current_aggregated_data = ts
    
    # Make forecast with ARIMA
    model = ARIMA(ts, order=(12,1,0))
    model_fit = model.fit()
    
    current_forecast_ts = model_fit.forecast(steps=forecast_len_field.value)
    
    show_forecast_plot()
    
    #spinner.stop()

# Have to wrap the clear fn to take care of the param
def clear_btn_on_click(b):
    plt.close('all')
    figure_display_area.clear_output()


# In[5]:


# Functions for plotting

def show_forecast_plot():
    with figure_display_area:
        #plt.close('all')
        #figure_display_area.clear_output() # Clear the old plot
        
        # Something north of here is async and I don't seem to have any API to mess with it,
        # but it makes the UI fuck up sometimes. For now, I'm going to add a 1 second sleep
        # and just pray that nobody comes to (rightfully) kill me in my sleep.
        #time.sleep(1) # God help me.
        
        plt.figure(figsize=(12,8))
        plt.title(f"Forecasting: {current_forecast_key}", size=15)
        plt.plot(current_aggregated_data, label="Training Data", marker="o")
        plt.plot(np.arange(forecast_len_field.value) + len(current_aggregated_data), current_forecast_ts, color="orange", label="Forecast", marker="o")
        plt.ylabel("Utility Usage", size=12)
        plt.legend()
        plt.show()


# In[6]:


# Functions for data cleaning and imputation
# It's possible this should be a class inside a module - let's discuss that with Grace

def clean_data():
    pass


# In[7]:


# Functions for forecasting
# This should almost certainly be a module - discuss with Shota

def forecast_sarimax():
    pass


# In[8]:


# Configure on_click and on_change behaviors
display_btn.on_click(display_forecast_on_click)
clear_btn.on_click(clear_btn_on_click)

# Dummy forecast globals
current_forecast_key = ""
current_forecast_ts = np.asarray([])
current_aggregated_data = np.asarray([])

# Start the app
render()

# May need to somehow declare current_forecast_ts_df and current_forecast_key here.

