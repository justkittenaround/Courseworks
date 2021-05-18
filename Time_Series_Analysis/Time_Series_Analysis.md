---
bibliography:
- 'bibliography.bib'
title: Time Series Analysis
---

**Time Series Analysis**\
[Getting Started Workbook]{}

 

<span style="font-variant:small-caps;">Spring Semester, Florida Atlantic
University</span>\
<span
style="font-variant:small-caps;">github.com/justkittenaround/Coursework/Time\_Series\_Analysis</span>\
This research was done under the supervision of Dr. William Edward Hahn
from January 11th to May 6th.\
*First release, May 2020*

Introduction
============

Objective
---------

This workbook seeks to provide the basic tools and information for
time-series analysis in way that is efficient for practical
applications. The structure of this notebook serves as the general
pipeline any time-series data project could be developed in and two
fully developed applications to real-world data. Elementary concepts
will be described in a series of short coding examples that gradually
increase in complexity. These introductions are followed by a more
integrative exploratory analysis of audio data. Finally, full projects
for real data are shared in step-by-step fashion. In writing this
workbook, I have tried to simplify the process by explaining the most
key elements while working through code examples, leaving out
explanations for some of the more standard question that could be
answered quickly by external resources. It is my hope that anyone using
this notebook as an intro into time-series analysis will develop not
only the awareness of technical skills needed to perform analysis but
also some sense of style in the general approach to doing so. At the end
of this workbook, you should have several examples to add to your
toolbox in approaching future problems.

The Workspace
-------------

Code is written in Python3 and will be provided as snippets throughout
the workbook. Full implementations will be linked to a Github repository
of Ipython notebooks that can be run with Google Colaboratory. Comments
are denotd with the \# symbol and appear in green text. It may be
beneficial to the reader to write line-by-line into a Colaboratory
workbook alongside with reading through this document, making extra code
comments when necessary to deepen the understanding of the code
implementation. Each chapter is meant to be its own Collab Notebook
file. If your not familiar with Google Colaboratory, read the tutorial
[here](https://colab.research.google.com/notebooks/intro.ipynb). While
you don’t need to know the Python programming language, having a basic
understanding of how it works is beneficial. This short
[book](https://canvas.fau.edu/courses/75705/files/17384552/download?wrap=1)
by Jamie Chan can serve as the basic pre-requisite information needed.
Make sure to save your work as some notebooks will be completed in
stages as we build upon the skills as we go.

Obtaining and Visualizing Data
==============================

The first step in any time-series analysis is understanding your
time-series. This may sound redundant, but thouroughly knowing your data
from the beginning stages of your approach can save you a lot of time
and make your journey working with the data much easier. First things
first, you’ll need to gather data. Real-world datasets are not often in
forms that are ready for analysing. They may have missing values,
improper entries, contain unnecessary parts, not be formatted correctly,
etc. Another reason we need to really try and understand our data is so
we know what we are working with. Often, before we can start to think
about how to extract information, we need to understand on a basic
level, what information this data has to offer us. Understanding your
data will come as you work with it more and more. Try to think about the
limitations of the data; what would make it better? What are the
strengths of the dataset, what does the information rely upon? What
assumptions might I be making about this data? All of these questions
should be considered before any serious analysis can be attempted. So,
how do we get data and how do we get to know it? While there are many
ways to investigate and clean the data, a good place is to start is just
by taking a look at it from various angles. We can do this by plotting
our time-series using quick visualization techniques.

Generating and Plotting
-----------------------

Time-series analysis implies there is a time-series component. Namely, a
collection of data points that incorporate some element of time. Most
information in the world can be represented as data with some degree of
temporal resolution. Understanding the data is the key beginning step in
preforming an analysis. While this may be a seemingly intuitive step,
underestimating the importance of getting to know your data could cause
poor choices in model design, false reliability in model accuracy, and
other downstream complications. Ultimately, good data can lead to more
efficient development and creating a better understanding of the
analysis as a whole. Instead of working with real-world datasets that
need pre-processing steps, we can have more control over the integrity
of our data if we create it ourselves. In this section, we’ll generate
our own data and visualize it while trying out different techniques.

### Random Data

It may be useful to create some random data to use in practice for
learning how to handle the basics of data and visualization techniques.
Most important we need to know how to hold our data in the computer and
how to look at it. In the following three exercises we will use python
library package numpy to generate random numerical data samples and
matplotlib.pyplot to plot our data in different ways. It’s important to
note, since we are randomly generating the data, if you do this exercise
on your own it will be slightly different as you random numbers will be
different unless you use the numpy seed function which will preserve the
original random numbers. Don’t worry about all the details of numpy or
matplotlib.pyplot, just know they are tools that other teams of
developers have made for us to use.

**As a exercise for the reader:** Imagine for each of the following
three examples a real-world example each dataset and plot could
represent, jot these down in the comments of your own workbook.

    '''In this first example, we will create 3 different datasets of random numbers in a normal distribution and plot them all as histograms on the same plot.'''

    #for using matplotlib in Google Colaboratory
    %matplotlib inline 

    #access important python libraries
    import numpy as np
    import matplotlib.pyplot as plt



    #specify some traits of the data that we will need for the plot
    number_of_bins = 20                                #group similar entries into 20 groups
    number_of_data_points = 93                         #the number of datapoints we want to make
    labels = ["sample 1", "sample 2", "sample 3"]      #names of the groups we will separate the datapoints into



    #make 3 random datasets using the numpy (np) random normal function in the form of (mean, standard deviation, size) and store them in a list 
    data_sets = [np.random.normal(0, 1, number_of_data_points),
                 np.random.normal(6, 1, number_of_data_points),
                 np.random.normal(3, 1, number_of_data_points)]
                 
                 
                 
    #the rest of the code is available from matplotlib examples. So for now, we just need to know it is going to make a histogram plot of all three of our datasets
    hist_range = (np.min(data_sets), np.max(data_sets)) 
    binned_data_sets = [np.histogram(d, range=hist_range, bins=number_of_bins)[0] for d in data_sets]
    binned_maximums = np.max(binned_data_sets, axis=1)
    x_locations = np.arange(0, sum(binned_maximums), np.max(binned_maximums))
    bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
    centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]  #find where to place the 
    heights = np.diff(bin_edges)
    fig, ax = plt.subplots()
    ax.set_xticks(x_locations)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Data values")
    plt.title('Multi-histogram Plot of Three Datasets')
    for x_loc, binned_data in zip(x_locations, binned_data_sets):
        lefts = x_loc - 0.5 * binned_data
        ax.barh(centers, binned_data, height=heights, left=lefts)
        
    #show us the dataset (if your in Colaboratory, you won't need this function)
    plt.show()

![Plotting multiple datasets as histograms on one
plot.[]{data-label="fig:galaxy"}](Pictures/Multi-hist.png){width="75.00000%"}

    '''In this second example, we will create two separate datasets of random numbers in a specified range and from a gamma distribution then plot the information as separate event plots. An event plot shows sequences of events with various line properties. The plot is shown in both horizontal and vertical orientations.'''

    %matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    matplotlib.rcParams['font.size'] = 8.0 #set a standard font size for matplotlib


    #a random seed will allow all random numbers to be "saved" so they are the same even if you call the function again
    np.random.seed(19680801)


    # create random dataset 1 from range 0 to 6 of 50 numbers 
    data1 = np.random.random([6, 50])
    print(data1)  #if you want to see what the random numbers are

    #functions from matplotlib for organizing the first dataset and preparing it for the plot
    colors1 = ['C{}'.format(i) for i in range(6)]
    lineoffsets1 = np.array([-15, -3, 1, 1.5, 6, 10])
    linelengths1 = [5, 2, 1, 1, 3, 1.5]
    fig, axs = plt.subplots(2)
    axs[0].eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
                        linelengths=linelengths1)
    axs[1].eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
                        linelengths=linelengths1, orientation='vertical')
                        
    -----------------------------------------------------------
    # create another set of random data.
    # the gamma distribution is only used for aesthetic purposes and to see a different distribution of datapoints
    data2 = np.random.gamma(4, size=[60, 50])


    #functions from matplotlib for organizing the first dataset and make the plots
    fig, axs = plt.subplots(2)
    colors2 = 'black'
    lineoffsets2 = 1
    linelengths2 = 1
    axs[0].eventplot(data2, colors=colors2, lineoffsets=lineoffsets2,
                        linelengths=linelengths2)
    axs[1].eventplot(data2, colors=colors2, lineoffsets=lineoffsets2,
                        linelengths=linelengths2, orientation='vertical')

![Vertical (bottom) and Horizontal (top) Event Plots of Range (left) and
Gamma (right)
Datasets](Pictures/data-event.png "fig:"){width=".8\linewidth"}
\[fig:test1\]

![Vertical (bottom) and Horizontal (top) Event Plots of Range (left) and
Gamma (right)
Datasets](Pictures/gama-event.png "fig:"){width=".8\linewidth"}
\[fig:test2\]

    '''In this third example, we will create one dataset of an array of random numbers and plot them in four different spy plots, all with different plotting parameters. Spy plots visualize the non-zero values of the array, they plot sparsity.'''

    %matplotlib inline 
    import matplotlib.pyplot as plt
    import numpy as np

    #create the plotting area and define coordinates of each plot
    fig, axs = plt.subplots(2, 2) 
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]

    #make data arrays of random numbers then make some entries 0
    x = np.random.randn(24, 31)
    x[5, :] = 0.
    x[:, 12] = 0.

    #plot sparsities of X
    ax1.spy(x, markersize=5) # mark
    ax2.spy(x, precision=0.1, markersize=5) #precision is the threshold
    ax3.spy(x)
    ax4.spy(x, precision=0.5)

    #print the shape of the array and the first row of the array
    print(x.shape, x[0])

For a compiled Google Collaboratory notebook of the above random data
generation and plotting examples, see at
[Matplotlib-Time-Series-Tutorial.ipynb. After you get these examples
working, feel free to play around with some of the code and see how the
plots
change.](https://github.com/justkittenaround/Courseworks/blob/master/Time_Series_Analysis/Matplotlib-Time-Series-Tutorials.ipynb)

![Plotting one dataset array as spy plots with different
parameters.[]{data-label="fig:galaxy"}](Pictures/spy.png){width="75.00000%"}

### Mathematical Functions

    '''We can also generate random data and apply mathematical functions as time-series information.
    '''
    #import some libraries
    import numpy as np
    from scipy import signal
    import matplotlib.pyplot as plt

    #create and array of numbers to as input to the functions
    x = np.linspace(-2*np.pi, 2*np.pi, 1000)

    #create sine and cosine waves of x
    y = np.sin(x)
    y2 = np.cos(x)

    #plot the sin waves
    plt.plot(y, 'r')
    plt.plot(y2, 'b')

    #we can use scipy package to make a sawtooth signal and plot it using matplotlib package
    t = np.linspace(0, 1, 500)
    y = signal.sawtooth(2 * np.pi * 5 * t)
    plt.plot(t, y)

    #we can do the same for a square wave
    framerate = 44100
    t = np.linspace(0,5,framerate*5)
    data = signal.square(2*np.pi*t) 
    plt.plot(t, data)
    plt.title('Square Wave')

![Plots of manually generated sine and cosine functions (left), sawtooth
signal (middle), and square wave
(right).](Pictures/sin.png "fig:"){width="1\linewidth"} \[fig:test1\]

![Plots of manually generated sine and cosine functions (left), sawtooth
signal (middle), and square wave
(right).](Pictures/sawtooth.png "fig:"){width="1\linewidth"}
\[fig:test1\]

![Plots of manually generated sine and cosine functions (left), sawtooth
signal (middle), and square wave
(right).](Pictures/square.png "fig:"){width="1\linewidth"} \[fig:test2\]

### Summary

We’ve now seen how to generate our own data using different numpy
functions and how to plot this data with different matplotlib examples.
Some of the data was generated at random, some from mathematical
distributions. We’ve also gained more experience in how to use python
packages to write code. Hopefully you took the time to think about how
these plots could explain different real world examples. In all
examples, we could use the plots to compare two different datasets
quickly. The experience of what plot to use when will depend on your
data and become more intuitive with experience.

Loading Existing Data with Pandas
---------------------------------

Now we can move on to real-world examples. We will collect data
ourselves and organize it using the comprehensive data library Pandas.
Don’t worry about the complexities of Pandas functions, what is
important in this section is learning how to import the data and store
it in a way the computer can access it efficiently.

### Google Trends

    '''In this example, we will load data directly from \href{https://trends.google.com/trends/?geo=US}{Google Trends} into a pandas dataframe and plot the data using a simple line plot using matplotlib.
    '''

    #install the API package for accessing google trends with python3
    pip install pytrends

    #import libraries
    import pytrends
    import pandas as pd
    import matplotlib.pyplot as plt
    from pytrends.request import TrendReq

    #register the pytrends request to access google trends
    pytrends = TrendReq(hl = 'en-US', tz=360)

    #define the key words for the trend search
    kw_list = ["cats","dogs"]

    #get the trend data according to the specified arguments and keywords
    search_df = pytrends.get_historical_interest(kw_list, year_start=2017, month_start=1,day_start=1,hour_start=0, year_end=2017, month_end=2,day_end=1,hour_end=0, cat=0, geo='', gprop='', sleep=0)

    #let's view the raw data and see hot the cat data is stored in the dataframe
    print(search_df)
    search_df['cats'].shape

    #separate the cat and dog data
    cat_data = search_df['cats']
    dog_data = search_df['dogs']

    #view the cat and dog data
    plt.plot(cat_data,'r')
    plt.plot(dog_data,'b')

![Raw dataframe (left) and double line plot (right) of cat (red) and dog
trends in the past 3 years from Google
Trends.](Pictures/df.png "fig:"){width=".8\linewidth"} \[fig:test1\]

![Raw dataframe (left) and double line plot (right) of cat (red) and dog
trends in the past 3 years from Google
Trends.](Pictures/cat_dog.png "fig:"){width="1.2\linewidth"}
\[fig:test2\]

### Stocks

    ''' This example is an extension of the previous example which we learned how to get Google Trend data. Here, we access bitcoin trading data from Bitstamp and Bitcoin and Ethereum trend data from Google Trends, save them both as dataframes, and visualize some information from the datasets in tandem. 
    '''
    #import useful libraries for this task
    import requests
    from datetime import datetime
    import pandas as pd
    import matplotlib.pyplot as plt

    #initialize the key words for Google Trends
    kw_list = ["bitcoin","ethereum"]

    #collect the trend data
    search_df = pytrends.get_historical_interest(kw_list, year_start=2014, month_start=1,day_start=1,hour_start=0, year_end=2018, month_end=1, day_end=1,hour_end=0, cat=0, geo='', gprop='', sleep=0)

    #separate the data per key word
    bitcoin_data = search_df['bitcoin']
    ethereum_data = search_df['ethereum']

    #visualize the trend data as a quick check 
    print(bitcoin_data)
    plt.plot(bitcoin_data[-1000:],'.')
    plt.plot(ethereum_data)

![Raw dataframe of Bitcoin data (left), scatter plot of Bitcoin trend
data (middle) and line plot of Ethereum trend data
(right).](Pictures/rawbit.png "fig:"){width="1\linewidth"} \[fig:test1\]

![Raw dataframe of Bitcoin data (left), scatter plot of Bitcoin trend
data (middle) and line plot of Ethereum trend data
(right).](Pictures/bitplot.png "fig:"){width="1\linewidth"}
\[fig:test1\]

![Raw dataframe of Bitcoin data (left), scatter plot of Bitcoin trend
data (middle) and line plot of Ethereum trend data
(right).](Pictures/ethplot.png "fig:"){width="1\linewidth"}
\[fig:test2\]

    ''' python example 2.6 continued
    '''

    #define parameters for retrieving the bitcoin trading data
    from_symbol = 'BTC'
    to_symbol = 'USD'
    exchange = 'Bitstamp'
    datetime_interval = 'day'

    #define a function for formatting the filename of data retrieval from Bitstamp so we can save it as a document later
    def get_filename(from_symbol, to_symbol, exchange, datetime_interval, download_date):
        return '%s_%s_%s_%s_%s.csv' % (from_symbol, to_symbol, exchange, datetime_interval, download_date)


    #how to perform the retreival request from Bitstamp according to our parameters
    def download_data(from_symbol, to_symbol, exchange, datetime_interval):
        supported_intervals = {'minute', 'hour', 'day'}
        assert datetime_interval in supported_intervals,           'datetime_interval should be one of %s' % supported_intervals

        print('Downloading %s trading data for %s %s from %s' %
              (datetime_interval, from_symbol, to_symbol, exchange))
        base_url = 'https://min-api.cryptocompare.com/data/histo'
        url = '%s%s' % (base_url, datetime_interval)

        params = {'fsym': from_symbol, 'tsym': to_symbol,
                  'limit': 2000, 'aggregate': 1,
                  'e': exchange}
        request = requests.get(url, params=params)
        data = request.json()
        return data

    #store the data as a pandas dataframe for easy access 
    def convert_to_dataframe(data):
        df = pd.io.json.json_normalize(data, ['Data'])
        df['datetime'] = pd.to_datetime(df.time, unit='s')
        df = df[['datetime', 'low', 'high', 'open',
                 'close', 'volumefrom', 'volumeto']]
        return df

    #clean up the data from null entries
    def filter_empty_datapoints(df):
        indices = df[df.sum(axis=1) == 0].index
        print('Filtering %d empty datapoints' % indices.shape[0])
        df = df.drop(indices)
        return df

    #perform the data retrieval, storage, and clean-up
    data = download_data(from_symbol, to_symbol, exchange, datetime_interval)
    df = convert_to_dataframe(data)
    df = filter_empty_datapoints(df)

    #save the data as a csv for archival purposes
    current_datetime = datetime.now().date().isoformat()
    filename = get_filename(from_symbol, to_symbol, exchange, datetime_interval, current_datetime)
    print('Saving data to %s' % filename)
    df.to_csv(filename, index=False)

    #if you want to read in the file at a later time
    def read_dataset(filename):
        print('Reading data from %s' % filename)
        df = pd.read_csv(filename)
        df.datetime = pd.to_datetime(df.datetime) # change type from object to datetime
        df = df.set_index('datetime') 
        df = df.sort_index() # sort by datetime
        print(df.shape)
        return df

    df = read_dataset(filename)

    #let's take a look at some of the raw data
    print(df)
    print(df['close']

    #separate out the closing price information
    bitcoin_price = df['close']

    #now we can plot both Bitcoin datasets to manually access the information
    fig, ax1 = plt.subplots()
    t = np.arange(0.,1000,1) #to share the information on the same x-axis
    color = 'tab:red'
    ax1.set_ylabel('searches')
    ax2.set_xlabel('days', color=color) 
    ax1.plot(t, bitcoin_data[-1000:], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('price (usd)')
    ax2.plot(t, bitcoin_price[-1000:], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Bitcoin Google Trend Searches and Closing Price')
    fig.tight_layout() 

![Bitcoin Google Trends (red) compared to recent history of Bitcoin
closing prices
(blue).[]{data-label="fig:galaxy"}](Pictures/finalbtc.png){width="75.00000%"}

For a compiled Google Collaboratory notebook of the above trend and
trading data importation, accessing, and plotting examples, see
[TimeSeries\_Google\_Trends+bitcoin.ipynb](https://github.com/justkittenaround/Courseworks/blob/master/Time_Series_Analysis/Copy%20of%20TimeSeries_Google_Trends%2Bbitcoin.ipynb)

### Summary

The examples we have just explored allowe dus to get real-world data and
read it into our coding environment. We stored our data use the Pandas’
dataframe operator, which allow us to access different features of the
data easily in an organized fashion. We used simple line plots from
matplotlib to see how or trends look over time. Now you should be able
to collect your own data in the same examples and make comparisons
against two different trends.

Investigating and Pre-processing Data
=====================================

Now that we’ve gotten to know a couple of techniques for getting to know
or data, we can continue the process here. In this chapter, we will look
at how to clean up our datasets for easy access and storage. Then we can
some smoothing techniques to better understand our data and visualize
noisy datasets. As you move through this chapter, the basic focus is on
how to gather, sort, and transform data. At the very end of the chapter,
we will compile all of these tools to make our first end-to-end example
to predict future datapoints using convolutional neural networks. While
we won’t go into details about the network and the inter-workings of
each function, we can explore their utility and basic structure for
future use.

Preparing Data with Smoothing Operations
----------------------------------------

A simple explanation of smoothing data, is just to make outliers more
subtle and see the general trend of the data. This can be done with
several different functions. As we’ll see later, how some of these
functions operate are the basis of more complex analysis techniques.

### Storing, Formatting, and Accessing Data

As we’ve learned in the previous chapter, we first need to get our data
in an accessible form. The following section will show more examples of
how to access data from files, separate data, and format features of the
data.

    ''' In this example, we will import a pre-saved passenger count dataset file from our Google Drive and organize the data for later use.
    '''

    %matplotlib inline
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    from pandas import Series
    import warnings
    warnings.filterwarnings("ignore") #ignore filter warnings so collaboratory doesn't print a bunch of stuff
    plt.style.use('fivethirtyeight') #set a plotting style for matplotlib for aesthetics

    #mount google drive so we can access the data folder
    from google.colab import drive
    drive.mount('/content/drive')

    #gather the filenames of the data folder
    import os
    os.listdir('/content/drive/My Drive/School/Time_Series_Analysis/Data/')

    #read in the data files as panda dataframes (separated into train and test for later use)
    train = pd.read_csv("/content/drive/My Drive/School/Time_Series_Analysis/Data/Train_SU63ISt.csv")
    test = pd.read_csv("/content/drive/My Drive/School/Time_Series_Analysis/Data/Test_0qrQsBZ.csv")

    #view the raw data, data shapes, and check how the information is stored (dtype)
    print('train cols', train.columns, 'test cols', test.columns)
    print('train shape', train.shape, 'test shape', test.shape)
    print(train.dtypes,test.dtypes)

    #check if any values are missing
    train.isnull().values.any()

    #make copies of the original datasets for safe keeping
    train_original = train.copy()
    test_original = test.copy()

    #set the datetime information in the datasets to have a day, month, year, hour, minute format (this will allow us to access the information in the dataframe by indexing any of the formatted values)
    train['Datetime'] = pd.to_datetime(train.Datetime, format = '%d-%m-%Y %H:%M')
    test['Datetime'] = pd.to_datetime(test.Datetime, format = '%d-%m-%Y %H:%M')
    for i in (train, test):
        i['year'] = i.Datetime.dt.year
        i['month'] = i.Datetime.dt.month
        i['day']= i.Datetime.dt.day
        i['Hour']=i.Datetime.dt.hour
        
    #set the datetime information in the dataset to have a day of the week format 
    train['Day of week'] = train['Datetime'].dt.dayofweek

    #set datetime information in the dataset to be formatted as weekend if applicable
    def applyer(row):
        if row.dayofweek == 5 or row.dayofweek == 6:
            return 1
        else:
            return 0
        
    train['weekend'] = train['Datetime'].apply(applyer)

    #plot the passenger count per year-month
    train.index = train['Datetime']
    df = train.drop('ID',1)
    ts = df['Count']
    plt.figure(figsize = (16,8))
    plt.plot(ts)
    plt.title("Time Series")
    plt.xlabel("Time (year-month)")
    plt.ylabel("Passenger Count")
    plt.legend(loc = 'best')

    #plot the passenger count by year only
    train.groupby('year')['Count'].mean().plot.bar()

    #plot the passenger count by month only
    train.groupby('month')['Count'].mean().plot.bar()

    #plot the average passenger count monthwise
    temp = train.groupby(['year', 'month'])['Count'].mean()
    temp.plot(figsize =(15,5), title = "Passenger Count(Monthwise)", fontsize = 14)

    #we can reformate and separate the dataset to by hourly, daily, weekly, and monthly trends 
    train.Timestamp = pd.to_datetime(train.Datetime, format = '%d-%m-%y %H:%M')
    train.index = train.Timestamp

    #Hourly
    hourly = train.resample('H').mean()

    #Daily
    daily = train.resample('D').mean()

    #Weekly
    weekly = train.resample('W').mean()

    #Monthly
    monthly = train.resample('M').mean()

    #now we can plot these trends together
    fig,axs = plt.subplots(4,1)
    hourly.Count.plot(figsize = (15,8), title = "Hourly", fontsize = 10, ax = axs[0])
    daily.Count.plot(figsize = (15,8), title = "Daily", fontsize = 10, ax = axs[1])
    weekly.Count.plot(figsize = (15,8), title = "Weekly", fontsize = 10, ax = axs[2])
    monthly.Count.plot(figsize = (15,8), title = "Monthly", fontsize = 10, ax = axs[3])

    #separate the data into training and validation sets (for later),
    Train = train.ix['2012-08-25':'2014-06-24']
    valid = train.ix['2014-06-25':'2014-09-25']

    #plot training and validation (later we can use the training data to predict the forecast for the time-series information and validate our predictions with the validation set
    Train.Count.plot(figsize = (15,8), title = 'Daily Ridership', fontsize = 14, label = 'Train')
    valid.Count.plot(figsize = (15,8), title = 'Daily Ridership', fontsize =14, label = 'Valid')
    plt.xlabel('Datetime')
    plt.ylabel('Passenger Count')
    plt.legend(loc = 'best')

![Passenger count per visualized on different time-scales (left:
year-month, middle: year, right:
month).](Pictures/year-month.png "fig:"){width=".8\linewidth"}
\[fig:test1\]

![Passenger count per visualized on different time-scales (left:
year-month, middle: year, right:
month).](Pictures/year-only.png "fig:"){width=".8\linewidth"}
\[fig:test1\]

![Passenger count per visualized on different time-scales (left:
year-month, middle: year, right:
month).](Pictures/month-only.png "fig:"){width=".8\linewidth"}
\[fig:test2\]

![Monthly passenger count
trend.](Pictures/monthwise.png "fig:"){width="1\linewidth"}
\[fig:test1\]

![Different timescales for monthly passenger counts (left) and training
and validation dataset splits for monthly passenger counts
(right).](Pictures/trends.png "fig:"){width=".8\linewidth"}
\[fig:test1\]

![Different timescales for monthly passenger counts (left) and training
and validation dataset splits for monthly passenger counts
(right).](Pictures/train-val.png "fig:"){width=".8\linewidth"}
\[fig:test2\]

### Simple Exponential Smoothing

    ''' In this example, we will use the previously prepared data to explore a pre-built smoothing and forecasting function from statsmodel package to predict the 'valid' portion of our data trend. We then plot the original data and our forecast to see our results.'''

    from statsmodels.tsa.api import ExponentialSmoothing,SimpleExpSmoothing

    y_hat = valid.copy()
    fit2 = SimpleExpSmoothing(np.asarray(Train['Count'])).fit(smoothing_level = 0.6,optimized = False)
    y_hat['SES'] = fit2.forecast(len(valid))
    plt.figure(figsize =(15,8))
    plt.plot(Train['Count'], label = 'Train')
    plt.plot(valid['Count'], label = 'Validation')
    plt.plot(y_hat['SES'], labe

![Simple exponential smoothing of validation portion of the passenger
count data using statsmodel built-in
functions.](Pictures/exp-smooth.png "fig:"){width="1\linewidth"}
\[fig:test1\]

### Simple Convolutions with Randomly Generated Data

As we can see in the previous example, the pre-built functions, while
easy to implement, aren’t very accurate for this dataset. In this next
section, we will use a smoothing technique by convolutions. To better
understand this transformation, we will use randomly generated data.
Convolutions are an important mathematical operation that we will use
many times throughout the workbook. There are many ways to do
convolutions, here we will show a few. The point of a convolution is to
smooth the data by looking at surrounding datapoints in the trend and
using matrix multiplication to produce a transformed version of the
trend. We can think of the trend data as a string of numbers. We will
slide a small window of multipliers across the string to generate a new
value for each combination of string-to-window positions. The result is
a smoothed trendline that has been transformed since every datapoint in
the original trend.

    ''' In this example, we will generate random noisey data and perform various methods of convolutions. '''

    #create some random numbers        
    t = np.linspace(-4,4,100)

    #apply a sine wave function
    x = np.sin(t)

    #let's view the original data as a plot and its raw values
    plt.plot(x)
    print(x)

    #add some noise to the data so we can practice smoothing the noise with convolutions
    xn =  x + 0.1 * np.random.randn(len(x))

    #view the noisey data
    plt.plot(xn)

    #convolutions can be manually computed by with numpy's built-in operator
    w = np.ones(10, 'd') #create a filter window
    w = w/w.sum() #define the filter to be the average
    y = np.convolve(w, xn) #use numpy's convolve operation

    #plot the original noisey data and the smoothed data
    plt.plot(xn, 'r')
    plt.plot(y)


    #convolutions can also be manually computed by simple mathematical operations
    win_size=5 # define the window size to convolve over the data
    w = np.ones(win_size)/win_size 
    y = np.zeros(len(xn)) #set a place-holder for the outputs of the computed convolutions

    #iterate over the data and perform the computations of convolution according to the window size
    for i in range(int((win_size-1)/2), len(xn)-(int((win_size-1)/2))):
       y[i] =  xn[i-2] * w[0] + xn[i-1] * w[1] + xn[i] * w[2] + xn[i+1] * w[3] + xn[i+2] * w[4]

    #now we can view the convolutions compared to the original noisey data
    plt.plot(xn)
    plt.plot(y)

    #another way to convolve is by the dot product of the filter and data window
    win_size=7
    w = np.ones(win_size)/win_size
    y = np.zeros(len(xn))
    a = int((win_size-1)/2)

    for i in range(a, len(xn)-a):
        xi = xn[ i-a : i + a +1] #make window
        y[i] =  np.dot(w,xi)

    plt.plot(xn)
    plt.plot(y)


    #another way to convolve is by python's broadcasting operation
    from skimage.util.shape import view_as_windows as vaw

    window_shape = (7,1)

    #we can use the skimage view as windows function to seperate our data as small windows to which we can apply the convolution operation
    xw = vaw(xn, window_shape) 

    #broadcast the data windows with the filter window
    xw2 = w*xw

    #sum or dot product can be used to complete the convolution computation
    y = np.sum(xw2, 1)
    yd = np.dot(xw, w)

![Original generated data (left). Convolved data (blue) with original
data (red) using the built-in numpy
function(right).](Pictures/noisey.png "fig:"){width=".8\linewidth"}
\[fig:test1\]

![Original generated data (left). Convolved data (blue) with original
data (red) using the built-in numpy
function(right).](Pictures/simple-convs0.png "fig:"){width=".8\linewidth"}
\[fig:test2\]

![Original (blue) and convolved (red) trends using manual sliding window
(left) and dot product method
(right).](Pictures/simple-convs.png "fig:"){width=".8\linewidth"}
\[fig:test1\]

![Original (blue) and convolved (red) trends using manual sliding window
(left) and dot product method
(right).](Pictures/simple-convs2.png "fig:"){width=".8\linewidth"}
\[fig:test2\]

As an exercise for reader, try to apply any of the convolution smoothing
operations to the prepared passenger count data.

### Summary

In the previous sections, we’ve learned to format real data, smooth and
forecast with pre-built functions. Several methods of convolutions were
explored on random datasets using numpy, iterative window array
multiplication, window dot product and broadcasting. Convolutions
transform each datapoint by multiply it (and its neighbors) by an
window. The window is a selection of values that will date a fraction of
the datapoints’ values. This results in a new value for each combination
and a smoother trend. Hopefully you have an understanding of what
happens we we apply a convolution.

Dilated Convolutions with Stock Market Data
-------------------------------------------

We will use the previous knowledge of data pre-processing and
convolutions to construct a complete forecasting example. This means we
will read in our data, prepare it, construct and convolution neural
network to predict values of our dataset. You may want to revisit this
section after chapter four on forecasting (as this is an advanced
forecasting technique). However, I’ve included it hear since the
principal concept are convolutions. The basic understanding needed for
neural networks in this example relies upon a few concepts: network
architecture, backpropagation, and training/test datasets. The data,
needs to be divided into two parts. Training datasets will be used by
the network to learn features. The test set will be used by the network
to asses how accurately it preformed the learning portion. Training
precedes testing. During training, the network will iterate through each
datapoint and create weights (eg. parameters) that can be used to
describe the data. Weights, in this network, are the the multiplicative
values in the sliding window of our convolutions operators in the
previous examples. The weights aren’t assigned manually in this neural
network as we did earlier, they’re learned through a process called
backpropagation. In this process, the negative derivative with respect
to each parameter of the loss (difference between the prediction and the
actual value) for each datapoint is computed. These values (called the
gradient) are combined with a constant (called the learning rate) to
update the parameters. For this example, the important part is that
backpropagation does automatically assign weight values (which we call
parameters or weights) that are used to make predictions when new
un-trained data is seen by the network. In order for all of this to
happen, we need to create the parameters. These are described by the
network architecture. In our network, we will use a 2d-dialated
convolution operater followed by a rectified linear unit function
(ReLU). The mathematical operator ReLU transforms any parameters with
values below zeros to zero. We’ll stack these operators in layers. To
move the data through the network, we’ll describe a forward pass. This
function just retrieves the output from each layer in the network and
provides it as input to the next layer. In order to learn parameters
that represent the data well, we’ll repreat the whole proccess over
several passes (eg. epochs). In order to call upon the data easily,
we’ll create a class called a data-loader that holds various functions
we might need to access the data. The package we are going to use for
most of these operation is called PyTorch or Torch. It’s specifically
made for handling tensors (special arrays of data) for deep learning
tasks. Just like our previous examples, we’ll load in our data, clean it
up, apply some convolutions and mathematical operators, and plot our
predictions with our original data.

    '''In this notebook, we will download stock market SP500 data, define a dilated convolutional network, and use it to make predictions of the stock prices.'''

    #Import the important functions
    import os
    import torch
    import numpy as np
    import pandas as pd
    from torch import nn
    from torch import optim
    from matplotlib import cm
    from itertools import repeat
    from google.colab import drive
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from torch.autograd import Variable
    from torch.optim import lr_scheduler
    from torch.utils.data import DataLoader
    from torch.utils.data.dataset import Dataset
    from sklearn.preprocessing import MinMaxScaler
    from matplotlib.dates import MonthLocator, DateFormatter

    #Mount the drive and access data
    drive.mount('/gdrive')
    %cd /gdrive
    os.listdir('PATH_TO_DATA') 

    #Define the SP500 dataset creator
    class SP500(Dataset):
        def __init__(self, folder_dataset, T=10, symbols=['AAPL'], use_columns=['Date', 'Close'], start_date='2012-01-01', end_date='2015-12-31', step=1):

            self.scaler = MinMaxScaler()
            self.symbols = symbols
            self.start_date = start_date
            self.end_date = end_date
            self.use_columns = use_columns
            self.T = T
        
            self.dates = pd.date_range(self.start_date, self.end_date)
            
            self.df_data = pd.DataFrame(index=self.dates)

            for symbol in symbols:
                # fn = os.path.join(folder_dataset, symbol + "_data.csv")
                fn = "/gdrive/My Drive/Datasets/" + folder_dataset + "/" + symbol + "_data.csv"
                print(fn)
                df_current = pd.read_csv(fn, index_col='Date', usecols=self.use_columns, na_values='nan', parse_dates=True)
                df_current = df_current.rename(columns={'Close': symbol})
                self.df_data = self.df_data.join(df_current)

            # Replace NaN values with forward then backward filling
            self.df_data.fillna(method='ffill', inplace=True, axis=0)
            self.df_data.fillna(method='bfill', inplace=True, axis=0)
            
            self.numpy_data = self.df_data[self.symbols].values
            self.train_data = self.scaler.fit_transform(self.numpy_data)

            # Get history for each data point
            self.chunks = torch.FloatTensor(self.train_data).unfold(0, self.T, step).permute(0, 2, 1)

        def __getitem__(self, index):
            x = self.chunks[index, :-1, :]
            y = self.chunks[index, -1, :]
            return x, y

        def __len__(self):
            return self.chunks.size(0)
        
        
        
            
    #Define the multistep SP500 dataset creator
    class SP500Multistep(Dataset):
        def __init__(self, folder_dataset, symbols=['AAPL'], use_columns=['Date', 'Close'], start_date='2012-01-01', end_date='2015-12-31', step=1, n_in=10, n_out=5):

            self.scaler = MinMaxScaler()
            self.symbols = symbols
            self.start_date = start_date
            self.end_date = end_date
            self.use_columns = use_columns

            self.dates = pd.date_range(self.start_date, self.end_date)
            self.df_data = pd.DataFrame(index=self.dates)

            for symbol in symbols:
                fn = os.path.join(folder_dataset, symbol + "_data.csv")
                fn = "/gdrive/My Drive/Datasets/" + folder_dataset + "/" + symbol + "_data.csv"
                print(fn)
                df_current = pd.read_csv(fn, index_col='Date', usecols=self.use_columns, na_values='nan', parse_dates=True)
                df_current = df_current.rename(columns={'Close': symbol})
                self.df_data = self.df_data.join(df_current)

            # Replace missing values with forward then backward filling
            self.df_data.fillna(method='ffill', inplace=True, axis=0)
            self.df_data.fillna(method='bfill', inplace=True, axis=0)

            self.numpy_data = self.df_data[self.symbols].values
            self.train_data = self.scaler.fit_transform(self.numpy_data)

            self.chunks = []
            self.chunks_data = torch.FloatTensor(self.train_data).unfold(0, n_in+n_out, step)

            k = 0
            while k < self.chunks_data.size(0):
                self.chunks.append([self.chunks_data[k, :, :n_in], self.chunks_data[k, :, n_in:]])
                k += 1

        def __getitem__(self, index):
            x = torch.FloatTensor(self.chunks[index][0])
            y = torch.FloatTensor(self.chunks[index][1])
            return x, y

        def __len__(self):
            return len(self.chunks)
            
            
            
    #Specify the dialated convolutional neural network architecture
    class DilatedNet(nn.Module):
        def __init__(self, num_securities=5, hidden_size=64, dilation=2, T=10):

            #param num_securities: int, number of stocks
            #param hidden_size: int, size of hidden layers
            #param dilation: int, dilation value
            #param T: int, number of look back points

            super(DilatedNet, self).__init__()
            self.dilation = dilation
            self.hidden_size = hidden_size

            self.dilated_conv1 = nn.Conv1d(num_securities, hidden_size, kernel_size=2, dilation=self.dilation)
            self.relu1 = nn.ReLU()

            self.dilated_conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=self.dilation)
            self.relu2 = nn.ReLU()

            self.dilated_conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=self.dilation)
            self.relu3 = nn.ReLU()

            self.dilated_conv4 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=self.dilation)
            self.relu4 = nn.ReLU()

            self.conv_final = nn.Conv1d(hidden_size, num_securities, kernel_size=1)

            self.T = T
        
        #how to evaluate each datapoint when calling the layers of the network
        def forward(self, x):

            #batch_size x n_stocks x T

            out = self.dilated_conv1(x)
            out = self.relu1(out)

            out = self.dilated_conv2(out)
            out = self.relu2(out)

            out = self.dilated_conv3(out)
            out = self.relu3(out)

            out = self.dilated_conv4(out)
            out = self.relu4(out)

            out = self.conv_final(out)
            out = out[:, :, -1]

            return out
            
            

    #Describe the hyperparameters for the CovNet to run
    learning_rate = 0.001
    batch_size = 16
    display_step = 500
    max_epochs = 1000
    symbols = ['AAPL']
    # symbols = ['GOOGL', 'AAPL', 'AMZN', 'FB', 'ZION', 'NVDA', 'GS']
    n_stocks = len(symbols)
    n_hidden1 = 128
    n_hidden2 = 128
    n_steps_encoder = 20  # time steps, length of time window
    n_output = n_stocks
    T = 30
    start_date = '2013-01-01'
    end_date = '2013-10-31'



    #Load the training data with dataloaders
    dset = SP500('data/sandp500/individual_stocks_5yr', symbols=symbols, start_date='2013-01-01', end_date='2013-07-31', T=T, step=1)

    train_loader = DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    x, y = train_loader.dataset[0]

    print(x.shape,y.shape)
    #>torch tensor of of (29,1), and torch tensor of (1,)<


    #Initialize the model and send it to the GPU
    model = DilatedNet(num_securities=n_stocks, T=T).cuda()

    #Call the function to update the model parameters
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.0)  # n

    #Call the scheduler to control and change the learning rate as we progress
    scheduler_model = lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)

    #Call the loss function
    criterion = nn.MSELoss(size_average=True).cuda()




    ###Run the model and predict the data
    losses = []
    it = 0
    for i in range(max_epochs):    #iterate through the training/predicting process multiple times
        loss_ = 0.                #initialize the loss at 0
        predicted = []
        gt = []
        for batch_idx, (data, target) in enumerate(train_loader):     #for each subsample of the data and its respective target in the train dataloader
        
            data = Variable(data.permute(0, 2, 1)).contiguous()   #make the data in a proper shape for py torch and a variable 
            target = Variable(target.unsqueeze_(0)) #create a false batch dimension for pytorch

            #Send the data and the target to the GPU
            data = data.cuda()       
            target = target.cuda()
            
            #zero the gradient, so variables can be updated according to the new loss we are about to calculate
            optimizer.zero_grad()

            if target.data.size()[1] == batch_size:

                output = model(data)   #run the data through the convnet
                loss = criterion(output, target)   #calculate the loss
                loss_ += loss.item()               #add it to the total loss

                loss.backward()                    #calculate the gradients for each parameter
                optimizer.step()                   #use the gradient information and optimizer to update the parameters to better values
                
                #store the predicted and actual values so we can plot them later
                for k in range(batch_size):
                    predicted.append(output.data[k, 0])
                    gt.append(target.data[:, k, 0])
                    
            it += 1  #increase the iteration step

        print(i, loss_)
        losses.append(loss_)

        scheduler_model.step()    #update the learning rate accoarding to the scheduler

        #plot the actual vs predicted every 20 iterations
        if i % 20 == 0:
            predicted = np.array(predicted).reshape(-1, 1)
            gt = np.array(gt).reshape(-1, 1)
            x = np.array(range(predicted.shape[0]))
            h = plt.figure()
            plt.plot(x, predicted[:, 0], label="predictions")
            plt.plot(x, gt[:, 0], label="true")
            plt.legend()
            plt.show()




    #save the model and plot the loss over training time
    torch.save(model, 'dilated_net_1d.pkl')
    h = plt.figure()
    x = np.arange(len(losses))
    plt.plot(x, np.array(losses), label="loss")
    plt.legend()
    plt.show()


    #use the trained model to predict unseen data (not in training)
    predictions = np.zeros((len(train_loader.dataset.chunks), n_stocks))
    ground_tr = np.zeros((len(train_loader.dataset.chunks), n_stocks))
    batch_size_pred = 4

    dtest = SP500('data/sandp500/individual_stocks_5yr', symbols=symbols, start_date='2013-01-01', end_date='2013-10-31', T=T)
    test_loader = DataLoader(dtest, batch_size=batch_size_pred, shuffle=False, num_workers=4, pin_memory=True)


    predictions = [[] for i in repeat(None, len(symbols))]
    gts = [[] for i in repeat(None, len(symbols))]
    k = 0


    for batch_idx, (data, target) in enumerate(test_loader):
        data = Variable(data.permute(0, 2, 1)).contiguous()
        target = Variable(target.unsqueeze_(1))

        data = data.cuda()
        target = target.cuda()

        if target.data.size()[0] == batch_size_pred:
            output = model(data)
            for k in range(batch_size_pred):
                s = 0
                for stock in symbols:
                    predictions[s].append(output.data[k, s])
                    gts[s].append(target.data[k, 0, s])
                    s += 1
            k += 1

    if len(symbols) == 1:
        pred = dtest.scaler.inverse_transform(np.array(predictions[0]).reshape((len(predictions[0]), 1)))
        gt = dtest.scaler.inverse_transform(np.array(gts[0]).reshape(len(gts[0]), 1))
    if len(symbols) >= 2:
        p = np.array(predictions)
        pred = dtest.scaler.inverse_transform(np.array(predictions).transpose())
        gt = dtest.scaler.inverse_transform(np.array(gts).transpose())

    x = np.array(range(pred.shape[0]))
    x = [np.datetime64(start_date) + np.timedelta64(x, 'D') for x in range(0, pred.shape[0])]
    x = np.array(x)
    months = MonthLocator(range(1, 10), bymonthday=1, interval=1)
    monthsFmt = DateFormatter("%b '%y")

    s = 0

    for stock in symbols:
        fig, ax = plt.subplots()
        plt.plot(x, pred[:, s], label="predictions", color=cm.Blues(300))
        plt.plot(x, gt[:, s], label="true", color=cm.Blues(100))
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthsFmt)
        plt.title(stock)
        plt.xlabel("Time (2013-01-01 to 2013-10-31)")
        plt.ylabel("Stock Price")
        plt.legend()
        fig.autofmt_xdate()
        plt.show()
        s += 1

![Actual targets compared to predicted stock value per month (left)
during training. Total loss value per training iteration
(right).](Pictures/s9500.png "fig:"){width=".8\linewidth"} \[fig:test1\]

![Actual targets compared to predicted stock value per month (left)
during training. Total loss value per training iteration
(right).](Pictures/sploss.png "fig:"){width=".8\linewidth"}
\[fig:test2\]

![Predicted stock value (dark blue) vs. actual stock value (light blue)
per month during testing of un-trained
data.](Pictures/sptest.png "fig:"){width="1\linewidth"} \[fig:sp500\]

As an exercise for the reader, the notebook [Dilated Convolutions with
SP500](https://github.com/justkittenaround/Courseworks/blob/master/Time_Series_Analysis/Dilated%20Convolutions%20with%20SP500)
has two further exercises. One with multistep data, and one with 2D
dilated convolutions.

Fourier Analysis and Synthesis
------------------------------

Another way to investigate and transform our data is using Fourier
analysis and synthesis. Fourier analysis is based on the concept that
that any function or signal is created by different combinations of
basic geometric functions. Thus, any function could be decomposed into
separate parts of a cosine or sine function. This process is called
analysis and involves the signal (data) and the Fourier matrix (many
cosines of different coefficients). Building the function from the
components is called synthesis and can be thought of as the opposite
process of analysis (as we’ll show in the following example). These
techniques break down the data into the most basic geometric functions.
Such applications are important for time series analysis as they allow
us to describe our data in alternative ways and allow us to rebuild the
signal without noise.

### Fourier Series

Fourier series is a special type of Fourier analysis that it only uses
cosine and sine functions to comprise the Fourier matrix. When the
Fourier matrix and the signal are multiplied, the output has
coefficients of the signal. In practice, you may have a signal which you
don’t know the coefficient already (since we’re generating it
ourselves). This principal is based of the orthaganal relationship of
cosine and sine.

    ''' In this example, we'll create a signal (y) and a Fourier matrix (F) and multiply them together to get the coefficients of the signal in terms of cosine functions. Then, we will do this mathematical process in reverse for synthesis by multiplying the transposed Fourier matrix. '''

    import numpy as np
    import matplotlib.pyplot as plt

    #create a signal (y)
    x = np.linspace(0,1000,1000)
    y = np.cos(150*3.14159*x) 
    ys = y.shape[0]

    #create an array to store the Fourier matrix
    F = np.zeros((1000,1000))

    #populate the Fourier matrix
    for i in range(ys): 
        F[i,:] = np.cos(i*3.14159*x/ys) 

    #view the Fourier matrix
    plt.imshow(F) #as a whole
    plt.plot(F[0:10,:].T); #as a subsample of the matrix 

    #multily the Fourier matrix with the signal
    fy = np.matmul(F,y)

    #view the output of the Fourier series analysis and print the signal coefficient
    fy = fy / (1000/2)
    plt.plot(np.abs(fy))
    print(np.argmax(np.abs(fy)))

    #if we multiply the transposed Fourier matrix and the Fourier coefficient we should get back our original signal
    y2 = np.matmul(F.T,fy)
    plt.plot(y2[1:100], 'r')
    plt.plot(y[1:100], '.')

![Original signal (left) and Fourier matrix
(right).](Pictures/f-sig.png "fig:"){width=".8\linewidth"} \[fig:test1\]

![Original signal (left) and Fourier matrix
(right).](Pictures/f-matrix.png "fig:"){width=".8\linewidth"}
\[fig:test2\]

![Fourier matrix (eg. K-space) of component principal functions (left).
Maximum value of Fourier series transformation product. The coefficient
of the signal in terms of cosine components
(right).](Pictures/f-sub.png "fig:"){width=".8\linewidth"} \[fig:test1\]

![Fourier matrix (eg. K-space) of component principal functions (left).
Maximum value of Fourier series transformation product. The coefficient
of the signal in terms of cosine components
(right).](Pictures/singal-co.png "fig:"){width=".8\linewidth"}
\[fig:test2\]

![Smoothed signal (red) using matrix multiplication and before noise,
original signal (dots).](Pictures/f-synth.png){width="1\linewidth"}

### Fast Fourier Transformations with Convolutions

    ''' We can combine the Fourier transformations with convolutions to smooth a signal. We'll create a noisey signal, create the convolutional window, apply the numpy fast Fourier transformation function to the convolutional window and to the noisey signal separately. Then we will use the inverse of the fast Fourier transformation function on the convolution product.  '''

    #create the signal
    x = np.linspace(-2*np.pi,2*np.pi,101)
    y = np.sin(x)
    plt.plot(x,y)

    #create some random noise to add to the signal (to simulate real-world problems)
    n = 0.1*np.random.randn(y.shape[0],)
    yn = y + n 
    plt.plot(yn)

    #create the convolutional window with 1/5 operator values
    w = np.ones(5,)
    w = w / 5

    #use numpy's built-in fast Fourier transformation operator on the window
    np.fft.fft(w)

    #store the shapes to make convolution easier
    s = yn.shape[0]
    ws = w.shape[0]

    #create a convolutional signal the size of our original signal yn
    z = np.zeros(101,)
    z[(s-1)//2 - (ws-1)//2 : (s-1)//2 + (ws-1)//2+1] = w
    plt.plot(z)

    #use numpy to compute the fast Fourier transformation of the convolver
    fz = np.fft.fft(z)
    plt.plot(fz)

    #use numpy to compute the fast Fourier transformation of the noisey signal
    fyn = np.fft.fft(yn)
    plt.plot(fyn)

    #if we apply the inverse of our fast Fourier transformation analysis on the convolution product we should get back a smoothed original signal 
    y2 =  np.fft.ifft(fz*fyn)
    plt.plot(y2, 'r') #smoothed
    plt.plot(yn, 'b') #original

![Maximum value of Fourier series transformation product on the
convolution signal (left). Smoothed signal (red) using numpy’s fast
Fourier transformation and convolutions and before noise, original
signal (blue)
(right).](Pictures/ff-sig.png "fig:"){width=".8\linewidth"}
\[fig:test1\]

![Maximum value of Fourier series transformation product on the
convolution signal (left). Smoothed signal (red) using numpy’s fast
Fourier transformation and convolutions and before noise, original
signal (blue)
(right).](Pictures/ff-synth.png "fig:"){width=".8\linewidth"}
\[fig:test2\]

For a the data file used in the above convolution and Fourier series
examples, see
[Convolution\_Fourier\_Data.zip](https://github.com/justkittenaround/Courseworks/blob/master/Time_Series_Analysis/Convolution_Fourier_Data.zip).
If you’d like to use this in the Collaboratory workspace, you’ll need to
upload this folder to your google drive and change the path in python
example 3.1 line 20 to your google drive path.

For a compiled Google Collaboratory notebook of the above convolution
and Fourier series analysis, see
[TimeSeries1-FFT\_Conv.ipynb](https://github.com/justkittenaround/Courseworks/blob/master/Time_Series_Analysis/Copy%20of%20Time%20Series1-FFT_Conv.ipynb)

Forecasting
===========

Time-series data are periodic functions. Often with such data, we want
to predict or extend the dataset. We’ve already seen two forecasting
techniques with convolutions. Those techniques are a bit more advanced.
In this chapter we’ll go back to the basics for forecasting approaches
for more hands-on experience.

Naive Approaches
----------------

The first and most basic approach, naive forecasting, is simply guessing
what comes next will most likely be what has just occurred. This is a
good technique for some problems and largely depends on the time scale.
As a though experiment, think about measuring water levels in a large
bucket that’s been punctured with a pin-hole. If you measure on a
millisecond time scale, guessing what the water level in the next
millisecond time-step will likely be what you just previously measured.
However, if you did this same experiment by measuring every minutes,
you’d often be wrong. In the example below, our naive forecast doesn’t
follow the original withheld data from the trend too closely, but it
does pas through many of the datapoints.

    ''' In this example, we'll use the previous data from section 3.1.1 to extend the existing training data from the final datapoint through the length of the validation set.  '''

    dd = np.asarray(Train.Count)
    y_hat = valid.copy()
    y_hat['naive']= dd[len(dd)- 1]
    plt.figure(figsize = (12,8))
    plt.plot(Train.index, Train['Count'],  label = 'Train')
    plt.plot(valid.index, valid['Count'],  label = 'Validation')
    plt.plot(y_hat.index, y_hat['naive'],  label = 'Naive')
    plt.legend(loc = 'best')
    plt.title('Naive Forecast')

![Naive forecast of passenger count data, dependent on the last training
datapoint value.](Pictures/naive.png){width="1\linewidth"}

Root Mean Squared Error
-----------------------

Another, and perhaps more educated, forecasting method would be to would
be to take an average of so many of the previous values. Determining how
many previous values to include is can be optimized by assessing how
incorrect our forecast was. By calculating the root of average
difference between our naive prediction and the actual trend, we can use
this value as a metric for testing different window sizes. In the
example below, we show this dataset does best when looking at windows
fewer than 7 datapoints in length.

    ''' In this example, we will calculate the root mean squared error for the true validation set and the naive forecast using a moving average of the last 10 datapoints. We will then iteravely change the window size for taking the average, calculate the mean squared errors, and plot these against the window sizes to see what average of how many previous datapoints produces forecast with the least error. '''

    from sklearn.metrics import mean_squared_error
    from math import sqrt

    y_hat_avg = valid.copy()
    y_hat_avg['moving_average_forecast'] = Train['Count'].rolling(10).mean().iloc[-1]

    plt.figure(figsize = (15,5))
    plt.plot(Train['Count'], label = 'Train')
    plt.plot(valid['Count'], label = 'Validation')
    plt.plot(y_hat_avg['moving_average_forecast'], label = 'Moving Average Forecast with 10 Observations')
    plt.legend(loc = 'best')
    plt.show()


    rmse = sqrt(mean_squared_error(valid['Count'], y_hat_avg['moving_average_forecast']))
    rmse


    maxDays = 365
    summaryArray = np.zeros([maxDays, 2])

    for j in range (1,maxDays):
        rmse = 0
        y_hat_avg = valid.copy()
        y_hat_avg['moving_average_forecast'] = Train['Count'].rolling(j).mean().iloc[-1] #rolling arg is how many days to look back on to forecast
        rmse = sqrt(mean_squared_error(valid['Count'], y_hat_avg['moving_average_forecast']))
        #print(j,rmse)
        summaryArray[j,:] = [j,rmse]


    plt.plot(summaryArray[1:60,0], summaryArray[1:60,1])
    plt.xlabel('Rolling Window Size')
    plt.ylabel('RMSE')

![Moving average of window size 10 prediction (left) and rolling window
size error
(right).](Pictures/moving-avg.png "fig:"){width="1\linewidth"}
\[fig:test1\]

![Moving average of window size 10 prediction (left) and rolling window
size error (right).](Pictures/rmse.png "fig:"){width=".8\linewidth"}
\[fig:test2\]

Conclusions
===========

We’ve now explored the beginning facets of time-series analysis. The
notebooks you’ve created and worked on can be modified and reformed as
you extend your experience in data analysis. Many real-world problems
can be explored with these basic techniques. As we’ve learned, first we
need to get the data and clean it up. Then we can use forecasting
techniques to make predictions about the data. Plotting along the way
helps us visualize what information the time-series is trying to convey.
We saw how comparing multiple feautures within a dataset helped us
better understand the trends happening within. There are many more
advanced ways to perform analysis on periodic data. Popular approaches
such as Long-short-term memory neural networks, Box-Jenkins models, and
other statistical methods are good starting places to further increase
your tools for time-series analysis.
