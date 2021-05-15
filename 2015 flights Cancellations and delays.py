import pandas as pd
import numpy as np 
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
#import matplotlib as mpl
#from matplotlib.gridspec import GridSpec
#from mpl_toolkits.basemap import Basemap
#from collections import 
#from collections import OrderedDict
#from IPython.core.interactiveshell import InteractiveShell


df = pd.read_csv("flights.csv", low_memory=False)
airports = pd.read_csv("airports.csv")
airlines = pd.read_csv("airlines.csv")
wind = pd.read_csv("wind_speed.csv")
humidity = pd.read_csv("humidity.csv")
temp = pd.read_csv("temperature.csv")
pressure = pd.read_csv("pressure.csv")
	
#cleaning, filtering and joining the Weather data
wind = wind.drop(wind.columns.difference(['datetime','New York']), 1, inplace=False)
wind = wind[(wind['datetime'] > '2015-01-01') & (wind['datetime'] < '2016-01-01')]
wind = wind[wind.datetime.notnull()]
wind = wind.rename(columns={'New York':"wind speed"})

humidity = humidity.drop(humidity.columns.difference(['datetime','New York']), 1, inplace=False)
humidity = humidity[(humidity['datetime'] > '2015-01-01') & (wind['datetime'] < '2016-01-01')]
humidity = humidity[humidity.datetime.notnull()]
humidity = humidity.rename(columns={'New York':"humidity"})

temp = temp.drop(temp.columns.difference(['datetime','New York']), 1, inplace=False)
temp = temp[(temp['datetime'] > '2015-01-01') & (wind['datetime'] < '2016-01-01')]
temp = temp[temp.datetime.notnull()]
temp = temp.rename(columns={'New York':"temperature"})
temp['temperature'] = 1.8 * (temp['temperature'] - 273) + 32

pressure = pressure.drop(pressure.columns.difference(['datetime','New York']), 1, inplace=False)
pressure = pressure[(pressure['datetime'] > '2015-01-01') & (wind['datetime'] < '2016-01-01')]
pressure = pressure[pressure.datetime.notnull()]
pressure = pressure.rename(columns={'New York':"pressure"})

temp["wind"] = wind["wind speed"]
temp["humidity"] = humidity["humidity"]
temp["pressure"] = pressure["pressure"]


#df = df[df['MONTH'] == 1]

#crating DATE column
df['DATE'] = pd.to_datetime(df[['YEAR','MONTH', 'DAY']])

#move DATE to the front
col = df.pop("DATE")
df.insert(0, col.name, col)

#remove the null in DEPARTURE_DELAY and ARRIVAL_DELAY
df = df[df.DEPARTURE_DELAY.notnull()]
df = df[df.ARRIVAL_DELAY.notnull()]

#filttering for NYC
df = df[(df.ORIGIN_AIRPORT.str.contains("JFK")) | (df.ORIGIN_AIRPORT.str.contains("LGA")) |
          ( df.ORIGIN_AIRPORT.str.contains("EWR"))]

#droping the unwanted variables 
df = df.drop(['YEAR','MONTH', 'DAY', "DAY_OF_WEEK", "FLIGHT_NUMBER", "TAXI_OUT",
              "WHEELS_OFF", "ELAPSED_TIME", "WHEELS_ON", "TAXI_IN"],  axis="columns", inplace=False)

#join with airlines
df = df.merge(airlines, left_on='AIRLINE', right_on='IATA_CODE', how='inner')

#droping the unwanted vriables  fron the AIRLINE data
df = df.drop(['AIRLINE_x','IATA_CODE'], axis=1)
df = df.rename(columns={"AIRLINE_y":"AIRLINE"})

#move AIRLINE to the third column
col = df.pop("AIRLINE")
df.insert(2, col.name, col)  



# Function that convert the 'HHMM' string to datetime.time
def format_heure(chaine):
    if pd.isnull(chaine):
        return np.nan
    else:
        if chaine == 2400: chaine = 0
        chaine = "{0:04d}".format(int(chaine))
        heure = datetime.time(int(chaine[0:2]), int(chaine[2:4]))
        return heure
#_____________________________________________________________________
# Function that combines a date and time to produce a datetime.datetime
def combine_date_heure(x):
    if pd.isnull(x[0]) or pd.isnull(x[1]):
        return np.nan
    else:
        return datetime.datetime.combine(x[0],x[1])
#_______________________________________________________________________________
# Function that combine two columns of the dataframe to create a datetime format
def create_flight_time(df, col):    
    liste = []
    for index, cols in df[['DATE', col]].iterrows():    
        if pd.isnull(cols[1]):
            liste.append(np.nan)
        else:
            cols[1] = format_heure(cols[1])
            liste.append(combine_date_heure(cols))
    return pd.Series(liste) 


df['SCHEDULED_DEPARTURE'] = create_flight_time(df, 'SCHEDULED_DEPARTURE')
df['DEPARTURE_TIME'] = df['DEPARTURE_TIME'].apply(format_heure)

#rounding the minutes in SCHEDULED_DEPARTURE so we could join with waether
df['SCHEDULED_DEPARTURE'] = df['SCHEDULED_DEPARTURE'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour,60*(dt.minute // 60)))

#remove Date (we don't need it anymore, we got SCHEDULED_DEPARTURE in datetime format)
df = df.drop('DATE',  axis="columns", inplace=False)

#move SCHEDULED_DEPARTURE to the front
col = df.pop("SCHEDULED_DEPARTURE")
df.insert(0, col.name, col)

#join with temp(weather with flights)
#frames = [temp, df]
#df = pd.concat(frames)
#df['SCHEDULED_DEPARTURE']=df['SCHEDULED_DEPARTURE'].astype('datetime64[ns]')
#converting the datetime column in temp to datetime format so we could join (it was in object format)
temp['datetime']=temp['datetime'].astype('datetime64[ns]')
#df.info()
#temp.info()
df = df.merge(temp, left_on='SCHEDULED_DEPARTURE', right_on='datetime', how='inner')

#cheking for null in all columns
null = df.isnull().sum()

#transform the NAN-data to the value "0.0" because there was no impact on the flight by these data that causes a delay 
df['AIRLINE_DELAY'] = df['AIRLINE_DELAY'].fillna(0)
df['AIR_SYSTEM_DELAY'] = df['AIR_SYSTEM_DELAY'].fillna(0)
df['SECURITY_DELAY'] = df['SECURITY_DELAY'].fillna(0)
df['LATE_AIRCRAFT_DELAY'] = df['LATE_AIRCRAFT_DELAY'].fillna(0)
df['WEATHER_DELAY'] = df['WEATHER_DELAY'].fillna(0)


#counting the cancel flights by the cancel reason
cancel_count = df['CANCELLATION_REASON'].value_counts()

#transform the rest of NAN CANCELLATION_REASON to "0.0"
df['CANCELLATION_REASON'] = df['CANCELLATION_REASON'].fillna(0)

#cheking again for null in all columns
null = df.isnull().sum()

df = df.drop('datetime',  axis="columns", inplace=False)

#=================================================================================
#Project.Descriptive.Analysis

df1 = pd.read_csv("flights.csv")


df1 = df1.drop(["DAY_OF_WEEK", "FLIGHT_NUMBER", "TAXI_OUT",
              "WHEELS_OFF", "ELAPSED_TIME", "WHEELS_ON", "TAXI_IN"],  axis="columns", inplace=False)


df1 = df1[df1.ARRIVAL_DELAY > 14 ]

print("\nAirlines average delay = best performance")
average_delay = df1.groupby("AIRLINE")["ARRIVAL_DELAY"].mean()
print(average_delay)


print ("\n Airlines with the highest number of delays... ")

df1["Total Arrival Delays"] = df1.groupby("AIRLINE")["ARRIVAL_DELAY"].transform("count")
Totaldelays = df1.groupby ("AIRLINE").count().reset_index()[["AIRLINE","Total Arrival Delays"]]
sort_Totaldelays = Totaldelays.sort_values(by=["Total Arrival Delays"], ascending = False)
print(sort_Totaldelays.head(1))
print(sort_Totaldelays.tail(1))
print (" WN = Southwest Airlines Co. recorded the highest number of delays")
print (" HA = Hawaiian Airlines Inc. recorded the lowest number of delays")


print ("\n Origin to destination airports with the highest number of delays... ")
df1["Trajectory"] = df1["ORIGIN_AIRPORT"] + df1["DESTINATION_AIRPORT"]
df1 = df1.drop(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"], axis=1)
 
df1["Trajectory Delays"] = df1.groupby("Trajectory")["ARRIVAL_DELAY"].transform("count")
Totaldestination = df1.groupby ("Trajectory").count().reset_index()[["Trajectory","Trajectory Delays"]]
sort_Totaldestination = Totaldestination.sort_values(by=["Trajectory Delays"], ascending = False)
print(sort_Totaldestination.head(1))
print (" flights From San Francisco International Airport to Los Angeles International Airport recorded the highest number of delays")


print ("\nAirlines with A Cancellations = Air Carrier.. ")
bo = pd.read_csv("flights.csv")

bo1 = bo[(bo["CANCELLATION_REASON"] == "A")]
bo2 = bo[(bo["CANCELLATION_REASON"] == "B")]
bo3 = bo[(bo["CANCELLATION_REASON"] == "C")]
bo4 = bo[(bo["CANCELLATION_REASON"] == "D")]


#BLUE = Air Carrier
gf = bo1.groupby("MONTH").count()["CANCELLATION_REASON"]
gf.plot()

#YELLOW = Extreme Weather     
gf = bo2.groupby("MONTH").count()["CANCELLATION_REASON"]
gf.plot()

#GREEN = National Aviation System
gf = bo3.groupby("MONTH").count()["CANCELLATION_REASON"]
gf.plot()

#RED = Security
gf = bo4.groupby("MONTH").count()["CANCELLATION_REASON"]
gf.plot()




bo["Total Cancellations"] = bo.groupby("AIRLINE")["CANCELLATION_REASON"].transform("count")
aa = bo[bo.CANCELLATION_REASON == "A"]
aa1 = aa.groupby ("AIRLINE").count().reset_index()[["AIRLINE", "Total Cancellations"]]
sort_cancellationsA = aa1.sort_values(by=["Total Cancellations"], ascending = False)
print(sort_cancellationsA.head(3))
print (" WN = Southwest Airlines Co. recorded the highest number of Cancellations")


print ("\nAirlines with B Cancellations = Extreme Weather.. ")
bo["Total Cancellations"] = bo.groupby("AIRLINE")["CANCELLATION_REASON"].transform("count")
bb = bo[bo.CANCELLATION_REASON == "B"]
bb1 = bb.groupby ("AIRLINE").count().reset_index()[["AIRLINE", "Total Cancellations"]]
sort_cancellationsB = bb1.sort_values(by=["Total Cancellations"], ascending = False)
print(sort_cancellationsB.head(3))
print (" MQ = American Eagle Airlines Inc. recorded the highest number of Cancellations")


print ("\nAirlines with C Cancellations = National Aviation System,.. ")
bo["Total Cancellations"] = bo.groupby("AIRLINE")["CANCELLATION_REASON"].transform("count")
cc = bo[bo.CANCELLATION_REASON == "C"]
cc1 = cc.groupby ("AIRLINE").count().reset_index()[["AIRLINE", "Total Cancellations"]]
sort_cancellationsC = cc1.sort_values(by=["Total Cancellations"], ascending = False)
print(sort_cancellationsC.head(3))
print (" EV = Atlantic Southeast Airlines recorded the highest number of Cancellations")

print ("\nAirlines with D Cancellations = Security.. ")
bo["Total Cancellations"] = bo.groupby("AIRLINE")["CANCELLATION_REASON"].transform("count")
dd = bo[bo.CANCELLATION_REASON == "D"]
dd1 = dd.groupby ("AIRLINE").count().reset_index()[["AIRLINE", "Total Cancellations"]]
sort_cancellationsD = dd1.sort_values(by=["Total Cancellations"], ascending = False)
print(sort_cancellationsD.head(3))
print (" WN = Southwest Airlines Co. recorded the highest number of Cancellations")


########################### ESMERALDA ################################
#convert categorical data to numerical data AIRLINE and DESTINATION_AIRPORT
print("MODEL")
data1 = df[["DEPARTURE_DELAY", "pressure",
            "humidity", "wind", "temperature"]]

print(data1.dtypes)

correlation = data1.corr()
print(correlation)

#independent variables
X = data1[["pressure",
            "humidity", "wind", "temperature"]]

#we have AIRLINE and DESTINATION_AIRPORT as a categorical variable, 
#so we need to add dummy variables instead.
#X = pd.get_dummies(data=X, drop_first=True)
#print(X.head())

#dependent variable
Y = data1['DEPARTURE_DELAY']

#Creating a train and test dataset.
#split the data into,  training 70 % and testing 30% 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=99)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train,y_train)

print(model.intercept_)
coeff_parameter = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
print(coeff_parameter)

#The sign of each coefficient indicates the direction of the relationship between a predictor variable 
#and the response variable.

predictions = model.predict(X_test)
print(predictions)

import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.regplot(y_test,predictions)

#Below we are checking Rsquare value
import statsmodels.api as sm
X_train_Sm= sm.add_constant(X_train)
X_train_Sm= sm.add_constant(X_train)
ls=sm.OLS(y_train,X_train_Sm).fit()
print(ls.summary())

#=====================================

fig_dim = (14,18)
f, ax = plt.subplots(figsize=fig_dim)
quality=df1["AIRLINE"].unique()
size=df1["AIRLINE"].value_counts()

plt.pie(size,labels=quality,autopct='%1.0f%%')
plt.show()

"""
def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}
#_______________________________________________________________
# Creation of a dataframe with statitical infos on each airline:
global_stats = df['DEPARTURE_DELAY'].groupby(df['AIRLINE']).apply(get_stats).unstack()
global_stats = global_stats.sort_values('mean')

#=====================================

identify_airport = airports.set_index('IATA_CODE')['CITY'].to_dict()
latitude_airport = airports.set_index('IATA_CODE')['LATITUDE'].to_dict()
longitude_airport = airports.set_index('IATA_CODE')['LONGITUDE'].to_dict()
full_name = airlines.set_index('IATA_CODE')['AIRLINE'].to_dict()

def make_map(df, carrier, long_min, long_max, lat_min, lat_max):
    fig=plt.figure(figsize=(7,3))
    ax=fig.add_axes([0.,0.,1.,1.])
    m = Basemap(resolution='i',llcrnrlon=long_min, urcrnrlon=long_max,
                  llcrnrlat=lat_min, urcrnrlat=lat_max, lat_0=0, lon_0=0,)
    df2 = df[df['AIRLINE'] == carrier]
    count_trajectories = df2.groupby(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']).size()
    count_trajectories.sort_values(inplace = True)
    
    for (origin, dest), s in count_trajectories.iteritems():
        nylat,   nylon = latitude_airport[origin], longitude_airport[origin]
        m.plot(nylon, nylat, marker='o', markersize = 10, markeredgewidth = 1,
                   color = 'seagreen', markeredgecolor='k')

    for (origin, dest), s in count_trajectories.iteritems():
        nylat,   nylon = latitude_airport[origin], longitude_airport[origin]
        lonlat, lonlon = latitude_airport[dest], longitude_airport[dest]
        if pd.isnull(nylat) or pd.isnull(nylon) or \
                pd.isnull(lonlat) or pd.isnull(lonlon): continue
        if s < 100:
            m.drawgreatcircle(nylon, nylat, lonlon, lonlat, linewidth=0.5, color='b',
                             label = '< 100')
        elif s < 200:
            m.drawgreatcircle(nylon, nylat, lonlon, lonlat, linewidth=2, color='r',
                             label = '100 <.< 200')
        else:
            m.drawgreatcircle(nylon, nylat, lonlon, lonlat, linewidth=2, color='gold',
                              label = '> 200')    
    #_____________________________________________
    # remove duplicate labels and set their order		
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    key_order = ('< 100', '100 <.< 200', '> 200')                
    new_label = OrderedDict()
    for key in key_order:
        if key not in by_label.keys(): continue
        new_label[key] = by_label[key]
    plt.legend(new_label.values(), new_label.keys(), loc = 'best', prop= {'size':8},
               title='flights per month', facecolor = 'palegreen', 
               shadow = True, frameon = True, framealpha = 1)    
    m.drawcoastlines()
    m.fillcontinents()
    ax.set_title('{} flights'.format(full_name[carrier]))

coord = dict()
coord['DL'] = [-165, -60, 10, 55]
coord['WN'] = [-182, -63, 10, 75]
coord['EV'] = [-180, -65, 10, 52]
for carrier in ['DL', 'WN', 'EV']: 
    make_map(df, carrier, *coord[carrier])
    
#===================================== 
"""
#=====================================
"""
#=====================================
def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}
# Creation of a dataframe with statitical infos on each airline:
global_stats = df['DEPARTURE_DELAY'].groupby(df['AIRLINE']).apply(get_stats).unstack()
global_stats = global_stats.sort_values('mean')

abbr_companies = airlines.set_index('IATA_CODE')['AIRLINE'].to_dict()

# Group by airline and sum up / count the values
df_flights_grouped_sum = df.groupby('AIRLINE', as_index= False)['ARRIVAL_DELAY'].agg('mean').rename(columns={"ARRIVAL_DELAY":"ARRIVAL_DELAY_SUM"})
df_flights_grouped_cnt = df.groupby('AIRLINE', as_index= False)['ARRIVAL_DELAY'].agg('count').rename(columns={"ARRIVAL_DELAY":"ARRIVAL_DELAY_CNT"})

# Merge the two groups together
df_flights_grouped_delay = df_flights_grouped_sum.merge(df_flights_grouped_cnt, left_on='AIRLINE', right_on='AIRLINE', how='inner')
# Calculate the average delay per airline
df_flights_grouped_delay.loc[:,'AVG_DELAY_AIRLINE'] = df_flights_grouped_delay['ARRIVAL_DELAY_SUM'] / df_flights_grouped_delay['ARRIVAL_DELAY_CNT']

df_flights_grouped_delay.sort_values('ARRIVAL_DELAY_SUM', ascending=False)


font = {'family' : 'normal', 'weight' : 'bold', 'size'   : 15}
mpl.rc('font', **font)
import matplotlib.patches as mpatches
#__________________________________________________________________
# I extract a subset of columns and redefine the airlines labeling 
df2 = df.loc[:, ['AIRLINE', 'DEPARTURE_DELAY']]
df2['AIRLINE'] = df2['AIRLINE'].replace(abbr_companies)
#________________________________________________________________________
colors = ['royalblue', 'grey', 'wheat', 'c', 'firebrick', 'seagreen', 'lightskyblue',
          'lightcoral', 'yellowgreen', 'gold', 'tomato', 'violet', 'aquamarine', 'chartreuse']
#___________________________________
fig = plt.figure(1, figsize=(16,15))
gs=GridSpec(2,2)             
ax1=fig.add_subplot(gs[0,0]) 
ax2=fig.add_subplot(gs[0,1]) 
ax3=fig.add_subplot(gs[1,:]) 
#------------------------------
# Pie chart nº1: nb of flights
#------------------------------
labels = [s for s in  global_stats.index]
sizes  = global_stats['count'].values
explode = [0.3 if sizes[i] < 20000 else 0.0 for i in range(len(abbr_companies))]
patches, texts, autotexts = ax1.pie(sizes, explode = explode,
                                labels=labels, colors = colors,  autopct='%1.0f%%',
                                shadow=False, startangle=0)
for i in range(len(abbr_companies)): 
    texts[i].set_fontsize(14)
ax1.axis('equal')
ax1.set_title('% of flights per company', bbox={'facecolor':'midnightblue', 'pad':5},
              color = 'w',fontsize=18)
#_______________________________________________
# I set the legend: abreviation -> airline name
comp_handler = []
for i in range(len(abbr_companies)):
    comp_handler.append(mpatches.Patch(color=colors[i],
            label = global_stats.index[i] + ': ' + abbr_companies[global_stats.index[i]]))
ax1.legend(handles=comp_handler, bbox_to_anchor=(0.2, 0.9), 
           fontsize = 13, bbox_transform=plt.gcf().transFigure)
#----------------------------------------
# Pie chart nº2: mean delay at departure
#----------------------------------------
sizes  = global_stats['mean'].values
sizes  = [max(s,0) for s in sizes]
explode = [0.0 if sizes[i] < 20000 else 0.01 for i in range(len(abbr_companies))]
patches, texts, autotexts = ax2.pie(sizes, explode = explode, labels = labels,
                                colors = colors, shadow=False, startangle=0,
                                autopct = lambda p :  '{:.0f}'.format(p * sum(sizes) / 100))
for i in range(len(abbr_companies)): 
    texts[i].set_fontsize(14)
ax2.axis('equal')
ax2.set_title('Mean delay at origin', bbox={'facecolor':'midnightblue', 'pad':5},
              color='w', fontsize=18)
#------------------------------------------------------
# striplot with all the values reported for the delays
#___________________________________________________________________
# I redefine the colors for correspondance with the pie charts
colors = ['firebrick', 'gold', 'lightcoral', 'aquamarine', 'c', 'yellowgreen', 'grey',
          'seagreen', 'tomato', 'violet', 'wheat', 'chartreuse', 'lightskyblue', 'royalblue']
#___________________________________________________________________
ax3 = sns.stripplot(y="AIRLINE", x="DEPARTURE_DELAY", size = 4, palette = colors,
                    data=df2, linewidth = 0.5,  jitter=True)
plt.setp(ax3.get_xticklabels(), fontsize=14)
plt.setp(ax3.get_yticklabels(), fontsize=14)
ax3.set_xticklabels(['{:2.0f}h{:2.0f}m'.format(*[int(y) for y in divmod(x,60)])
                         for x in ax3.get_xticks()])
plt.xlabel('Departure delay', fontsize=18, bbox={'facecolor':'midnightblue', 'pad':5},
           color='w', labelpad=20)
ax3.yaxis.label.set_visible(False)
#________________________
plt.tight_layout(w_pad=3)
"""


