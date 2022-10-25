import streamlit as st
from streamlit_folium import folium_static
import pandas as pd
import pydeck as pdk
import folium   
from folium import plugins
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Loading data set
def set_data():
    df = pd.read_csv("taxi.txt", sep =',',names=['taxi_id', 'date_time', 'longitude', 'latitude'],nrows=10000)#,nrows=10000
    df['taxi_id'] = df['taxi_id'].apply(pd.to_numeric, errors='coerce')
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.dropna() #Dropping missings values
    df=df.drop_duplicates() #Dropping duplicates values
    return df
    
df = set_data()

#Processing data
def processing_data(df):
    #We take the next point of the taxi so we will get the distance between the two points
    #Using shift fuction to take the next point from the next row in the dataframe
    df['end_time'] = df['date_time'].shift(-1)
    df['end_lat'] = df['latitude'].shift(-1)
    df['end_lon'] = df['longitude'].shift(-1)
    
    # Drop the last row for each taxi id because you can't move from the last point
    df['taxi_id_prev'] = df['taxi_id'].shift(-1)
    df['id_check'] = (df['taxi_id'] == df['taxi_id_prev'])
    df = df[df['id_check'] == True]

    # Cleaning up the data
    del df['taxi_id_prev']
    del df['id_check']
    df = df.reset_index(drop=True)
    #adding the new columns
    df.columns = ['taxi_id', 'start_time', 'start_lon', 'start_lat','end_time','end_lat','end_lon']
    # Calculate time between two points in seconds
    df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds()
    return df

df = processing_data(df)


def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # We use haversine formula to calculate the distance between two points
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

#Add/update a column to the data frame with the distances (in metres)
df['dist_kilometer'] = df.apply(lambda x : haversine(x['start_lon'], x['start_lat'], x['end_lon'], x['end_lat']), axis=1)
df['speed'] = df['dist_kilometer'] / df['duration'] * 3600

def get_time(df):
    df['start_hour']=df['start_time'].dt.hour
    df['day']=df['start_time'].dt.weekday
    return df
df = get_time(df)

@st.cache()
def period_of_day(t):
    if t in range(6,12):
        return 'Morning'
    elif t in range(12,14):
        return 'Noon'
    elif t in range(14,21):
        return 'Afternoon'
    else:
        return 'Night'
df['period_of_day'] = df['start_hour'].apply(period_of_day)



# LINEAR REGRESSION 
feature = ["start_hour",'day', 'start_lon', 'start_lat','end_lat', 'end_lon']
target = ['duration']
feature_variables = df[feature] 
target_variables = df[target]
# Set variables for the targets and features
Y = target_variables
Y.duration = Y.duration.apply(lambda x: int(x))
X = feature_variables
# Split the data into training and validation sets
#X_train, X_val, Y_train, Y_val
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state=7)
reg = LinearRegression().fit(X, Y)
reg.score(X, Y)
model = LinearRegression()
fit_model = model.fit(X_train,Y_train)
y_pred= fit_model.predict(X_test)

def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
     
    return "%d:%02d:%02d" % (hour, minutes, seconds)

## SETTING DEFAULT PARAMETERS ON THE SIDEBAR
start_hour = st.sidebar.text_input("Start hours","18")
day = st.sidebar.text_input("Day","0")
start_lat = st.sidebar.text_input("Start Latitude","40.00178")
start_lon = st.sidebar.text_input("Start longitude","116.40153")
end_lat = st.sidebar.text_input("End latitude","39.99985")
end_lon = st.sidebar.text_input("End longitude","116.40153")



def duration(start_hour,day,start_lon,start_lat,end_lat,end_lon):
    df_test = pd.DataFrame({'start_hour': [start_hour],'day' : [day], 'start_lon' : [start_lon], 'start_lat': [start_lat],'end_lat': [end_lat], 'end_lon' : [end_lon]})
    model_test = LinearRegression()
    fit_model_test = model_test.fit(X_train,Y_train)
    y_pred_test= fit_model_test.predict(df_test)
    return y_pred_test

y_pred_test1 = duration(start_hour,day,start_lon,start_lat,end_lat,end_lon)

st.sidebar.write("Driving time will be approximately : ",convert(y_pred_test1[0][0]))


if st.checkbox("Show values",False):
    st.write(df.astype('object'))

st.title("Initial Data")
df_scatter = pd.read_csv("./very_first_dataframe.csv")
df_scatter = df_scatter.loc[0:800000]
st.pydeck_chart(pdk.Deck(
        initial_view_state = pdk.ViewState(
        latitude=39.88602894100377, 
        longitude=116.36502017149408, 
        zoom=9.853944941202695, 
        bearing=0, 
        pitch=0
    ),

    layers = [pdk.Layer(
    'ScatterplotLayer',     # Change the `type` positional argument here
    df_scatter,
    get_position=['longitude', 'latitude'],
    auto_highlight=True,
    get_radius=20,          # Radius is given in meters
    get_fill_color=[180, 0, 200, 140],  # Set an RGBA value for fill
    pickable=True)

    ],
))

df['marker_color'] = pd.cut(df['speed'], bins=5,labels=['blue','green', 'yellow','orange', 'red'])
def folium_map(df):
    map = folium.Map(location=[df['start_lat'].iloc[0], df['start_lon'].iloc[0]], tiles='cartodbpositron', zoom_start=11)
    for index, row in df.iterrows():
        folium.Marker([row['start_lat'], row['start_lon']], 
        popup="Speed : "+str(row['speed']),
        icon=plugins.BeautifyIcon(number=index, border_color=row['marker_color'], text_color='black', inner_icon_style='margin-top:0px;')
        ).add_to(map)
    return map

if st.checkbox("Taxi Interactive Map",False):    
    df_map = df.sample(frac=0.1, random_state=1)
    map = folium_map(df_map)
    map.add_child(folium.LatLngPopup())
    folium_static(map)

#cartodbpositron
def folium_map_low():
    map = folium.Map(location= [39.9350, 116.3892], tiles='cartodbpositron', zoom_start=11)
    return map
map_low = folium_map_low()
map_low.add_child(folium.LatLngPopup())
folium_static(map_low)


def density(weekday) :
    df_density = df[df['day']== weekday] #getting the number of taxis for each weekday
    nb_taxis = df_density['day'].value_counts()[weekday]
    tot_taxis = nb_taxis*6.6  #getting the total nummber of taxis in 2008 compared to our dataset
    "This is the total number of taxis on the day",tot_taxis
    nb_cars = tot_taxis*43 #getting the total number of cars in 2008 ratio of approx 1 taxi for 43 cars
    "This is the total number of cars for the day",nb_cars
    density = nb_cars/16411 #dividing by the surface of Beijing in square kilometers to get density
    "Average density of cars per square kilometer : ",density


st.title("Density Heatmap")

weekday = st.slider('What day would you like to observe ?', 0, 6, 0)
density(weekday)

@st.cache()
def get_weekday_df(wd):
    return df_scatter[df_scatter['weekday'] == weekday]

by_weekday_df = get_weekday_df(weekday)
st.pydeck_chart(pdk.Deck(
        initial_view_state = pdk.ViewState(
        latitude=39.88602894100377, 
        longitude=116.36502017149408, 
        zoom=9.853944941202695, 
        bearing=0, 
        pitch=0
    ),

    layers = [pdk.Layer(
        "HeatmapLayer",
        by_weekday_df,
        get_position='[longitude, latitude]'
    )

    ],
))

st.title("Beijing road network")
df_edge = pd.read_csv("./Edge.csv")
st.pydeck_chart(pdk.Deck(
        initial_view_state = pdk.ViewState(
        latitude=39.88602894100377, 
        longitude=116.36502017149408, 
        zoom=9.853944941202695, 
        bearing=0, 
        pitch=0
    ),

    layers = [pdk.Layer(
        "LineLayer",
        df_edge,
        get_source_position='[s_lng, s_lat]',
        get_target_position='[e_lng, e_lat]',
        get_color=[255, 255, 0],
        auto_highlight=True,
        pickable=True,
    )

    ],
))

