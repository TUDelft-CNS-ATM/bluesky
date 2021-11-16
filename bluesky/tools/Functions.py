import psycopg2
import sqlite3
import pandas as pd
from math import floor, ceil

def Connect_SQL_DB():
    conn = psycopg2.connect(
        host="localhost",
        database="bluesky_2309",
        user="postgres",
        password="lucht_1")
    return conn

def query_DB_to_DF(query):
    conn = Connect_SQL_DB()
    sql_query = pd.read_sql_query(query, conn)
    df = pd.DataFrame(sql_query, columns=['timestamp_data', 'timestamp_prediction', 'lon', 'lat', 'alt', 'uwind', 'vwind'])
    return df

def find_datapoint_timeframe(df, point):
    """[time,alt,lat,lon]"""
    lon_min = floor(point[3] * 10) / 10
    lon_plus = ceil(point[3] * 10) / 10
    lat_min = floor(point[2] * 10) / 10
    lat_plus = ceil(point[2] * 10) / 10

    value_mmm, value_pmm = find_height_levels(df, point, lat_min, lon_min)
    value_mmp, value_pmp = find_height_levels(df, point, lat_min, lon_plus)
    value_mpm, value_ppm = find_height_levels(df, point, lat_plus, lon_min)
    value_mpp, value_ppp = find_height_levels(df, point, lat_plus, lon_plus)

    timeless_value  = interpolation(value_mmm, value_mmp, value_mpm, value_mpp, value_pmm, value_pmp, value_ppm, value_ppp, point[1], point[2], point[3])
    value = [point[0], timeless_value[0], timeless_value[1], timeless_value[2], timeless_value[3], timeless_value[4]]
    return value

def find_height_levels(df, point, lat, lon):
    df_t = df.query('timestamp_data == ' + str(point[0]) + 'and lat == ' + str(lat) + ' and lon ==' + str(lon))
    layer_min = layer_plus = -1
    for i in range(len(df_t)):
        if point[1] > df_t.iloc[i][4]:
            if i - 1 >= 0:
                layer_min = i
                layer_plus = i - 1
                break
    if layer_min != -1 and layer_plus != -1:
        value_m = [df_t.iloc[layer_min][4], lat, lon, df_t.iloc[layer_min][5], df_t.iloc[layer_min][6]]
        value_p = [df_t.iloc[layer_plus][4], lat, lon, df_t.iloc[layer_plus][5], df_t.iloc[layer_plus][6]]
    return value_m, value_p

def interpolation(value_mmm, value_mmp, value_mpm, value_mpp, value_pmm, value_pmp, value_ppm, value_ppp, alt, lat, lon):
    """[alt,lat,lon,uwind,vwind]"""
    value_mm = [value_mmm[0], value_mmm[1], lon, interpolate(lon, value_mmm[2], value_mmp[2], value_mmm[3], value_mmp[3]), interpolate(lon, value_mmm[2], value_mmp[2], value_mmm[4], value_mmp[4])]
    value_mp = [value_mpm[0], value_mpm[1], lon, interpolate(lon, value_mpm[2], value_mpp[2], value_mpm[3], value_mpp[3]), interpolate(lon, value_mpm[2], value_mpp[2], value_mpm[4], value_mpp[4])]
    value_pm = [value_pmm[0], value_pmm[1], lon, interpolate(lon, value_pmm[2], value_pmp[2], value_pmm[3], value_pmp[3]), interpolate(lon, value_pmm[2], value_pmp[2], value_pmm[4], value_pmp[4])]
    value_pp = [value_ppm[0], value_ppm[1], lon, interpolate(lon, value_ppm[2], value_ppp[2], value_ppm[3], value_ppp[3]), interpolate(lon, value_ppm[2], value_ppp[2], value_ppm[4], value_ppp[4])]

    value_m  = [value_mm[0], lat, lon, interpolate(lat, value_mm[1], value_mp[1], value_mm[3], value_mp[3]), interpolate(lat, value_mm[1], value_mp[1], value_mm[4], value_mp[4])]
    value_p  = [value_pm[0], lat, lon, interpolate(lat, value_pm[1], value_pp[1], value_pm[3], value_pp[3]), interpolate(lat, value_pm[1], value_pp[1], value_pm[4], value_pp[4])]

    value    = [alt, lat, lon, round(interpolate(alt, value_m[0], value_p[0], value_m[3], value_p[3]),4), round(interpolate(alt, value_m[0], value_p[0], value_m[4], value_p[4]),4)]
    return value

def interpolate(value , value_a , value_b, answer_a , answer_b):
    if value_a != value_b:
        answer = (value - value_a)/(value_b - value_a) * (answer_b-answer_a) + answer_a
        return answer
    return answer_a

#def winddata_location(point1, point2):  <-----------------------------------------------------

query = '''SELECT * FROM bluesky_2309_top WHERE timestamp_data = 210923003'''
df = query_DB_to_DF(query)
print(find_datapoint_timeframe(df, [210923003, 30000, 50.03, 2.63]))





#print(interpolation([10000,50,0,20,20],[10000,50,10,40,40],[10000,70,0,30,60],[10000,70,10,80,80],[5000,50,0,20,20],[5000,50,10,40,40],[5000,70,0,0,10],[5000,70,10,10,80],7500,60,5))

# query = '''SELECT * FROM bluesky_2309_top WHERE timestamp_data = 210923003 AND lon = -1 AND lat = 50 AND alt = 41628.25'''


# query = '''SELECT * FROM bluesky_2309_top WHERE timestamp_data = 210923003'''
# df = query_DB_to_DF(query)
# print("done")
#
# point = [1.05, 50.05, 10000]  # lon lat alt
# lon_min = floor(point[0] * 10) / 10
# lon_max = ceil(point[0] * 10) / 10
# lat_min = floor(point[1] * 10) / 10
# lat_max = ceil(point[1] * 10) / 10
#
# dfq = df.query('timestamp_data == 210923003 and lon ==' + str(lon_min) + 'and lat == ' + str(lat_min))
# print(dfq)
# print("---------------------------")
# print(dfq.iloc[0][0])

# dfq = df.query('timestamp_data == 210923003 and lon == -1 and lat == 50 and alt == 38850.42')
# print(dfq)
# dfq = df.query('timestamp_data == 210923003 and lon == -1 and lat == 50 and alt == 36201.84')
# print(dfq)
# dfq = df.query('timestamp_data == 210923003 and lon == -1 and lat == 50 and alt == 33659.22')
# print(dfq)
# dfq = df.query('timestamp_data == 210923003 and lon == -1 and lat == 50 and alt == 31221.21')
# print(dfq)
# dfq = df.query('timestamp_data == 210923003 and lon == -1 and lat == 50 and alt == 28887.25')
# print(dfq)