import psycopg2
import sqlite3
import pandas as pd

def Connect_SQL_DB():
    conn = psycopg2.connect(
        host="localhost",
        database="bluesky_3",
        user="postgres",
        password="lucht_1")

    return conn

def query_DB_to_DF(query):
    conn = Connect_SQL_DB()
    sql_query = pd.read_sql_query(query , conn)
    df = pd.DataFrame(sql_query, columns = ['timestamp_data', 'timestamp_prediction', 'lon', 'lat', 'alt', 'uwind' , 'vwind'])

    return df

query = '''SELECT * FROM '''
print(query_DB_to_DF(query))