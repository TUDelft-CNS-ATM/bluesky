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