# @author: Cai Jie
# @Date:   2019/2/28 15:56

import pandas as pd
import pyorient
import pymysql
import numpy as np
import collections
import math

ConnParam = {
    'host': '10.0.60.201',
    'user': 'root',
    'password': 'thinklanddsp',
    'database': 'sms_data',
    'charset': 'utf8',
    'port': 6508
}


def main():
    conn = pymysql.connect(**ConnParam)
    client = pyorient.OrientDB('localhost', 2424)
    session_id = client.connect('10.0.60.201', 'thinkland')
    dbname = 'demo5'
    usr = "admin"
    pwd = "admin"

    if client.db_exists(dbname):
        client.db_open(dbname, usr, pwd)
    else:
        client.db_create(dbname, pyorient.DB_TYPE_GRAPH, pyorient.STORAGE_TYPE_PLOCAL)

    print("abc")


if __name__ == '__main__':
    main()