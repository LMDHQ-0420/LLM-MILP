import json
import pickle
import pymysql
from pymysql.cursors import DictCursor
import ast

def load_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
        data = json.loads(data)
    return data

def save_json_file(path,data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
def load_pickle_file(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pickle_file(file_name, data):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

def save_file(path,data):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)

def load_file(path):
    with open("test.txt", "r", encoding="utf-8") as f:
        data = f.read()
    return data


def is_number(value):
    try:
        _ = float(value)
        return True
    except:
        return False

def get_key_from_value(value, dictionary):
    return next((key for key, val in dictionary.items() if val == value), None)


def connect_mysql(mysql_config):
    return pymysql.connect(
            host=mysql_config['host'],
            user=mysql_config['user'],
            password=mysql_config['password'],
            port=int(mysql_config['port']),
            database=mysql_config['database'],
            charset="utf8mb4",
            cursorclass= pymysql.cursors.DictCursor
        )

def execute_sql(sql, mysql_config):
    res = []
    try:
        conn = connect_mysql(mysql_config)
        cursor = conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchall()
        conn.commit()
    except Exception as e:
        print(f"数据库错误: {e}")
        conn.rollback()  # 发生错误时回滚
    finally:
        if conn:
            cursor.close()
            conn.close()
        return res


def str2json(s):
    try:
        s_json = json.loads(s)
    except:
        try:
            normalized = s.replace("'", '"')
            s_json = json.loads(normalized)  
        except json.JSONDecodeError:
            s_json = ast.literal_eval(s) 
    return s_json