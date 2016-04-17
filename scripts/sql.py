#!/usr/bin/python

import MySQLdb
import sys

#Deletes database for rebuilding. 
def clean():
    sql = "DROP DATABASE IF EXISTS NLP_VECTORS"

    cursor.execute(sql)
    
    #Commits changes. 
    db.commit()


#Creates database. 
def create_db():
    #Cleans before building. 
    clean()

    #Creates database if it doesn't exist. 
    sql = "CREATE DATABASE IF NOT EXISTS NLP_VECTORS"

    cursor.execute(sql)

    #Selects database.
    sql = "USE NLP_VECTORS"

    cursor.execute(sql)

    #Creates vector tables if doesn't exist yet. 
    sql = """CREATE TABLE IF NOT EXISTS WORD2VEC (
             WORD VARCHAR(%d) NOT NULL,
             VECTOR MEDIUMTEXT NOT NULL )""" % (max_word_len)

    cursor.execute(sql)

    sql = """CREATE TABLE IF NOT EXISTS GLOVE (
             WORD VARCHAR(%d) NOT NULL, 
             VECTOR MEDIUMTEXT NOT NULL )""" % (max_word_len)

    cursor.execute(sql)

    #Commits changes. 
    db.commit()

def build_table(table_name, file_name):
    if not table_name == "GLOVE" or table_name == "WORD2VEC":
        print "Wrong table name used! Try GLOVE or WORD2VEC silly."
    
    #Adds each vector to table. 
    for line in file_name:
        word, vector = line.split()

        sql = """INSERT INTO %s (WORD, VECTOR) 
                 VALUES('%s', '%s)""" % (word, vector)

        cursor.execute(sql)

    #Orders table in alphabetical order by word. 
    sql = """ALTER TABLE %s ORDER BY WORD"""
    cursor.execute(sql)

    #Commits changes. 
    db.commit()

if __name__ == "__main__":
    host = "localhost"
    user = "root"
    passwd = "julian"

    #Variables for table building. 
    max_word_length = 250

    #Opens database connection. 
    db = MySQLdb.connect(host, user, passwd)

    #Cursor object used to interact with MySQL
    cursor = db.cursor()

    #Ensures argument length correctness. 
    if len(sys.argv < 2):
        print "Need at least one argument!"

    #Command to execute in db. 
    command = sys.argv[1]

    #Different command cases. 
    if command == "clean":
        clean()
    elif command == "create_db":
        create_db()
    elif command == "build_table":
        #Ensures arguments are all there. 
        if not len(sys.argv == 4):
            print "Expecting 4 arguments!"
            print "Usage: $python sql.py build_table [table name] [word vector file name]"
        else:
            build_table(sys.argv[2], sys.argv[3])

    db.close()
