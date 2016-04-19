#!/usr/bin/python

import MySQLdb
import sys

# Usage to create database: python sql.py create_db 
# Usage to create one table: python sql.py build_table <"GLOVE" or "WORD2VEC"> <path to vectors>

class NLP_Database:
    def __init__(self):
        #Login info. 
        self.host = "localhost"
        self.user = "root"
        self.passwd = "julian"

        #Keeps track of table selection. 
        self.table = None
        self.num_rows = None
        self.rows_fetched = 0

        #Variables for table building. 
        self.max_word_len = 250

        #Opens database connection. 
        self.db = MySQLdb.connect(host, user, passwd)

        #Cursor object used to interact with MySQL
        self.cursor = self.db.cursor()

    def __del__(self):
        #Disconnects from database. 
        self.db.close()

    #This method loads one of the tables and all of its entries. 
    #Used before fetching entries. 
    def pick_table(self, table):
        if not (table == "GLOVE" or table == "WORD2VEC"):
            print "Incorrect table requested, try GLOVE or WORD2VEC"
            System.exit()
    
        #Pre-loads all entries and readies for fetching. 
        sql = "SELECT * FROM %s" % (table)
        self.num_rows = self.cursor.execute(sql)

        #Denotes that table has been selected. 
        self.table = table

    #Fetches next row from currently selected table.
    #Returns None if no next row available. 
    def fetch_one(self):
        if self.table == None:
            print "No table selected! Did you forget to call pick_table()?" 
            System.exit()
       
        #Returns None if no more rows to read. 
        if self.rows_fetched > self.num_rows:
            return None

        #Increments num fetched and returns row. 
        self.num_fetched += 1
        
        return self.cursor.fetchone()
        

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

#Builds either the GLOV or WORD2VEC tables in the db. 
def build_table(table_name, file_name):
    if not (table_name == "GLOVE" or table_name == "WORD2VEC"):
        print "Wrong table name used! Try GLOVE or WORD2VEC silly."
  
        sys.exit()

    #Selects database for use. 
    sql = "USE NLP_VECTORS"
    cursor.execute(sql)

    #Adds each vector to table. 
    word_file = open(file_name, 'r')

    for line in word_file:
        word, vector = line.split(' ', 1)

        #Escapes apostrophe and backslash by doubling them. 
        word = word.replace("'", "''").replace("\\", "\\\\")

        sql = """INSERT INTO %s (WORD, VECTOR) 
                 VALUES('%s', '%s')""" % (table_name, word, vector)

        cursor.execute(sql)

    #Orders table in alphabetical order by word. 
    sql = """ALTER TABLE %s ORDER BY WORD""" % (table_name)
    cursor.execute(sql)

    #Commits changes. 
    db.commit()

    word_file.close()

if __name__ == "__main__":
    #Ensures argument length correctness. 
    if len(sys.argv) < 2:
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
        if not len(sys.argv) == 4:
            print "Expecting 4 arguments!"
            print "Usage: $python sql.py build_table [table name] [word vector file name]"
        else:
            build_table(sys.argv[2], sys.argv[3])

    db.close()
