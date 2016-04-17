#!/usr/bin/python

import MySQLdb

host = "localhost"
user = "root"
passwd = "nlp"

#Opens database connection. 
db = MySQLdb.connect(host, user, passwd)

#Cursor object used to interact with MySQL
cursor = db.cursor()

#Creates database if it doesn't exist. 
sql = "CREATE DATABASE IF NOT EXISTS TESTDB"

cursor.execute(sql)

#Selects database.
sql = "USE TESTDB"

cursor.execute(sql)

#Creates table if doesn't exist yet. 
sql = """CREATE TABLE IF NOT EXISTS EMPLOYEE (
         FIRST_NAME CHAR(20) NOT NULL,
         LAST_NAME CHAR(20),
         AGE INT,
         SEX CHAR(1),
         INCOME FLOAT )"""

cursor.execute(sql)

#Inserts element into table. 
sql = """INSERT INTO EMPLOYEE (FIRST_NAME,
                               LAST_NAME,
                               AGE,
                               SEX,
                               INCOME)
                   VALUES('%s', '%s', '%d', '%c', '%d')""" % \
                         ('Mac', 'Mohan', 20, 'M', 2000)

try:
    cursor.execute(sql)

    #Commit changes to database. 
    db.commit()

except:
    #Rollback in case of error. 
    db.rollback()

    print "Error, had to rollback!"

#Reads from database using condition. 
sql = "SELECT * FROM EMPLOYEE WHERE INCOME > '%d'" % (1000)

cursor.execute(sql)

#Fetch all the rows in a list of lists. 
results = cursor.fetchall()

for row in results:
    fname = row[0]
    lname = row[1]
    age = row[2]
    sex = row[3]
    income = row[4]

    #prints result
    print "fname = %s, lname = %s, age = %d, sex = %c, income = %d" % \
           (fname, lname, age, sex, income)

sql = "UPDATE EMPLOYEE SET AGE = AGE + 1 WHERE SEX = '%c'" % ('M')

cursor.execute(sql)
db.commit()

sql = "DELETE FROM EMPLOYEE WHERE AGE > '%d'" % (20)

cursor.execute(sql)
db.commit()

db.close()
