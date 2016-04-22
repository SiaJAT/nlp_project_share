import MySQLdb
import sys

# overhead to setup interface to mysql
db = MySQLdb.connect("localhost", "root", "julian", "NLP_VECTORS")
cursor = db.cursor()

# sql query the database specified by argv[1]
sql = "SELECT * FROM " + sys.argv[1]

try:
    num_lines = cursor.execute(sql)
    num_lines = int(num_lines)
    for i in xrange(0, num_lines):
        cursor.fetchone()        
except:
    print "something went wrong"

db.close()
