import csv
import pymysql

# Connect to DB
conn = pymysql.connect(
    host="localhost",
    port= 3306,
    user="salvatorecirisano",
    password="",
    database="salvatorecirisano"

)
with conn.cursor as cursor:
    with open("Insert.csv", "r") as f:
        reader = csv.reader(f)
    
        for row in reader:
            cursor.execute(

                "Insert into picture(jpbtype, price, filepath, cost) Values (%s,%s,%s,%s)",
                row

            )

conn.commit()
conn.close()
