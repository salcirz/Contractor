import csv
import pymysql

#Salvatorecontractor1!

# Connect to DB
conn = pymysql.connect(
    host="salvatorecirisano.cd2am0uc4s40.us-east-2.rds.amazonaws.com",
    port= 3306,
    user="admin",
    password="Salvatorecontractor1!",
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
