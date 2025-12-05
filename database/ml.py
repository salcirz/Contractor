#this is used to fill the database with the csv file

import csv
import mysql.connector

# Connect to MySQL on the server
conn = mysql.connector.connect(
    host="salvatorecirisano.cd2am0uc4s40.us-east-2.rds.amazonaws.com",
    user="admin",
    password="Salvatorecontractor1!",               
    database="salvatorecirisano",
    port=3306
)

with conn.cursor() as cursor:
    with open("Insert.csv", newline='') as f:
        # CSV with comma delimiter and optional quotes
        reader = csv.reader(f)
        
        for row in reader:
            jobtype, price,time, filepath, cost = row

            # Remove commas from numeric values
            price = price.replace(',', '')
            cost = cost.replace(',', '')

            # Insert into table (id is auto-incremented)
            cursor.execute(
            "INSERT INTO picture (type, price,time, path, cost) VALUES (%s, %s, %s, %s, %s)",
            (jobtype, price,time, filepath, cost)
            )

conn.commit()
conn.close()
