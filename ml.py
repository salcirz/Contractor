import csv
import pymysql

# Connect to MySQL on the server
conn = pymysql.connect(
    host="localhost",
    user="salvatorecirisano",
    password="",               # empty password
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
            "INSERT INTO picture (jobtype, price,time, filepath, cost) VALUES (%s, %s, %s, %s, %s)",
            (jobtype, price,time, filepath, cost)
            )

conn.commit()
conn.close()
