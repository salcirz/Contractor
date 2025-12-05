
#this file is used to generate the unit table to make linear estimations

import mysql.connector


conn = mysql.connector.connect(
    host="salvatorecirisano.cd2am0uc4s40.us-east-2.rds.amazonaws.com",
    user="admin",
    password="Salvatorecontractor1!",               
    database="salvatorecirisano",
    port=3306
)


jobtypes = ["Lawn Mowing", "Foliage Removal", "Weed Whacking", 
            "Mulch","Overgrown Grass","Weed Pulling", 
            "Seeding", "Bush Trimming", "Leaf Removal", 
            "Junk Removal"]


with conn.cursor(dictionary=True) as cursor:

    for job in jobtypes :

        querey = "Select * from picture where type = %s"

        cursor.execute(querey, (job,))

        results = cursor.fetchall()
        
        price = 0
        cost = 0
        time = 0
        count = 0

        for row in results:

            price += row["price"]
            cost += row["cost"]
            time += row["time"]

            count += 1
        
        price = price/count
        cost = cost/count

        time = time/count

        cost = cost/time
        price = price/time
        
        q = "insert into units (type, price, cost) values (%s,%s,%s)"

        cursor.execute(q, (job, price, cost,))
    
conn.commit()
