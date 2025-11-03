from flask import Flask,request, render_template

import mysql.connector

app = Flask(__name__)

#db connection
def getdb ():

    conn = mysql.connector.connect(
    host="salvatorecirisano.cd2am0uc4s40.us-east-2.rds.amazonaws.com",
    user="admin",
    password="Salvatorecontractor1!",               
    database="salvatorecirisano",
    port=3306

    )

    return conn


#route to main 
@app.route("/")
def home():
    return render_template("main.html")


#heres the search for the db info 
@app.route("/getprice", methods= ["POST"])
def search():

    currentjob = request.form["jobsearch"] 
    totaltime = request.form["numberinput"]

    querey = "Select * from units where type = %s"

    db = getdb()
    cursor = db.cursor(dictionary = True)
    cursor.execute(querey, (currentjob,))
    result = cursor.fetchall()

    results = result[0]

    price = round(results["price"] * float(totaltime), 2)
    cost = round(results["cost"]  * float(totaltime), 2 )

    db.close()

    return render_template("main.html",price = price, cost = cost, time = totaltime, job = currentjob)


if __name__ == "__main__":
    app.run(debug=True)








