"""

docker run -it --rm -p 5000:5000 -v "${PWD}:/app" -w /app contractor-dev python frontend/simpleestimator.py

"""


from flask import Flask,request, render_template, jsonify, send_from_directory



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

@app.route("/registerpage.html")
def page ():
    return render_template("registerpage.html")

@app.route("/loginpage.html")
def getpage():
    return render_template("loginpage.html")




#heres the search for the db info 
@app.route("/getprice", methods = ["POST"])
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

    data = {

        "price": price,
        "cost": cost,
        "job": currentjob,
        "time": totaltime
    }

    return jsonify(data)

@app.route('/getimageprice', methods = ["POST"])
def getimageprices():

    jobtype = request.form["jobsearch"]

    querey = "Select path from picture where type = %s"

    db = getdb()
    cursor = db.cursor(dictionary = True)
    cursor.execute(querey, (currentjob,))
    result = cursor.fetchall()
    





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
