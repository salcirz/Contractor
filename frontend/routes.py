from flask import Flask,request

import mysql.connector

app = Flask(__name__)

def getdb ():

    conn = mysql.connector.connect(

        host = "compsci.adelphi.edu",
        user= "salvatorecirisano",
        password = "",
        database = "salvatorecirisano"

    )

    return conn


@app.route("/search", methods= ["POST"])
def search():
    currentjob = request.form["jobsearch"] 

    querey = "Select * from picture where jobtype = %s"

    db = getdb()

    cursor = db.cursor(dictionary = True)

    cursor.execute(querey, (currentjob,))

    results = cursor.fetchall()

    db.close()

    html = ""
    for row in results:
        html += f"<p>{row}</p>"

    return html








