from flask import Flask,request, render_template

import mysql.connector

app = Flask(__name__, template_folder=".")

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
    return render_template("mainpicturepage.html")


#heres the search for the db info 
@app.route("/search", methods= ["POST"])
def search():
    currentjob = request.form["jobsearch"] 

    querey = "Select * from picture where type = %s"

    db = getdb()

    cursor = db.cursor(dictionary = True)

    cursor.execute(querey, (currentjob,))

    results = cursor.fetchall()

    db.close()

    html = ""
    for row in results:
        html += f"<p>{row}</p>"

    return html


if __name__ == "__main__":
    app.run(debug=True)








