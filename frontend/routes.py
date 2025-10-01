from flask import Flask,request, render_template

import mysql.connector

app = Flask(__name__, template_folder=".")

def getdb ():

    conn = mysql.connector.connect(

        host = "localhost",
        port = "3307",
        user= "salvatorecirisano",
        password = "",
        database = "salvatorecirisano"

    )

    return conn

@app.route("/")
def home():
    return render_template("mainpicturepage.html")

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


if __name__ == "__main__":
    app.run(debug=True)








