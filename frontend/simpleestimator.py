"""

docker run -it --rm -p 5000:5000 -v "${PWD}:/app" -w /app contractor-dev python frontend/simpleestimator.py


"""

from flask import Flask,request, render_template, jsonify, send_from_directory, session
import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
from io import BytesIO
import mysql.connector
from dotenv import load_dotenv
import boto3
import os
import uuid
from argon2 import PasswordHasher

load_dotenv()

ph = PasswordHasher()
hashed = ph.hash("mypassword123")
ph.verify(hashed, "mypassword123")


AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET = os.getenv("S3_BUCKET")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

app = Flask(__name__)
app.secret_key = "secretttkeke"



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

@app.route("/reportpage")
def getreportpage():
    return render_template("reportpage.html")

@app.route("/getinfo")
def getinfo():
    return jsonify(session["result"])

@app.route("/getsessioninfo")
def getsessioninfo():
    return jsonify(session.get("username"))

@app.route("/clearsession")
def clearsession():
    session.clear()
    return render_template("main.html")

@app.route("/viewjobs")
def viewjobs():
    return render_template("viewjobs.html")



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

    session['linearoutput'] = data
    return jsonify(data)



@app.route('/getimageprice', methods=["POST"])
def getimageprices():
    jobtype = request.form["jobsearch"]
    image = request.files["fileinput"]

    image_bytes = image.read()
    image_stream = BytesIO(image_bytes)  # For PIL processing
    upload_stream = BytesIO(image_bytes) # For S3 upload


    filename = f"{uuid.uuid4()}_{image.filename}"

    s3.upload_fileobj(upload_stream, Bucket= S3_BUCKET, Key=filename)

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load model checkpoint
    modelpath = os.path.join("frontend", "models", jobtype + "_mode.pth")
    if not os.path.exists(modelpath):
        raise FileNotFoundError(f"Model not found: {modelpath}")

    checkpoint = torch.load(modelpath, map_location="cpu")

    # Recreate model architecture

    from torchvision import models
    resnet = models.resnet18(weights=None)
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features + 1, 3)  # +1 for job type


    model = ResNetWithJobType(resnet)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    label_mean = checkpoint["label_mean"]
    label_std = checkpoint["label_std"]

    # Preprocess image
    img = Image.open(image_stream).convert("RGB")
    img = transform(img).unsqueeze(0)

    # Encode job type numerically (match training encoding)
    jobtypeslist = [
        "Lawn Mowing", "Mulch", "Seeding", "Junk Removal",
        "Foliage Removal", "Weed Whacking", "Overgrown Grass",
        "Bush Trimming", "Weed Pulling", "Leaf Removal"
    ]
    job_index = torch.tensor([jobtypeslist.index(jobtype)], dtype=torch.float32)

    # Predict
    with torch.no_grad():
        output_norm = model(img, job_index)
        output = output_norm * label_std + label_mean

    output = output.squeeze().tolist()
     
    file_url = (
        f"https://{os.getenv('S3_BUCKET')}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{filename}"
    )

    result = {
        "jobtype": jobtype,
        "price": round(float(output[0]), 2),
        "cost": round(float(output[1]), 2),
        "time": round(float(output[2]), 2), 
        "file" : file_url,
    }
    
    print("Predicted:", result)
    
    session["result"] = result    
    return jsonify(result)


class ResNetWithJobType(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, image, job_type):
        # Run through ResNet layers
        x = self.base_model.conv1(image)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)

        # Add job type feature
        job_type = job_type.unsqueeze(1)
        x = torch.cat([x, job_type], dim=1)

        # Final output
        out = self.base_model.fc(x)
        return out

@app.route('/usernameexists', methods = ["POST"])
def usernameexists():

    db = getdb()

    cursor = db.cursor(dictionary= True)

    tempuse = request.form["username"]
    tempuser = tempuse.strip()

    querey = "select * from users where username = %s"

    cursor.execute(querey, (tempuser,))

    result = cursor.fetchall()
    print(result, " hello")
    if not result:
        print("false")
        data = {
            "userexist": False,
        }
    else:
        print("true")
        data = {
            "userexist" : True,
        }

    db.close()
    return jsonify(data)

 
@app.route('/registeruser', methods= ["POST"])
def registeruser():
    db = getdb()
    cursor = db.cursor(dictionary= True)
    
    ph = PasswordHasher()
    hashedpw = ph.hash(request.form["password"].strip())

    username = request.form["username"].strip()
    email = request.form["email"].strip()

    querey = "insert into users (username, password, email) values (%s, %s, %s)"

    cursor.execute(querey, (username, hashedpw,email))
    db.commit()

    db.close()
    cursor.close()

    userdata ={
        "username":username,
    }
    session["username"]  = userdata

    return None


@app.route('/loginuser', methods=["POST"])
def loginuser():

    db = getdb()
    cursor = db.cursor()

    username = request.form["username"].strip()  
    password = request.form["password"].strip()

    querey = "select * from users where username = %s"

    cursor.execute(querey, (username,))
    results = cursor.fetchone()
    

    if not results:
        return jsonify({"correct": False})
    else:
        ph = PasswordHasher()
        try:
            ph.verify(results[1].strip(), password)

            usernamedata = {
                "username": username,
            }
            session["username"] = usernamedata
            return jsonify({"correct": True})
        except Exception as e:
            return jsonify({"correct": False})
        

@app.route('/savejob', methods = ["POST"])
def savejob():
    db = getdb()


    cursor = db.cursor()
    querey = "insert into alljobs(filepath, date, price, time,cost,username, jobtype, linearprice, linearcost, inputtime) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)" 
    cjd = request.get_json()
    print("Received JSON:", cjd)

    lod = session["linearoutput"] 

    ltime = lod["time"]
    lprice = lod["price"]
    lcost = lod["cost"]
     

    cursor.execute(querey,(cjd["file"],cjd["date"],cjd["price"],cjd["time"],cjd["cost"],cjd["username"],cjd["type"],lprice, lcost,ltime))
    
    db.commit()
    cursor.close()
    db.close()

    return jsonify({"Status" : "success"})


@app.route('/getusersjobs', methods =["POST"])
def getusersjobs():

    db = getdb()  
    cursor = db.cursor(dictionary=True)

    querey = "select * from alljobs where username = %s"

    user = session["username"]["username"]

    cursor.execute(querey, (user,))
    results = cursor.fetchall()

    print(results)

    cursor.close()
    db.close()
    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
