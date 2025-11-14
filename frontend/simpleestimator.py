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
import base64



import mysql.connector

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
    img = Image.open(image.stream).convert("RGB")
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
   

    result = {
        "jobtype": jobtype,
        "price": round(float(output[0]), 2),
        "cost": round(float(output[1]), 2),
        "time": round(float(output[2]), 2), 
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
