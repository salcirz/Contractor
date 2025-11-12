
#to run:
# docker run -it --rm -v "${PWD}:/app" -w /app contractor-dev python database/modelgenerator.py

import os
import mysql.connector
import pandas as pd
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image

print("PyTorch version:", torch.__version__)


conn = mysql.connector.connect(
    host="salvatorecirisano.cd2am0uc4s40.us-east-2.rds.amazonaws.com",
    user="admin",
    password="Salvatorecontractor1!",               
    database="salvatorecirisano",
    port=3306

    )

querey = "select * from picture"

data = pd.read_sql(querey, conn)

conn.close()


jobtypes = list(data['type'].unique())
data['job_type_encoded'] = data['type'].apply(lambda x: jobtypes.index(x))

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

images, jobtypedata, labels = [], [], []

for index, row in data.iterrows():

    imagepath = os.path.join("frontend","static", row['path'] + ".jpg")

    if not os.path.exists(imagepath):
        print(f"Image not found: {imagepath}, skipping.")
        continue
    else:
        print("good")

    img = Image.open(imagepath).convert("RGB")
    img = transform(img)
    images.append(img)
    jobtypedata.append(row["job_type_encoded"])
    labels.append([row["price"], row["cost"], row["time"]])


images = torch.stack(images)
jobtypedata = torch.tensor(jobtypedata,dtype=torch.float32)
labels = torch.tensor(labels, dtype = torch.float32)


