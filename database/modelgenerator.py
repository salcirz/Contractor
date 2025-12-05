
#to run:
# docker run -it --rm -v "${PWD}:/app" -w /app contractor-dev python database/modelgenerator.py

#this file creates the AI models and saves them to the local folder 
import os
import mysql.connector
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
from torch.utils.data import Dataset, DataLoader


#check its working
print("PyTorch version:", torch.__version__)


#connect to db
conn = mysql.connector.connect(
    host="salvatorecirisano.cd2am0uc4s40.us-east-2.rds.amazonaws.com",
    user="admin",
    password="Salvatorecontractor1!",               
    database="salvatorecirisano",
    port=3306

    )

# Define all job types
jobtypeslist = [
    "Lawn Mowing", "Mulch", "Seeding", "Junk Removal",
    "Foliage Removal", "Weed Whacking", "Overgrown Grass",
    "Bush Trimming", "Weed Pulling", "Leaf Removal"
]


#img resize function data preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class ResNetWithJobType(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, image, job_type):
        
        #initial convolution, normalization, and activation.
        x = self.base_model.conv1(image)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        #deep residual blocks that extract image features.
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        job_type = job_type.unsqueeze(1)
        x = torch.cat([x, job_type], dim=1)

        #create last layer
        res = self.base_model.fc(x)
        return res
    

#dataset class for model interpretation
class ContractorDataset(Dataset):
    def __init__(self, images, jobtypes, labels):
        self.images = images
        self.jobtypes = jobtypes
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "job_type": self.jobtypes[idx],
            "label": self.labels[idx]
        }


for job in jobtypeslist:

    images, jobtypedata, labels = [], [], []

    #get all imgs
    querey = "SELECT * FROM picture WHERE type = %s"
    data = pd.read_sql(querey, conn, params=[job])


    #create # val for jobtypes
    jobtypes = list(data['type'].unique())
    data['job_type_encoded'] = data['type'].apply(lambda x: jobtypes.index(x))

    print("now modeling for" , job)
    #open all images and save with their data and pricing
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


    testdata = labels[0]


    #stack images and create vectors for model to process price data
    images = torch.stack(images)
    jobtypedata = torch.tensor(jobtypedata,dtype=torch.float32)
    labels = torch.tensor(labels, dtype = torch.float32)

    label_mean = labels.mean(dim=0, keepdim=True)
    label_std = labels.std(dim=0, keepdim=True)
    labels_norm = (labels - label_mean) / label_std



    dataset = ContractorDataset(images, jobtypedata, labels_norm)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)



    #pretrained model
    resnet = models.resnet18(weights = ResNet18_Weights.DEFAULT)


    #freeze layer weights to prevent overtraining
    for param in resnet.layer3.parameters():  # allow last 2 blocks to fine-tune
        param.requires_grad = True

    for param in resnet.layer4.parameters():  # allow last layer4 block to fine-tune
        param.requires_grad = True

    #keep all layers
    num_features = resnet.fc.in_features
    #add one feature (jobtype) 3 outputs price cost time
    resnet.fc = nn.Linear(num_features + 1, 3)


    #create new trained model
    model = ResNetWithJobType(resnet)


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    epochs = 200

    for epoch in range(epochs):
        for batch in dataloader:
            imgs = batch["image"]
            jobs = batch["job_type"]
            lbls = batch["label"]

            optimizer.zero_grad()
            outputs = model(imgs, jobs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    #example
    img_sample = images[0].unsqueeze(0)
    job_sample = jobtypedata[0].unsqueeze(0)
    print(jobtypedata[0])
    pred_norm = model(img_sample, job_sample)
    pred = pred_norm * label_std + label_mean  # den
    print("Predicted [price, cost, time]:", pred.detach().tolist())
    print("real price cost time" , testdata )


    modelfile= os.path.join("frontend", "models", job + "_mode.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "label_mean": label_mean,
        "label_std": label_std
    }, modelfile)

    print(f"Saved model to {modelfile}")

nn.close()
conn.close()
