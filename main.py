import joblib
import numpy as np
from fastapi import FastAPI
from fastapi import UploadFile, File, Request
import cv2
from io import BytesIO
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Predicting Covid Class")

#app.mount("/static", StaticFiles(directory="static"), name="static")
#templates = Jinja2Templates(directory="/app/templates")


@app.on_event("startup")
def load_clf():
    # Load classifier from pickle file
    path='Covid_model'
    global clf
    clf = joblib.load("/app/model.pkl")


@app.get("/")
def home(): #request: Request
    return "Hello" #templates.TemplateResponse("index.html", {"request": request})


@app.post("/")
async def home_predict( file: UploadFile = File(...) ): #request: Request,
    
    #img = cv2.imread(file.filename)
    img_bytes=file.file.read()
    nparr = np.fromstring(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (150,150))
    image = np.expand_dims(image, 0)
    image = image / 255.0
    pred = clf.predict(image)[0][0]
    
    if pred < 0.5:
    	out="Covid"
    else:
    	out="Normal"
    
    if pred < 0.5:
    	score=(1-pred) *100
    else:
    	score=pred *100	
    	
    result={"Prediction": out, "Confidence_Score" : str(score) + "%" }
    return result#templates.TemplateResponse("index.html", {"request": request, 'result': result})
    
@app.post("/predict")
async def predict(file: UploadFile = File(...) ):
    
    #img = cv2.imread(file.filename)
    img_bytes=file.file.read()
    nparr = np.fromstring(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (150,150))
    image = np.expand_dims(image, 0)
    image = image / 255.0
    pred = clf.predict(image)[0][0]
    
    if pred < 0.5:
    	out="Covid"
    else:
    	out="Normal"
    
    if pred < 0.5:
    	score=(1-pred) *100
    else:
    	score=pred *100	
    return {"Prediction": out, "Confidence Score" : str(score) + "%"}
    
    
    
