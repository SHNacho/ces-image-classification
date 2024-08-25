import io

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.templating import Jinja2Templates
from PIL import Image

from app.captioner import Captioner, CaptioningModels
from app.classifier import Classifier, ClassifierModels


app = FastAPI()
templates = Jinja2Templates(directory='templates')

captioner = Captioner()
classifier = Classifier()


@app.post("/classify-image/")
async def classify_image(
        file: UploadFile = File(...),
        captioning_model: CaptioningModels = CaptioningModels.blip,
        classifier_model: ClassifierModels = ClassifierModels.distilbert
    ):
    try:
        # Read the uploaded image file
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        caption = captioner.generate_caption(captioning_model, image)
        predicted_label = classifier.predict(caption, classifier_model, captioning_model)

        return {"class": predicted_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/")
async def form(request: Request):
    form = await request.form()
    classifier_model = form.get('classifier_model')
    captioning_model = form.get('captioning_model')
    file = form.get('file')

    response = await classify_image(file, captioning_model, classifier_model)
    
    return templates.TemplateResponse("home.html", {"request": request, "result": response})
