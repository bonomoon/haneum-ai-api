from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from io import BytesIO
import io
import os
import datetime
from fastapi.middleware.cors import CORSMiddleware
import json
origins = [
    "*",
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # cross-origin request에서 cookie를 포함할 것인지 (default=False)
    allow_methods=["*"],     # cross-origin request에서 허용할 method들을 나타냄. (default=['GET']
    allow_headers=["*"],     # cross-origin request에서 허용할 HTTP Header 목록
)
import torch
from PIL import Image
from torchvision import transforms
from timm import create_model
with open("labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

def load_model(path, num_classes):
    model = create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(path))
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def predict(model, image_path):
    device = 'cpu'
    image = preprocess_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

model = load_model('cell_recog_model.pth', 19)
model = model.to('cpu')
model.eval()

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # 파일 객체를 numpy 배열로 변환
        image_stream = BytesIO(await file.read())
        image_array = np.frombuffer(image_stream.getvalue(), dtype=np.uint8)
        # OpenCV를 사용하여 numpy 배열에서 이미지 읽기
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        img = Image.open(image_stream)
        gray_scale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _,img_bin = cv2.threshold(gray_scale,150,225,cv2.THRESH_BINARY)
        img_bin=~img_bin
        line_min_width = 15
        kernal_h = np.ones((1,line_min_width), np.uint8)
        kernal_v = np.ones((line_min_width,1), np.uint8)
        img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
        img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)

        # MIX Kernel
        img_bin_final=img_bin_h|img_bin_v
        final_kernel = np.ones((3,3), np.uint8)
        img_bin_final=cv2.dilate(img_bin_final, final_kernel, iterations=1)
        _, __, stats,_ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
        xpos, ypos = -1,0
        aw = sum([x[2] for x in stats[2:]])/len(stats[2:])
        ah = sum([x[3] for x in stats[2:]])/len(stats[2:])
        prex = 0
        preds = []
        temp = str(datetime.datetime.now()).split()[1].replace(".","").replace(":","")
        os.mkdir(temp)
        for x,y,w,h,area in stats[2:]:
            if aw*2> w > 10  and ah*2 > h >10:
                if x < prex:
                        ypos += 1
                        xpos = 0
                        prex = 0
                else:
                        xpos += 1
                        prex = x
                cr =img.crop((x, y, x+w, y+h))  
                cr.save(f"{temp}/{xpos,ypos}.png")
        for cell in sorted(os.listdir(f"{temp}"), key= lambda x: (-int(eval(x.split(".")[0])[0]), int(eval(x.split(".")[0])[1]))):
             prediction = labels[predict(model, f"{temp}/{cell}")]
             if prediction != "공백":
                  for p in prediction.split():
                       preds.append(p)
                
        for png in os.listdir(f"{temp}"):
            os.remove(f"{temp}/{png}")
        os.rmdir(temp)
        return JSONResponse(content={"notes": preds})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
