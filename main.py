from fastapi import FastAPI, File, UploadFile,HTTPException
from fastai.vision.all import *
from fastbook import *
# from fastai.learner import load_learner
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import __main__
import dill
from my_package import *
from fastapi.responses import StreamingResponse
import cv2
from fastapi.responses import HTMLResponse,FileResponse
import time
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse


# uvicorn main:app --host 0.0.0.0 --port 8000
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def label_function(files): pass
__main__.label_function = label_function
classifier = dill.load(open('modele/classif.pkl', 'rb'))

# =====================HOME========================================
@app.get("/")
async def read_root():
    content = """<!DOCTYPE html>
        <html>
        <head>
            <title>Restoration</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    text-align: center;
                    background-color: #f2f2f2;
                }
                h1 {
                    color: #333;
                }
                p {
                    color: #666;
                }
                button {
                    background-color: #007BFF;
                    color: #fff;
                    padding: 10px 20px;
                    border: none;
                    cursor: pointer;
                    margin: 10px;
                }
                button:hover {
                    background-color: #0056b3;
                }
            </style>
        </head>
        <body><center>
            <h1 color:blue>Demo Image Restoration  </h1>
            <p>IMPLEMENTATION OF A REAL-TIME IMAGE RESTORATION SOLUTION FOR VIDEO SURVEILLANCE</p>
            
            <button onclick="allerSurImage()">Image</button>
            <button onclick="allerSurVideo()">Video</button>
            <button onclick="allerSurCamera()">Camera</button>
            

            <script>
                function allerSurImage() {
                    window.location.href = "/image";
                }

                function allerSurVideo() {
                    window.location.href = "/video";
                }
             function allerSurCamera() {
                    window.location.href = "/camera";
                }
            </script>
            </center>
        </body>
        </html>
"""
    return HTMLResponse(content=content)

# FOR IMAGE RESTAUTATION 
# =====================================================================
@app.get("/image/")
async def image_root():
    content = """
    <html>
        <head>
            <title>Image Restoration</title>
            <script type="text/javascript">
                // Fonction pour vérifier si une image est chargée
                function checkFileInput() {
                    var fileInput = document.getElementById('fileInput');
                    var submitButton = document.getElementById('submitButton');

                    if (fileInput.value.length > 0) {
                        submitButton.disabled = false; // active le bouton si un fichier est sélectionné
                    } else {
                        submitButton.disabled = true;  // désactive le bouton s'il n'y a pas de fichier
                    }
                    }
                </script>
        </head>
        <body onload="checkFileInput()"> <!-- Vérifie dès le chargement de la page -->
            <center>
                <H1> Image Restoration </H1>
                <HR>
                <HR>

                <form action="/upload/" enctype="multipart/form-data" method="post">
                    <label for="fileInput">
                        Upload the image
                    </label>
                    <input type="file" name="file" id="fileInput"  onchange="checkFileInput()"> <!-- Vérifie chaque fois que le fichier change -->
                    <input type="submit" value="Restore" id="submitButton" disabled> <!-- Le bouton est désactivé par défaut -->
                </form>
            </center>
        </body>
    </html>
    """

    return HTMLResponse(content=content)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_bytes = await file.read()
    image_path =  "temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(file_bytes)
    temps_debut = time.time()
    processed_image , task = display_side_by_side_image(image_path, conditional_inferenceCV_im,classifier)
    temps_fin = time.time()
    ttime = round(temps_fin - temps_debut,2)
    output_path = "processed_image.jpg"
    cv2.imwrite(output_path, processed_image)
    content = f"""
        <html>
            <head>
                <title>Demo</title>
            </head>
            <body>
                <center>
                    <H1> Image Restoration </H1>
                    <HR>
                    <HR>                  
                    <img src="/get_image/" alt="Image processed" width="500" height="300">
                        <hr>
                        <p><H2><u>Operation info</u></H2></p>
                        <ul>
                            <li>Task:{task} </li>
                            <li>Execution time(s):{ttime} </li>
                        </ul>
                <center>
            </body>
        </html>
        """
    return HTMLResponse(content=content)


# VIDEO RESTAURATION 
############################################################

# # PLAY RESTORED VIDEO 

@app.get("/videom/", response_class=HTMLResponse)
async def home():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Lecture d'une vidéo depuis un fichier</title>
</head>
<body>
    <center>
    <h1>Video Restoration</h1>
    <form id="videoForm" action="/upload_video/" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".mp4, .avi">
        <input type="submit" value="Play">
    </form>
    <video id="videoPlayer" controls width="800" height="500" ></video>
    <script>
        const videoForm = document.getElementById("videoForm");
        const videoPlayer = document.getElementById("videoPlayer");

        videoForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            const formData = new FormData(videoForm);
            const response = await fetch("/upload_video/", {
                method: "POST",
                body: formData,
            });
            if (response.ok) {
                const { filename } = await response.json();
                const videoUrl = `/play/${filename}`;
                videoPlayer.src = videoUrl;
                videoPlayer.play(); // Jouer
            }
        });
    </script>
    </center>
</body>
</html>

    """
@app.post("/upload_video/")
async def upload_video(file: UploadFile=File(...)):
    video_storage = "/"
    if not file.filename.lower().endswith((".mp4", ".avi")):
        raise HTTPException(status_code=400, detail="Les fichiers vidéo MP4 et AVI sont pris en charge uniquement.")
    
    with open(f"{video_storage}{file.filename}", "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename}

@app.get("/play/{video_filename}")
async def play_video(video_filename: str):
    video_storage = "/"
    video_path = f"{video_storage}{video_filename}"
    return FileResponse(video_path, media_type="video/mp4")

@app.get("/get_image/")
async def get_image():
    return FileResponse("processed_image.jpg")


@app.get("/video/")
async def read_root():
    content = """
    <html>
        <head>
            <title>Restoration</title>
            <script type="text/javascript">
                // Fonction pour vérifier si une image est chargée
                function checkFileInput() {
                    var fileInput = document.getElementById('fileInput');
                    var submitButton = document.getElementById('submitButton');

                    if (fileInput.value.length > 0) {
                        submitButton.disabled = false; // active le bouton si un fichier est sélectionné
                    } else {
                        submitButton.disabled = true;  // désactive le bouton s'il n'y a pas de fichier
                    }
                    }
                </script>
        </head>
        <body onload="checkFileInput()"> <!-- Vérifie dès le chargement de la page -->
            <center>
                <H1> Video Restoration </H1>
                <HR>
                <HR>

                <form action="/uploadd/" enctype="multipart/form-data" method="post">
                    <label for="fileInput">
                        Upload the video
                    </label>
                    <input type="file" name="file" id="fileInput"  onchange="checkFileInput()"> <!-- Vérifie chaque fois que le fichier change -->
                    <input type="submit" value="Restore" id="submitButton" disabled> <!-- Le bouton est désactivé par défaut -->
                </form>
            </center>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

@app.post("/uploadd/")
async def upload_file(file: UploadFile = File(...)):
    file_bytes = await file.read()
    video_storage = "temp.mp4"
    with open(video_storage, "wb") as f:
        f.write(file_bytes)

    # Générer l'URL de streaming vidéo en utilisant url_for
    video_stream_url = app.url_path_for("video_stream")
    content = f"""
    <html>
    <head>
        <title>Video Restoration</title>
    </head>
    <body>
        <center>
        <h1>Video Restoration Demonstration</h1>
        <img src="{video_stream_url}" width="500" height="400" >
        </center>
    </body>
    </html>
    """

    return HTMLResponse(content=content)

@app.get("/video_stream")
async def video_stream():
    return StreamingResponse(
        gen_frames("temp.mp4", classifier),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

# VIDEO CAMERA  RESTAURATION 
############################################################

@app.get("/camera/")
async def camera():
    video_stream_url = app.url_path_for("video_stream2")
    # print('video_stream_url', video_stream_url)
    content = f"""
    <html>
    <head>
        <title>Video Restoration</title>
    </head>
    <body>
        <center>
        <h1>Video Restoration Demonstration</h1>
        <img src="{video_stream_url}" width="500" height="400" >
        </center>
    </body>
    </html>
    """

    return HTMLResponse(content=content)

@app.get("/video_stream2/")
async def video_stream2():
    return StreamingResponse(
        gen_frames2("1", classifier),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )
