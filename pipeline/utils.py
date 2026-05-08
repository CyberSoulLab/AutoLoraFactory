import numpy as np
import cv2

app = None

def get_face_app():
    global app
    if app is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis()
        app.prepare(ctx_id=0)
    return app

def get_embedding(img):
    app = get_face_app()
    faces = app.get(img)
    if not faces:
        return None
    return faces[0].embedding

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()