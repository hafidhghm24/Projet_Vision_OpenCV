import cv2
import numpy

# ============================================================
# PARAMÈTRES — chemin de ta vidéo
# ============================================================
fichier = '../../data/VIRAT.mp4'   # ← remplace par ton vrai fichier
cap     = cv2.VideoCapture(fichier)

if not cap.isOpened():
    print(f"ERREUR : impossible d'ouvrir {fichier}")
    exit()

# ============================================================
# INITIALISATION — première frame
# ============================================================
ret, old_frame = cap.read()
prvs           = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

hsv         = numpy.zeros_like(old_frame)
hsv[..., 1] = 255

# ============================================================
# BOUCLE PRINCIPALE
# ============================================================
while True:
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        continue

    scale_percent = 100
    width  = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame  = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Flôt dense Farnebäck
    flow = cv2.calcOpticalFlowFarneback(prvs, next_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    # Visualisation HSV
    mag, ang    = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / numpy.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr         = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame',  frame)
    cv2.imshow('frame2', bgr)

    prvs = next_gray

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()