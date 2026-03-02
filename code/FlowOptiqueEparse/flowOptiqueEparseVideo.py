import cv2
import numpy

# ============================================================
# PARAMÈTRES — mets le chemin de TA vidéo ici
# Exemple: '../../data/mavideo.mp4'
# ============================================================
fichier = '../../data/VIRAT.mp4'   # ← remplace par ton vrai fichier
cap     = cv2.VideoCapture(fichier)

if not cap.isOpened():
    print(f"ERREUR : impossible d'ouvrir {fichier}")
    print("Vérifie que le fichier vidéo existe dans data/")
    exit()

# ============================================================
# PARAMÈTRES DU DÉTECTEUR SHI-TOMASI (du PDF du prof)
# ============================================================
feature_params = dict(
    maxCorners   = 100,
    qualityLevel = 0.3,
    minDistance  = 7,
    blockSize    = 7
)

# ============================================================
# PARAMÈTRES LUCAS-KANADE (du PDF du prof)
# ============================================================
lk_params = dict(
    winSize  = (15, 15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

color = numpy.random.randint(0, 255, (100, 3))

# ============================================================
# INITIALISATION — première frame
# ============================================================
ret, old_frame = cap.read()
old_gray       = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0             = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask           = numpy.zeros_like(old_frame)

# ============================================================
# BOUCLE PRINCIPALE
# ============================================================
while True:
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p0       = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask     = numpy.zeros_like(frame)
        continue

    scale_percent = 100
    width  = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame  = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask  = cv2.line(mask,    (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)

        old_gray = frame_gray.copy()
        p0       = good_new.reshape(-1, 1, 2)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()