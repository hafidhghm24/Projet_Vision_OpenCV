import cv2
import numpy

# ============================================================
# PARAMÈTRES — on utilise PTZ001 cette fois
# data/PTZ001/in000001.jpg  →  imin=1, imax=100
# ============================================================
imin    = 1
imax    = 100
chemin  = 'data/PTZ001'
fformat = '{}/in{:06d}.jpg'

# ============================================================
# INITIALISATION — première image
# ============================================================
name      = fformat.format(chemin, imin)
old_frame = cv2.imread(name)

if old_frame is None:
    print(f"ERREUR : impossible de lire {name}")
    exit()

prvs = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Image HSV pour la visualisation colorée
hsv         = numpy.zeros_like(old_frame)
hsv[..., 1] = 255    # saturation max = couleurs toujours vives

# ============================================================
# BOUCLE PRINCIPALE
# ============================================================
current = imin + 1

while True:
    name  = fformat.format(chemin, current)
    frame = cv2.imread(name)

    if frame is None:
        break

    scale_percent = 100
    width  = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame  = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # FLÔT DENSE — Farnebäck (du PDF du prof)
    # --------------------------------------------------------
    flow = cv2.calcOpticalFlowFarneback(prvs, next_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    # --------------------------------------------------------
    # VISUALISATION HSV (du PDF du prof)
    # direction → couleur  |  vitesse → luminosité
    # --------------------------------------------------------
    mag, ang    = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / numpy.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr         = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame',  frame)   # image originale
    cv2.imshow('frame2', bgr)     # flôt dense coloré

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break

    if current == imax:
        current = imin + 1
    else:
        current += 1

    prvs = next_gray

cv2.destroyAllWindows()