import cv2      # import des fonctions OpenCV
import numpy    # import des fonctions de calcul scientifique

#emplacement du fichier video
fichier = '../VIRAT/VIRAT_S_000203_09_001789_001842.mp4' 

#ouvrire la video
cap = cv2.VideoCapture(fichier)

while (1):
    ret, frame = cap.read()
    if not ret:
       cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
       ret, frame = cap.read ()
    scale_percent = 100
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    # ...
    # traitement de la frame actuelle
    # ...
    cv2.imshow("frame",frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()