import cv2      # import des fonctions OpenCV
import numpy    # import des fonctions de calcul scientifique

imin = 1
imax = 75
chemin = '../DataSet 04/marple2'
fformat = '{}/marple2_{:02d}.jpg'

imin = 454
imax = 919
chemin = '../DataSet 04/tennis'
fformat = '{}/tennis{:03d}.jpg'

current = imin

while (1):
    name = fformat.format(chemin,current)
    frame = cv2.imread(name)
    scale_percent = 100
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    # ...
    # traitement de la frame actuelle
    # ...
    cv2.imshow("frame",frame)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    if current == imax:
        current = imin
    else:
        current = current + 1

cv2.destroyAllWindows()