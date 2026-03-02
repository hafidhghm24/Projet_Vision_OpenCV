import cv2
import numpy

#On travaille en niveaux de gris car le flôt optique se base sur les variations de luminance
# La pyramide sert à capturer les mouvement trop rapide pour que ça rentre dans la fenetre 

#adapter le chemin a chaque image
imin= 1
imax= 100
chemin  = 'data/PTZ001'
fformat = '{}/in{:06d}.jpg'

# INITIALISATION
name = fformat.format(chemin, imin)
image = cv2.imread(name)

# Vérification que l'image existe
if image is None:
    print(f"Probléme de lécture (flowOptiqueDanseImage){name}")
    exit()

#convertire "image" en niveau de gris
image_precedente = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# image en HSV pour voir les couleurs
hsv = numpy.zeros_like(image) #on crée une image vide
hsv[..., 1] = 255 # saturation max 255 pour bien voir les couleurs de mouvement



current = imin + 1

while (1):
    name = fformat.format(chemin, current)
    frame = cv2.imread(name)

    if frame is None:
        break

    scale_percent = 100
    width  = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame  = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    image_suivante = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Methode de Farneback
    flow = cv2.calcOpticalFlowFarneback(image_precedente, image_suivante, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Visualisation du HSV en couleur
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]) #On transforme les coordonnées cartésiennes en polaires (pour avoir la vitesse et direction)
    
    hsv[..., 0] = ang * 180 / numpy.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    #on repasse en couleur BGR pour visualisation
    imageBGR = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame',  frame)   # affichage de l'image originale
    cv2.imshow('frame2', imageBGR)     # affichage du flôt dense coloré


    #lecture du clavier 100s pour sortir de la fenetre
    k = cv2.waitKey(100) & 0xff
    if k == 27: #27 represente la touche "echap" 
        break

    if current == imax:
        current = imin + 1
    else:
        current += 1

    image_precedente = image_suivante

cv2.destroyAllWindows()