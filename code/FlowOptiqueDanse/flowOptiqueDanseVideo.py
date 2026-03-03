import cv2
import numpy

#adapter le chemin a chaque video
fichier = 'data/VIRAT01.mp4'
#fichier = 'data/VIRAT03.mp4.mp4'

cap= cv2.VideoCapture(fichier)

if not cap.isOpened():
    print(f"Probléme de lécture (flowOptiqueDanseVideo) {fichier}")
    exit()

# INITIALISATION
ret, image = cap.read()

#convertire "image" en niveau de gris
image_precedente = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# image en HSV pour voir les couleurs
hsv = numpy.zeros_like(image) #on crée une image vide
hsv[..., 1] = 255 # saturation max = couleurs toujours vives



while (1):
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        image_precedente = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        continue

    scale_percent = 100
    width  = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame  = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    image_suivante = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Methode de Farneback
    flow = cv2.calcOpticalFlowFarneback(image_precedente, image_suivante, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Visualisation du HSV
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]) #On transforme les coordonnées cartésiennes en polaires (pour avoir la vitesse et direction)
    
    hsv[..., 0] = ang * 180 / numpy.pi / 2 #bleu pour la gauche, rouge pour la droite
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) #plus le point bouge vite plus il est brillant
    
    #on repasse en couleur BGR pour visualisation
    imageBGR = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame', frame)   # affichage de l'image originale
    cv2.imshow('frame2', imageBGR)     # affichage du flôt dense coloré

    image_precedente = image_suivante

    #lecture du clavier 30s pour sortir de la fenetre
    k = cv2.waitKey(30) & 0xff
    if k == 27: #27 represente la touche "echap" 
        break

cv2.destroyAllWindows()