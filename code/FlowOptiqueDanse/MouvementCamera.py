import cv2
import numpy

#adapter le chemin a chaque video
#fichier = 'data/VIRAT01.mp4'
fichier = 'data/VIRAT03.mp4'

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

    # ANALYSE MOUVEMENT CAMÉRA
    moy_x = numpy.mean(flow[..., 0])  # mouvement moyen horizontal
    moy_y = numpy.mean(flow[..., 1])  # mouvement moyen vertical

    # Vitesse globale cest la longueur du vecteur moyen (Pythagore)
    vitesse = numpy.sqrt(moy_x**2 + moy_y**2)


    # en dessous du seuil (brui), on considère la caméra fixe
    seuil_camera = 0.5

    if vitesse < seuil_camera:
        # Vitesse trop faible ( caméra considérée fixe)
        label = "Camera : Fixe"
        couleur = (0, 255, 0)   # vert

    elif abs(moy_x) > abs(moy_y):
        # moy_x > 0 : les pixels vont vers la droite
        direction = "Droite" if moy_x > 0 else "Gauche"
        label= f"Panoramique {direction}"
        couleur = (0, 165, 255)   # orange

    else:
        # moy_y > 0 : les pixels vont vers le bas
        direction = "Bas" if moy_y > 0 else "Haut"
        label= f"Travelling {direction}"
        couleur = (255, 0, 0)   # bleu


    # VISUALISATION DU FLÔT DENSE
    mag, ang= cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / numpy.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # AFFICHAGE DU TEXTE sur l'image
    cv2.putText(frame, label,(20, 50),cv2.FONT_HERSHEY_SIMPLEX,1.0,couleur,2)

    cv2.imshow('frame',  frame)   # image originale + texte
    cv2.imshow('frame2', bgr)     # flôt dense coloré

    prvs = image_suivante   # mise à jour pour la prochaine image

    #lecture du clavier 30s pour sortir de la fenetre
    k = cv2.waitKey(30) & 0xff
    if k == 27: #27 represente la touche "echap" 
        break

cv2.destroyAllWindows()