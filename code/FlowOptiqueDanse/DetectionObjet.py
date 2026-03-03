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

# Élément structurant pour la morphologie (ellips) de taille (5,5)pour nettoyer les petits points de bruit dans le masque
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

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

    #Magnitude = vitesse de chaque pixel
    #convertit (Δx, Δy) en (magnitude, angle) et on prend que la magnitude 
    mag, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # mag contient des valeurs float on les ramène entre 0 et 255 pour pouvoir seuiller
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')


    #seuillage noir et blanc
    seuil = 25
    _, masque_binaire = cv2.threshold(mag_norm, seuil, 255, cv2.THRESH_BINARY) # On ignore le premier retour _

    #Nettoyer le masque
    # MORPH_OPEN(ouverture) = érosion PUIS dilatation (supprime le bruit)
    # MORPH_CLOSE(fermeture) = dilatation PUIS érosion (rempli les trou)
    masque = cv2.morphologyEx(masque_binaire, cv2.MORPH_OPEN,  kernel)
    masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, kernel)

    #cherche les bords des formes blanches dans le masque binaire nettoyé
    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    #Dessiner les boîtes englobantes
    for contour in contours:

        # On ignore les très petits contours
        if cv2.contourArea(contour) > 500:

            # calcule le rectangle englobant 
            # x,y (coin supérieur gauche du rectangle)
            # w,h (largeur et hauteur du rectangle)
            x, y, w, h = cv2.boundingRect(contour)

            # Dessiner le rectangle 
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Fleche de direction
            centre_x = x + w // 2
            centre_y = y + h // 2
            dx = int(flow[centre_y, centre_x, 0] * 5)
            dy = int(flow[centre_y, centre_x, 1] * 5)
            cv2.arrowedLine(frame, (centre_x, centre_y), (centre_x + dx, centre_y + dy), (0, 255, 255), 2, tipLength=0.4)

    # Afficher les deux fenêtres
    cv2.imshow('frame',  frame)    # image originale + rectangles verts
    cv2.imshow('masque', masque)   # masque binaire

    image_precedente = image_suivante

    #lecture du clavier 30s pour sortir de la fenetre
    k = cv2.waitKey(30) & 0xff
    if k == 27: #27 represente la touche "echap" 
        break

cv2.destroyAllWindows()