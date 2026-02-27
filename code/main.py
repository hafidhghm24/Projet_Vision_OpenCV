import cv2 
import numpy as np


PATH_VIDEO = '/home/hafidh/Documents/GitHub/Projet_Vision_OpenCV/data/VIRAT03.mp4' 

#Initialisation de la capture
cap = cv2.VideoCapture(PATH_VIDEO) 

print("Début de la lecture ! Appuie sur 'Echap' pour quitter.")
while True:
    # Lecture de l'image actuelle
    ret, frame = cap.read() 
    
    #Si la vidéo est introuvable, on arrête la boucle
    if not ret: 
        print("Fin de la vidéo ou erreur de lecture.")
        break
        
    # --- ALGORITHMES---






    #---------------------------
    
    #Affichage de l'image
    cv2.imshow('Ma Video OpenCV', frame) 
    
    #Laisser 30s dattente entre les frames pour lire le clavier
    k = cv2.waitKey(30) & 0xff 
    if k == 27: # 27 correspond à la touche 'Echap'
        print("Fermeture par l'utilisateur.")
        break 


#libération de la mémoire
cap.release()
cv2.destroyAllWindows()