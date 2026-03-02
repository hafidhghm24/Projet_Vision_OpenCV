import cv2
import numpy

#adapter le chemin a chaque image
imin    = 1
imax    = 234
chemin  = 'data/marple17'
fformat = '{}/marple17_{:03d}.jpg' #remplacer les {} par les le chemin et le {} par le numero de limage


# PARAMÈTRES DE SHI-TOMASI
feature_params = dict(
    maxCorners   = 100,
    qualityLevel = 0.3,
    minDistance  = 7,
    blockSize    = 7
)


# PARAMÈTRES DE LUCAS-KANADE
lk_params = dict(
    winSize  = (15, 15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Couleurs aléatoires pour les trajectoires
color = numpy.random.randint(0, 255, (100, 3))


# INITIALISATION
name = fformat.format(chemin, imin) 
image = cv2.imread(name) #on lit la premiére image pour comparé

# Vérification que l'image existe
if image is None:
    print(f"Probléme de lecture (flowOptiqueEparseVideo){name}")
    exit()

#convertire "image" en niveau de gris
image_precedente = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


p0       = cv2.goodFeaturesToTrack(image_precedente, mask=None, **feature_params)
mask     = numpy.zeros_like(image)


current = imin + 1

while (1):
    name  = fformat.format(chemin, current)
    frame = cv2.imread(name)

    if frame is None:
        break

    scale_percent = 100
    width  = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame  = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcul Lucas-Kanade
    p1, st, err = cv2.calcOpticalFlowPyrLK(image_precedente, frame_gray, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask  = cv2.line(mask,   (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)

        image_precedente = frame_gray.copy()
        p0       = good_new.reshape(-1, 1, 2)

    #lecture du clavier 100s pour sortir de la fenetre
    k = cv2.waitKey(100) & 0xff
    if k == 27:#27 represente la touche "echap" 
        break

    if current == imax:
        current = imin + 1
        mask    = numpy.zeros_like(image)
    else:
        current += 1

cv2.destroyAllWindows()