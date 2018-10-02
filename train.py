import os
import cv2
from PIL import Image
import numpy as np
import pickle

id = 0
labelDiccionario = {}
labels = [] #Pasar los nombres a numeros
train = [] #pasar la imagenes a array de pixeles
clasificador = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
reconocedor = cv2.face.LBPHFaceRecognizer_create()

if __name__ == '__main__':
    rutaBase = os.path.dirname(os.path.abspath(__file__))
    rutaImagenes = os.path.join(rutaBase, 'imagenes')

    for raiz, carpetas, archivos in os.walk(rutaImagenes):
        for archivo in archivos:
            if archivo.endswith("png") or archivo.endswith("jpg"):
                rutaArchivo = os.path.join(raiz, archivo)
                label = os.path.basename(raiz)
                
                if not label in labelDiccionario:
                    labelDiccionario[label] = id
                    id += 1

                _id = labelDiccionario[label]
                print(labelDiccionario)

                imagenGS = Image.open(rutaArchivo).convert("L") #Imagen en escala de grises
                arrayImagen = np.array(imagenGS, "uint8")
                caras = clasificador.detectMultiScale(arrayImagen, scaleFactor=1.5, minNeighbors=5)
                
                for (x, y, w, h) in caras:
                    regionCara = arrayImagen[y:y+h, x:x+w]
                    train.append(regionCara)
                    labels.append(_id)

    with open("labels.pickle", "wb") as p:
        pickle.dump(labelDiccionario, p)

    reconocedor.train(train, np.array(labels))
    reconocedor.save("train.yml")