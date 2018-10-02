import os
import cv2
import pickle

clasificador = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
reconocedor = cv2.face.LBPHFaceRecognizer_create()
reconocedor.read("train.yml")
labels = {"nombre": 1}

if __name__ == '__main__':
    with open("labels.pickle", "rb") as p:
        _labels = pickle.load(p)
        labels = {valor: clave for clave, valor in _labels.items()}

    for archivo in os.listdir("entrada/"):

        imagen = cv2.imread("entrada/" + archivo) 
        imagenGS = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        caras = clasificador.detectMultiScale(imagenGS, scaleFactor = 1.5, minNeighbors = 5)
        
        for (x, y, w, h) in caras:
            #Imprimir coordenadas en consola print(x,y,w,h)
            region = imagenGS[y:y+h, x:x+w]

            # Reconocer caras
            _id, certeza = reconocedor.predict(region)

            cv2.rectangle(imagen, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(imagen, labels[_id].capitalize(), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow(archivo, imagen)

    cv2.waitKey(0)
    cv2.destroyAllWindows()