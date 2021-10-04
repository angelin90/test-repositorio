
import numpy as np
import pickle

import cv2

import os


class Imatge():
    ROOT_DIR = os.path.abspath("")
    def __init__(self):
        self.__img_name = []
        self.__imgs_np_arrays = []
        self.__imgs_np_grays = []
        self.__bow = []

    def get_representacio(self):
        var=3
        var2=25
       print("El numero es: " + var)
       print("El numero es: " + var2)


        for i in self.__imgs_np_grays:
            self.__bow.append(self.compute_bow_images(i, vocabulary))
        return self.__bow


    def llegeix(self, nom_fitxer):
        print("hola mundo")
        print("maria")
        try:
            img = cv2.imread(nom_fitxer)
        except Exception as e:
            print(e)
        self.__imgs_np_arrays.append(img)
        self.__img_name.append(nom_fitxer)
        self.__imgs_np_grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    def calcula_distancia(self, imatge):
        return None

    def init_bow_img(self, file_vocabulary):
        '''
        creamos el objecto sift, FlannBasedMatcher
        Leemos el fichero de vocabulario vocabulary.dat
        Cramos la extracciónes BowImDreciptor
        Set del vocabulario
        Devolvemos el bow_extrator

        :param file_vocabulary:
        :return: bow_extrator
        '''
        sift = cv2.SIFT_create()
        matcher = cv2.FlannBasedMatcher()
        with open(file_vocabulary, 'rb') as file:
            vocabulary = pickle.load(file)
        bow_extrator = cv2.BOWImgDescriptorExtractor(sift, matcher)
        bow_extrator.setVocabulary(vocabulary)
        return bow_extrator

    def compute_bow_images(self, img, bow_extractor: cv2.BOWImgDescriptorExtractor):
        '''
        Creamos el objeto sift
        Detectamos los puntos de la imagen
        Si ha detectado se computa
        En caso contrario se llena de ceros

        :param img:
        :param bow_extractor:
        :return:
        '''
        sift = cv2.SIFT_create()
        keypoints = sift.detect(img)
        if (keypoints != []):
            return bow_extractor.compute(img, keypoints)
        else:
            return np.zeros((1, bow_extractor.descriptorSize()))

    def get_img_np(self):
        return self.__imgs_np_arrays

    def get_img_np_gray(self):
        return self.__imgs_np_grays


if __name__ == "__main__":
    img = Imatge()
    dir_imgs = os.path.join(img.ROOT_DIR, "train")
    img_list = [os.path.join(dir_imgs, x) for x in os.listdir(dir_imgs) if ".jpg" in x]

    img_horse = [x for x in img_list if "airplane" in x]
    img_airplanes = [x for x in img_list if "horse" in x]

    horse_hundred = img_horse[:1000]
    airplanes_hundred = img_airplanes[:1000]

    TOTAL_IMAGES = horse_hundred + airplanes_hundred
    for i in TOTAL_IMAGES:
        img.llegeix(i)

    vocabulary = img.init_bow_img("vocabulary.dat")

    representacio = img.get_representacio()
    print(representacio)

    ##visualizar las imagenes
    for i in img.get_img_np_gray():
        cv2.imshow("test", i)
        cv2.waitKey(0)



