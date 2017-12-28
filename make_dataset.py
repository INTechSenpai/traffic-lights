# -*- coding: utf-8 -*-
'''
Copyright (C) 2017 Pierre-François Gimenez

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>
'''

from ast import literal_eval
from PIL import Image, ImageEnhance
from math import sin, cos, pi
import cPickle as pickle
import numpy as np
import random
import os.path

train_set_x = []
train_set_y = []

nbTotalImages = 236
nbImageTrainingSet = 200
nbImageTestSet = nbTotalImages - nbImageTrainingSet
nbGenerated = 100
#nbTotalTrainingSet = nbGenerated * nbImageTrainingSet
#nbTotalTestSet = nbGenerated * nbImageTestSet
#final_size = 64
final_size = 128

def generate(img_name,original_target_x,original_target_y):
    
    # les paramètres de la génération
#    crop_size = 512
    crop_size = 1024
    delta_x = random.randint(-crop_size/3,crop_size/3)
    delta_y = random.randint(-crop_size/3,crop_size/3)
    rotation_angle = 0
    delta_contrast = random.uniform(-.2,.2)
    delta_brightness = random.uniform(-.2,.2)
    delta_color = random.uniform(-.2,.2)
    img = Image.open(img_name)

    # quelques paramètres fixés
    rotation_center_x = original_target_x
    rotation_center_y = original_target_y
    crop_topleft_x = rotation_center_x + delta_x - crop_size / 2
    crop_topleft_y = rotation_center_y + delta_y - crop_size / 2

    enhancerContrast = ImageEnhance.Contrast(img)
    img = enhancerContrast.enhance(1 + delta_contrast)
    enhancerB = ImageEnhance.Brightness(img)
    img = enhancerB.enhance(1 + delta_brightness)
    enhancerColor = ImageEnhance.Color(img)
    img = enhancerColor.enhance(1 + delta_color)
    img = img.rotate(rotation_angle).crop((crop_topleft_x, crop_topleft_y, crop_topleft_x + crop_size, crop_topleft_y + crop_size)) # param : x_haut_gauche, y_haut_gauche, x_bas_droite, y_bas_droite

    # calcul de la rotation
    angle_rad = -rotation_angle * pi / 180 # négatif car la base est dans le sens inverse
    target_x = (original_target_x - rotation_center_x) * cos(angle_rad) - (original_target_y - rotation_center_y) * sin(angle_rad)
    target_y = (original_target_x - rotation_center_x) * sin(angle_rad) + (original_target_y - rotation_center_y) * cos(angle_rad)

    # calcul du crop
    target_x = int(round(target_x - delta_x + crop_size / 2))
    target_y = int(round(target_y - delta_y + crop_size / 2))

    # calcul du resize
    target_x = int(round(target_x * final_size / crop_size))
    target_y = int(round(target_y * final_size / crop_size))

    img = img.resize((final_size,final_size), Image.ANTIALIAS)
    img.save("img_cible.jpg")
#    print target_x,target_y
    if target_x < 0 or target_y < 0:
        print "Erreur !",target_x,target_y
    return img.convert("RGB"), target_x, target_y


def compute(rng, nbGenerated, lines):
    set_x = []
    set_y = []
    for i in rng:
	if os.path.isfile("data/img/"+str(i+1)+".jpg"):
            print "Ligne ",i,": ",lines[i]
            x_target, y_target = literal_eval(lines[i])
            for j in range(nbGenerated):
                img, x, y = generate("data/img/"+str(i+1)+".jpg", x_target, y_target)
                set_x.append(np.array(img)[...,:3]) # on retire le canal alpha
                set_y.append(np.array((x,y)))
        else:
	    print "No file : ","data/img/"+str(i+1)+".jpg"
    return np.asarray(set_x), np.asarray(set_y)

file_target = open("data/target.txt", "r")
lines = file_target.readlines()
file_target.close()

index = range(0, nbTotalImages)
random.shuffle(index)

print index

print "Training set"
train_set = compute(index[0:nbImageTrainingSet], nbGenerated, lines)
nbTotalTrainingSet = len(train_set[0])
train_set_x = train_set[0].reshape(nbTotalTrainingSet, 3, final_size, final_size)
train_set_y = train_set[1].reshape(nbTotalTrainingSet, 2)

print "Test set"

test_set = compute(index[nbImageTrainingSet : nbTotalImages], nbGenerated, lines)
nbTotalTestSet = len(test_set[0])
test_set_x = test_set[0].reshape(nbTotalTestSet, 3, final_size, final_size)
test_set_y = test_set[1].reshape(nbTotalTestSet, 2)

pickle.dump((train_set_x, train_set_y, test_set_x, test_set_y), open("dataset-128-big2-norot.dat","wb"), 2)
