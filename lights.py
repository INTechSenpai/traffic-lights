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

from numpy import median
#import picamera
#import picamera.array
from keras.models import load_models
from time import time

colors = {'J' : [247,180,0],
          'B' : [0,90,139],
          'O' : [218,113,52],
          'N' : [0,0,0],
          'V' : [97,153,58],
	  'G' : [180,176,160]} # gris du contour

patterns = [
        ['O','N','V'], ['V','N','O'],
        ['J','N','B'], ['B','N','J'],
        ['B','V','O'], ['O','V','B'],
        ['J','V','N'], ['N','V','J'],
        ['N','J','O'], ['O','J','N'],
        ['V','J','B'], ['B','J','V'],
        ['B','O','N'], ['N','O','B'],
        ['V','O','J'], ['J','O','V'],
        ['N','B','V'], ['V','B','N'],
        ['O','B','J'], ['J','B','O']]

def distance_couleur(c1, c2):
    return 2 * (c1[0] - c2[0]) ** 2 + 4 * (c1[1] - c2[1]) ** 2 + 3 * (c1[2] - c2[2]) ** 2

def distance_pattern(l, pattern):
    return sum([distance_couleur(l[i], colors[pattern[i]]) for i in range(3)])

def is_grey(l):
    minDistance = distance_couleur(l, colors['G'])
    for key in colors:
        dist = distance_couleur(l, colors[key])
        if dist < minDistance:
            return False
    return True
 
def most_probable_pattern(l):
    argmin = len(patterns)-1;
    minDistance = distance_pattern(l, patterns[argmin])
    for i in range(len(patterns)-1):
        dist = distance_pattern(l, patterns[i])
        if dist < minDistance:
            argmin = i
            minDistance = dist
    return patterns[argmin]

def closest_patterns():
    minVal = 10000000000
    for i in range(len(patterns)-1):
        for j in range(i+1, len(patterns)):
            if i != j and (i%2 != 0 or j != i+1): # l'ordre n'importe pas
                tmp = sum([distance_couleur(colors[patterns[i][k]], colors[patterns[j][k]]) for k in range(3)])
                if tmp < minVal:
                    minI = i
                    minJ = j
                    minVal = tmp

    print distance_couleur(colors['J'], colors['O']), 'entre Orange et Jaune'
    print minVal, 'entre', patterns[minI], 'et', patterns[minJ]

closest_patterns()

# Camera configuration
# TODO : potentiellement à modifier
w = 128

# TODO : rogner                                 
camera.zoom = (0.0, 0.0, 1.0, 1.0)
camera.resolution = (w, w)

print 'Neural network loading'

model = load_model('model.h5')

print 'Ready !'

# TODO attendre le signal

print 'Begin processing !'

debut = time()
with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as output:
        camera.capture(output, 'rgb')

print 'Image capture time :',time()-debut
img = output.array.reshape(1,3,w,w)

debut = time()
prediction = model.predict(img)

print 'Inference time :',time()-debut

x = prediction[0]
y = prediction[1]
width = prediction[2] / 3

couleurs = []

# pour chaque carré du pattern
for k in range(3):
    lr = []
    lg = []
    lb = []
    # on itère sur sa largeur
    for i in range(width):
        # et sur sa hauteur
        for j in range(width):
            # pour chaque couleur du pixel
            c = img[0][width * k + i][j]
            # en ignorant les pixels gris
            if not is_grey(c):
                lr.append(c[0]) 
                lg.append(c[1]) 
                lb.append(c[2])
    couleurs.append([median(lr), median(lg), median(lb)])

print most_probable_pattern(couleurs)
