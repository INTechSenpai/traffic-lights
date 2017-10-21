# -*- coding: utf-8 -*-
'''
Copyright (C) 2013-2017 Pierre-François Gimenez

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

import picamera
import picamera.array
from keras.models import load_models

colors = {'J' : [247,180,0],
          'B' : [0,90,139],
          'O' : [218,113,52],
          'N' : [0,0,0],
          'V' : [97,153,58]}

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

def most_probable_pattern(l):
    argmin = len(patterns)-1;
    minDistance = distance_pattern(l, patterns[argmin])
    for i in range(len(patterns)-1):
        dist = distance_pattern(l, patterns[i])
        if dist < minDistance:
            argmin = i
            minDistance = dist
    return patterns[argmin]

# Camera configuration
# TODO : potentiellement à modifier
w = 320
h = 240

# TODO : rogner                                 
camera.zoom = (0.0, 0.0, 1.0, 1.0)
camera.resolution = (w, h)

print 'Neural network loading'

model = load_model('model.h5')

with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as output:
        camera.capture(output, 'rgb')

img = output.array.reshape(1,w,h,3)
prediction = model.predict(img)


