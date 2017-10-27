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
from PIL import Image
import cPickle as pickle
import numpy as np

train_set_x = []
train_set_y = []

nbImageTrainingSet = 1
nbImageTestSet = 0
nbGenerated = 1
nbTotalTrainingSet = nbGenerated * nbImageTrainingSet
nbTotalTestSet = nbGenerated * nbImageTestSet

def generate(a,b,c):
    return Image.open(a).resize((32,32),Image.ANTIALIAS).convert("RGB"),b,c

def compute(rng, nbGenerated, lines):
    set_x = []
    set_y = []
    for i in rng:
        print "Contenu:",lines[i]
        x_target, y_target = literal_eval(lines[i])
        for j in range(nbGenerated):
            img, x, y = generate("data/pic"+str(i)+".jpg", x_target, y_target)
            set_x.append(np.array(img)[...,:3]) # on retire le canal alpha
            set_y.append((x,y))
    return set_x, set_y

file_target = open("data/target.txt", "r")
lines = file_target.readlines()
file_target.close()

train_set = compute(range(nbImageTrainingSet), nbGenerated, lines)

print train_set

train_set_x = train_set[0].reshape(nbTotalTrainSet, 3, 32, 32)
train_set_y = train_set[1].reshape(nbTotalTrainSet, 2)

test_set = compute(range(nbImageTrainingSet), nbGenerated, lines)
test_set_x = train_set[0].reshape(nbTotalTrainSet, 3, 32, 32)
test_set_y = train_set[1].reshape(nbTotalTrainSet, 2)

pickle.dump((train_set_x, train_set_y, test_set_x, test_set_y), open("dataset.dat","wb"), 2)
