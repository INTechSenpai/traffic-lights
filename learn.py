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

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# TODO
(X_train, y_train), (X_test, y_test) = None

# nouveau réseau de neurones
model = Sequential()

# l'extraction de feature se fait avec Conv2D -> augmentation des dimensions
# MaxPooling permet de réduire les dimensions
# Toujours utiliser une activation "relu"
model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(1,28,28)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten : applatit en une seule dimension
model.add(Flatten())

# Couche dense
model.add(Dense(128, activation='relu'))

# Permet d'éviter l'overfitting
model.add(Dropout(0.5))

# pas d'activation car c'est l'output
# 3 provient du fait qu'on attend trois valeurs (x, y, width)
model.add(Dense(3))

# compilation du modèle + paramètres d'évaluation et d'apprentissage
model.compile(loss='mean_squared_error', optimizer='adam')

# l'apprentissage des poids
model.fit(X_train, Y_train, 
          batch_size=32, epochs=10, verbose=1)

# sauvegarde du modèle appris
model.save('model.h5')
