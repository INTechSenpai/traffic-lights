# traffic-lights (WIP)

Réseau de neurones pour la lecture du pattern de couleur pour la Coupe de France de Robotique (Eurobot) 2018.

## Fonctionnement

### Prétraitement

- rognage de la majorité de l'image
- espace de couleur : conversion en HSV ?

### Réseau de neurones convolutif

- renvoie la position du coin inférieur gauche du pattern ainsi que ses dimensions

### Traitement classique

- découpage du pattern en trois carrés
- renvoie le pattern le plus proche
- distance entre deux couleurs : 2×ΔR² + 4×ΔG² + 3×ΔB²

## Autre fonctionnalité

- stream de la caméra (accessible depuis un smartphone)

## Matériel

- Raspberry Pi Zero W
- Caméra raspberry pi v2
