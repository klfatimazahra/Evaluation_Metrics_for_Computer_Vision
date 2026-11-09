# Évaluation des métriques en vision par ordinateur

## Présentation

Ce notebook compare les méthodes d’évaluation de quatre tâches de vision par ordinateur :

1. **Classification d’images** : attribuer une seule classe à une image entière.
2. **Détection d’objets** : reconnaître les objets et localiser chacun avec une boîte englobante.
3. **Segmentation sémantique** : attribuer une classe à chaque pixel, sans distinguer les objets individuels d’une même classe.
4. **Segmentation par instance** : attribuer un masque séparé à chaque objet, même lorsque plusieurs objets appartiennent à la même classe.

Le notebook utilise deux jeux de données et deux modèles : ResNet-18 pour la classification sur CIFAR-10, puis Mask R-CNN pour la détection et les deux formes de segmentation sur Penn-Fudan Pedestrian.

## Jeux de données

### CIFAR-10 — classification

CIFAR-10 contient **60 000 images couleur de 32 × 32 pixels**, réparties en dix classes : avion, automobile, oiseau, chat, cerf, chien, grenouille, cheval, bateau et camion.

Le découpage utilisé dans le notebook est :

| Partition | Nombre d’images | Utilisation |
|---|---:|---|
| Entraînement | 45 000 | Apprendre les paramètres du modèle |
| Validation | 5 000 | Suivre l’apprentissage et choisir les meilleurs poids |
| Test | 10 000 | Évaluer le modèle final, sans l’utiliser pendant l’entraînement |
| **Total** | **60 000** | |

La validation est prélevée de manière stratifiée dans les 50 000 images officielles d’entraînement. Le jeu de test officiel reste séparé. La graine aléatoire utilisée est `42`.

### Penn-Fudan Pedestrian — détection et segmentation

Penn-Fudan contient **170 images de rue** avec des masques identifiant les piétons. Les boîtes englobantes sont calculées directement à partir de ces masques.

Le notebook évalue un échantillon reproductible de **30 images**. Dans l’exécution enregistrée, cet échantillon contient **84 instances annotées**. Il n’y a pas de séparation entraînement/validation/test pour Penn-Fudan : le modèle Mask R-CNN est utilisé tel quel, sans réentraînement sur ce jeu de données.

## Modèles utilisés

### Classification : ResNet-18 adapté à CIFAR-10

- Réseau convolutif ResNet-18 initialisé sans poids pré-entraînés (`weights=None`) et entraîné sur CIFAR-10.
- Première convolution adaptée aux petites images 32 × 32 : noyau 3 × 3 et stride 1 ; max-pooling initial supprimé.
- Couche de sortie de dix classes et dropout de 0,25.
- Augmentation des données pendant l’entraînement : recadrage aléatoire et retournement horizontal.
- Régularisation : `weight_decay = 0.0005` et `label_smoothing = 0.05`.
- Optimiseur SGD avec momentum et décroissance du taux d’apprentissage.
- Jusqu’à 30 époques ; meilleure époque enregistrée : **30**.

### Détection et segmentation : Mask R-CNN ResNet-50 FPN

- Modèle TorchVision `maskrcnn_resnet50_fpn` avec les poids **COCO_V1**.
- Le modèle est pré-entraîné sur COCO et utilisé pour détecter la classe « personne » dans Penn-Fudan.
- Seules les prédictions dont le score est supérieur ou égal à **0,05** sont conservées.
- Dans l’exécution enregistrée, 363 prédictions ont été conservées pour 84 instances annotées.

## Métriques calculées

| Tâche | Métriques principales |
|---|---|
| Classification | Accuracy, balanced accuracy, précision, rappel et F1-score (macro et pondérés), matrice de confusion, Top-1, Top-5 et ROC-AUC macro (one-vs-rest) |
| Détection | IoU des boîtes, AP50, AP75 et mAP moyenne pour les seuils d’IoU de 0,50 à 0,95 |
| Segmentation sémantique | Pixel accuracy, IoU par classe, mIoU, précision, rappel et Dice pour la classe piéton |
| Segmentation par instance | IoU des masques, Mask AP50, Mask AP75 et Mask mAP de 0,50 à 0,95 |

**Rappel :** l’IoU mesure le recouvrement entre une prédiction et la vérité terrain. L’AP résume la courbe précision-rappel à un seuil d’IoU donné ; la mAP moyenne les AP sur plusieurs seuils.

## Résultats de l’exécution enregistrée

Les scores ci-dessous proviennent des sorties sauvegardées dans le notebook. Ils peuvent varier lors d’une nouvelle exécution selon l’environnement, les versions logicielles et l’échantillon évalué.

| Tâche | Métrique | Score |
|---|---|---:|
| Classification CIFAR-10 | Accuracy | **93,15 %** |
| Classification CIFAR-10 | F1 macro | **93,15 %** |
| Classification CIFAR-10 | Top-5 accuracy | **99,73 %** |
| Détection Penn-Fudan | AP50 | **99,25 %** |
| Détection Penn-Fudan | AP75 | **92,84 %** |
| Détection Penn-Fudan | mAP@[0,50:0,95] | **82,29 %** |
| Segmentation sémantique | Pixel accuracy | **94,82 %** |
| Segmentation sémantique | IoU piéton | **78,04 %** |
| Segmentation sémantique | mIoU | **85,85 %** |
| Segmentation sémantique | Dice piéton | **87,67 %** |
| Segmentation par instance | Mask AP50 | **99,25 %** |
| Segmentation par instance | Mask AP75 | **89,46 %** |
| Segmentation par instance | Mask mAP@[0,50:0,95] | **73,76 %** |

Le notebook affiche également des exemples d’images, les matrices de confusion, les courbes d’apprentissage, les courbes précision-rappel, les boîtes prédites et les masques réels/prédits.

## Installation et exécution

1. Ouvrir le fichier `evaluation_metriques_vision_ordinateur_FR_CNN_progression.ipynb` dans Jupyter Notebook, JupyterLab ou Google Colab.
2. Installer PyTorch et TorchVision dans des versions compatibles. Le notebook vérifie que TorchVision est en version 0.16 ou supérieure.
3. Installer, si nécessaire, les bibliothèques complémentaires :

   ```bash
   pip install scikit-learn matplotlib pandas pillow tqdm packaging
   ```

4. Dans Google Colab, activer un accélérateur GPU pour accélérer l’entraînement de ResNet-18.
5. Exécuter les cellules dans l’ordre, du début à la fin. Les jeux de données et les poids pré-entraînés sont téléchargés automatiquement lors de la première exécution.

Les données sont placées dans le dossier `data_vision/`. Les tableaux et graphiques sont affichés directement dans le notebook.

## Organisation du notebook

| Cellules | Contenu |
|---|---|
| 0–2 | Présentation, dépendances et configuration |
| 3–15 | Classification CIFAR-10, entraînement de ResNet-18 et métriques |
| 16–30 | Chargement de Penn-Fudan, Mask R-CNN et métriques de détection |
| 31–35 | Segmentation sémantique et métriques pixel |
| 36–39 | Segmentation par instance et métriques de masque |
| 40–42 | Comparaison des métriques, tableau récapitulatif et configuration finale |

## Limites à connaître

- Les résultats Penn-Fudan portent sur **30 images sur les 170 disponibles** ; ils ne représentent pas une évaluation exhaustive du jeu de données.
- Mask R-CNN est pré-entraîné sur COCO et n’est pas ajusté sur Penn-Fudan dans ce notebook. Cette expérience mesure donc sa généralisation à ces images.
- Le calcul AP/mAP est une implémentation pédagogique inspirée du protocole COCO. Comme indiqué dans le notebook, elle ne reproduit pas tous les détails de l’évaluateur officiel `pycocotools`. Les scores Penn-Fudan ne doivent donc pas être présentés comme des scores officiels de benchmark COCO.

## Références

- [CIFAR-10 — site officiel](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Penn-Fudan Pedestrian — page du jeu de données](https://www.cis.upenn.edu/~jshi/ped_html/)
- [ResNet-18 — documentation TorchVision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [Mask R-CNN ResNet-50 FPN — documentation TorchVision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html)
