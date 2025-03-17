# SciML - Scientific Machine Learning

<div align="center">
  
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange)

</div>

## 📝 Description

Ce projet implémente des architectures de réseaux de neurones modernes pour la modélisation d'opérateurs et la résolution d'équations aux dérivées partielles (EDPs). Il est développé dans le cadre d'une collaboration avec les Professeurs Hadrien Montanelli et Samuel Kokh.

## 🎯 Objectifs

- Implémenter et comparer deux architectures de deep learning pour la modélisation d'opérateurs :
  - **DeepONet** (Deep Operator Network) - Pour apprendre des opérateurs entre espaces de fonctions
  - **FNO** (Fourier Neural Operator) - Pour exploiter l'analyse spectrale dans la modélisation d'opérateurs

- Appliquer ces architectures à la résolution d'équations aux dérivées partielles (EDPs), avec un focus particulier sur l'équation de la chaleur

## 🧠 Concepts clés

- **Modélisation d'opérateurs** : Apprendre à cartographier des espaces d'entrée fonctionnelle vers des espaces de sortie fonctionnelle
- **Analyse spectrale** : Utilisation de transformées de Fourier pour capturer efficacement les dynamiques spatiales
- **Apprentissage supervisé** : Entraînement sur des paires entrée-sortie générées par des solveurs numériques classiques

## 📁 Structure du projet

```
sciml/
├── data/              # Données d'entraînement et scripts de génération
├── model/             # Implémentations des architectures
│   ├── deeponet/      # Implémentation de DeepONet
│   └── fno/           # Implémentation de Fourier Neural Operator
├── notebooks/         # Notebooks Jupyter pour les expériences et visualisations
├── utils/             # Fonctions utilitaires
├── tests/             # Tests unitaires et d'intégration
└── logs/              # Journaux d'entraînement et résultats
```

## 📊 Format des données

Les données d'entraînement sont structurées en triplets (mu, x, sol) où :
- `mu` représente les fonctions d'entrée (ex: conditions aux limites, conditions initiales)
- `x` représente les points dans le domaine spatial
- `sol` représente les solutions attendues aux points `x`

## 🚀 Utilisation

### Installation

```bash

git clone https://github.com/username/sciml.git
cd sciml

chmod+X launch.sh


```


## Si bug


(SCIML) (base) jovyan@jupyter-janis-2eaiad-40polytechnique-2eedu:~/persistent/SCIML$ history
    1  ls
    2  cd persistent/
    3  ls
    4  cd SCIML/
    5  LS
    6  ls
    7  git checkout janis
    8  ./launch.sh
    9  git branch idcs
   10  git branch
   11  git checkout idcs
   12  git status
   13  git add .
   14  git status
   15  git commit -m "idcs setup"
   16  git config --global user.email "janis.aiad@polytechnique.edu"
   17  git commit -m "idcs setup"
   18  git push
   19  git push origin idsc
   20  git push origin idcs
   21  ls
   22  pwd
   23  /home/jovyan/persistent/SCIML/.venv/bin/python /home/jovyan/persistent/SCIML/data/generation/generate_heat_fno_big.py
   24  /home/jovyan/persistent/SCIML/.venv/bin/python /home/jovyan/persistent/SCIML/data/generation/generate_big_heat_fno.py
   25  /home/jovyan/persistent/SCIML/.venv/bin/python /home/jovyan/persistent/SCIML/notebooks/training_fno_heat_big.py
   26  uv remove jax
   27  uv pip uninstall jax
   28  uv add jax
   29  /home/jovyan/persistent/SCIML/.venv/bin/python /home/jovyan/persistent/SCIML/notebooks/training_fno_heat_big.py
   30  uv pip uninstall jaxlib
   31  source .venv/bin/activate
   32  uv remove jaxlib
   33  uv pip uninstall jaxlib
   34  uv add jaxlib
   35  /home/jovyan/persistent/SCIML/.venv/bin/python /home/jovyan/persistent/SCIML/notebooks/training_fno_heat_big.py
   36  uv remove jax
   37  uv pip uninstall jax
   38  uv add jax
   39  uv pip list
   40  uv remove jaxlib
   41  uv add jaxlib==0.4.17
   42  /home/jovyan/persistent/SCIML/.venv/bin/python /home/jovyan/persistent/SCIML/notebooks/training_fno_heat_big.py
   43  history
(SCIML) (base) jovyan@jupyter-janis-2eaiad-40polytechnique-2eedu:~/persistent/SCIML$ 

### Entraînement des modèles

```python
from sciml.model.fno import FNO
from sciml.model.deeponet import DeepONet

# Configuration et entraînement de FNO
fno_model = FNO(hyper_params, regular_params, fourier_params)
fno_model.fit()

# Configuration et entraînement de DeepONet
deeponet_model = DeepONet(hyper_params, regular_params)
deeponet_model.fit()
```

## 📄 Licence

Ce projet est distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

# Jupytext 
find . -name "*.ipynb" -type f -exec jupytext --to py {} \;


# Observation

Many fourier layers lead to a very unstable and low training
Try to infer time after time with a pipeline
Use a NACA dataset
add an rb and pod comparison for deeponet
for fno heat, we can try to infer on a completely different solutions to see
dependency to number of fourier layers
for neuraloperator, don't forget torch_harmonics and wandb
we can compare the fourier weights for every layers
make a computationnal 
faire attention au casting et aux coefficients, et à la phase en multipliiant chaque coefficient, donc on fait bien ce qu'il faut