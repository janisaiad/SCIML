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

## 📊 Format des données

Les données d'entraînement sont structurées en triplets (mu, x, sol) où :
- `mu` représente les fonctions d'entrée (ex: conditions aux limites, conditions initiales)
- `x` représente les points dans le domaine spatial
- `sol` représente les solutions attendues aux points `x`

## 📁 Structure du projet

```
sciml/
├── data/              # Données d'entraînement et scripts de génération
├── model/             # Implémentations des architectures
│   ├── deeponet/      # Implémentation de DeepONet
│   └── fno/           # Implémentation de Fourier Neural Operator
├── notebooks/         # Notebooks Jupyter pour les expériences et visualisations
│   ├── deeponet/      # Notebooks pour DeepONet
│   └── fno/           # Notebooks pour FNO
├── utils/             # Fonctions utilitaires
├── tests/             # Tests unitaires et d'intégration
└── logs/              # Journaux d'entraînement et résultats
```

## 🚀 Installation et configuration

### Prérequis

- Python 3.9+
- TensorFlow 2.8+
- Environnement virtuel (recommandé)

### Étapes d'installation

1. Cloner le dépôt
   ```bash
   git clone https://github.com/username/sciml.git
   cd sciml
   ```

2. Configurer l'environnement et installer les dépendances
   ```bash
   chmod +x launch.sh
   ./launch.sh
   ```

3. Activation de l'environnement virtuel
   ```bash
   source .venv/bin/activate  # Pour Linux/MacOS
   # ou
   .\.venv\Scripts\activate   # Pour Windows
   ```

## 📚 Utilisation

### Génération de données

Les scripts de génération de données se trouvent dans le répertoire `data/generation/`. Exemple d'utilisation :

```bash
python data/generation/generate_big_heat_fno.py
```

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

### Notebooks d'exemples

Plusieurs notebooks sont disponibles pour démontrer l'utilisation des modèles :

- Pour DeepONet : `notebooks/deeponet/TORUNDEEPONETCOMPARISON.ipynb`
- Pour FNO : `notebooks/fno/TORUNFNO.ipynb`

### Conversion de notebooks avec Jupytext

Pour convertir tous les notebooks en fichiers Python :

```bash
find . -name "*.ipynb" -type f -exec jupytext --to py {} \;
```

## 🔍 Dépannage

Si vous rencontrez des problèmes avec JAX/JAXlib lors de l'entraînement des modèles FNO, essayez les commandes suivantes :

```bash
# Activation de l'environnement virtuel si ce n'est pas déjà fait
source .venv/bin/activate

# Désinstallation des versions actuelles
uv pip uninstall jax jaxlib

# Installation des versions compatibles
uv add jax
uv add jaxlib==0.4.17
```

## 📝 Notes de recherche et observations

### Performances du FNO
- trop de couches de fourier déstabilise l'entraînement
- considérer l'inférence temporelle séquentielle
- tester sur d'autres données (naca)

### améliorations deeponet
- comparer avec rb et pod
- tester architectures branch/trunk alternatives

### analyse spectrale
- impact nb couches fourier sur généralisation
- comparaison poids fourier entre couches  
- vérifier cast/coeff et phase multiplication

## 📄 Licence

Ce projet est distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
to push on neuraloperator library & some other opensource things, fftnd