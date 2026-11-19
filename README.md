# Activité pratique — Bases de PyTorch

Notebook d'introduction à PyTorch, puis exercices de régression et de classification.

## Contenu

Le notebook `Activité pratique Bases de Pytorch.ipynb` suit le cycle d'apprentissage :

**données → tenseurs → modèle → prédictions → loss → backward → optimizer**

La première partie couvre les briques du framework :

- tenseurs et déplacement CPU / GPU ;
- dérivation automatique avec Autograd ;
- réseaux `nn.Module` et fonctions d'activation ;
- fonctions de coût et optimiseurs ;
- boucle d'entraînement, `train()` / `eval()` ;
- `Dataset` et `DataLoader` ;
- sauvegarde d'un modèle avec `state_dict`.

Les exercices appliquent cette mécanique :

1. **Régression** — prédire un salaire (DH) à partir de l'ancienneté et du niveau d'expertise.
2. **Iris** — classer une fleur parmi setosa, versicolor et virginica.
3. **MNIST (MLP)** — reconnaître des chiffres manuscrits avec des mini-lots, puis mesurer accuracy, rappel, précision et F1.
4. **MNIST (CNN)** — même tâche avec deux convolutions, selon l'architecture 16 puis 32 filtres.

## Installation

Python 3.10 ou plus récent.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Sous macOS ou Linux, l'activation du venv est `source .venv/bin/activate`.

La cellule qui dessine le graphe de calcul utilise [Graphviz](https://graphviz.org/). Sans ce programme, le reste du notebook s'exécute quand même.

Le premier lancement des exercices MNIST télécharge les images dans le dossier `data/` (ignoré par Git).

## Exécution

Ouvrir le notebook dans JupyterLab, VS Code ou Cursor, puis lancer toutes les cellules dans l'ordre.

```bash
jupyter notebook "Activité pratique Bases de Pytorch.ipynb"
```

Les graines aléatoires sont fixées (`manual_seed`) pour que les résultats restent comparables d'une exécution à l'autre. Les accuracy exactes peuvent varier légèrement selon le matériel.

## Résultats attendus

- Régression salariale : les coefficients retrouvés sont proches de 600 DH par année, 1 500 DH par point d'expertise et 4 000 DH de salaire de base. La prédiction pour 7 ans et une expertise de 4/5 est proche de 14 200 DH.
- Iris : accuracy de test élevée sur 30 fleurs. La fleur `[5.1, 3.5, 1.4, 0.2]` est une setosa.
- MNIST : le perceptron multicouche dépasse en général 95 % d'accuracy, le CNN 97 % après quelques époques.
