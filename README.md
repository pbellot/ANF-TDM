# Atelier "Apprentissage automatique pour la classification textuelle"
## ANF CNRS "Exploration documentaire" 2021

## Présentation des 2 carnets proposés

PARTIE CLASSIFICATION THEMATIQUE - Le carnet ClassificationMetaISTEX.ipynb
  - Il contient le code Python pour d'une part mettre en forme les données issues de la plateforme ISTEX (CorpusCovid.csv) mais aussi pour réaliser différentes classification non supervisée (partitionnement) avec les méthodes des k-moyennes (kMeans) et construire des cartes auto-organisées (Self Organized Maps SOM). 
  - Les modules Python utilisés sont Pandas (manipulation des données), SciKit Learn (kMeans), somoclu (cartes auto-organisées). 
  - Le carnet ClassificationMetaISTEX produit le fichier CorpusWekaResumes.csv qui doit être utilisé dans Weka. Ce fichier est également disponible ici si vous voulez sauter l'étape de mise en forme et directement utiliser Weka uqe nous utilisons pour apprendre des modèles de classification (approche bayésienne et arbres de décision) en catégories scientifiques. Il est bien sûr possible de construire des modèles équivalents en restant dans l'environnement Python. 
  - installer l'environnement Weka : https://www.cs.waikato.ac.nz/~ml/weka/

PARTIE ANALYSE DE SENTIMENTS - Le carnet IMDB.ipynb
  - Il contient le code Python nécessaire pour effectuer une analyse de sentiment de critiques de films de type "classification binaire de la polarité de la critique". Appliquée à un corpus de 50 000 critiques en anglais de films issues de la plateforme IMDB, la tâche consiste à apprendre un modèle numérique qui détermine automatiquement la polarité de la critique : polarité négative versus polarité positive. L'approche n'utilise que très peu de ressources linguistiques puisque seule une liste de "mots outils" est exploitée. Il est ainsi très facile de reconstruire un modèle similaire pour le français. 
  - Les méthodes appliquées sont la classification bayésienne naïve (modèles dits "sacs de mots" où les mots sont considérés indépendamment les uns des autres) puis différentes réseaux de neurones profonds (deep learning). Il est intéressant de relever l'écart de performance entre les solutions et de le mettre en regard avec les temps de calcul nécessaires à l'apprentissage des modèles. On voit dans le code à quel point les architectures neuronales sont configurables.
  - Le code s'appuie sur les modules Python Pandas (manipulation des données), SciKit Learn (classification bayésienne), Keras/Tensorflow (réseaux de neurones).
  - Vous devez télécharger le fichier https://ai.stanford.edu/~amaas/data/sentiment/ qui contient les critiques de films utilisées pour l'analyse de sentiment. Il s'agit d'un corpus bien connu, autour duquel beaucoup d'expérimentations sont réalisées (voir par exemple https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/code). Ces données sont décrites dans Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011) - https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf 
  
  NB : une version en un seul fichier .csv du corpus IMDB est ici : https://drive.google.com/file/d/1LnfB59FpNkmX3xVXAL3oc4YyeAIiXod5/view?usp=sharing 

## Commentaires

Ces ateliers de TDM (https://anf-tdm-2020.sciencesconf.org/329939 et https://anf-tdm-2021.sciencesconf.org) permettent d'explorer des tâches de classification automatique non supervisée (k-moyennes et cartes auto-organisées) et supervisée (classification bayésienne, arbres de décision, réseaux de neurones profonds et plongements lexicaux). Les données sont d'une part les méta-données de documents issus d'ISTEX concernant le mot clé "covid" (catégorisation en domaines à partir des résumés puis partitionnement) et une collection de critiques IMDB de films pour l'analyse de sentiment.
Les environnements sont Jupyter Notebook et Weka. Le langage est Python.

Pour réaliser les exemples de l'atelier, vous devez : 

(le plus simple) Pour une exécution du code Python dans l'environnement distant Google Colab (http://colab.research.google.com) : 
  - disposer d'un compte Google, accéder à votre Google Drive http://drive.google.com/
  - ouvrir les deux Notebooks (.ipynb) disponibles ici dans l'environnement Google Colab : cliquer sur le nom du carnet puis sur l'icône Colab présent en haut de la zone ouverte. NB : cela créera un dossier ColabNotebooks dans votre GoogleDrive si vous n'en avez pas déjà.
- déposer dans votre Google Drive / ColabNotebooks les fichiers CorpusCovid.csv présent ici pour la partie "classification thématique" ainsi que les critiques de films IMDB (voir plus bas) pour la partie "analyse de sentiments". 

L'alternative à l'utilisation de Google Colab, plus complexe car elle nécessite de nombreuses installations, consiste à utiliser Python 3 sur votre propre ordinateur. Pour cela je vous conseille fortement d'utiliser Jupyter Notebook (https://jupyter.org) via un gestionnaire d'environnements (par ex. https://anaconda.org) ou un IDE tel que PyCharm (https://www.jetbrains.com/fr-fr/pycharm/). Un environnement Linux avec carte GPU NVIDIA est le plus adapté mais MacOS ou Windows sont possibles d'autant que l'utilisation d'un GPU n'est pas obligatoire. Le code des carnets a été testé avec un environnement Python 3.7, Tensorflow 2.0.0, Jupyter 6.x, Keras 2.3.1, Pandas 1.2.2, SciKit 0.24.1. Le plus critique est la version de Tensorflow qui n'est pas celle proposée par défaut par le gestionnaire d'environnement Anaconda avec Python 3.7 (problèmes d'incompatibilités avec les versions 2.2 et 2.4 et certaines fonctionnalités absentes avec les versions antérieures). Pour forcer l'installation de la version 2.0.0, utiliser la commande conda install tensorflow=2.0.0 dans le Terminal. Comme d'habitude il est fortement conseillé de travailler dans un environnement Python dédié au projet afin de ne pas modifier votre système de base.

Note pour MacOS : pour utiliser les accélérations GPU et si vous disposez d'une carte graphique AMD Radeon ou Intel intégrée, vous devez utiliser le module PlaidML (https://plaidml.readthedocs.io, voir aussi https://towardsdatascience.com/deep-learning-using-gpu-on-your-macbook-c9becba7c43) mais cela n'est absolument pas nécessaire pour expérimenter l'atelier, le CPU étant tout à fait suffisant pour les données manipulées (voir https://towardsdatascience.com/macbook-pro-for-deep-learning-lets-try-841ab8ffee7e). Par ailleurs, de nouvelles solutions se mettent en place pour se passer de PlaidML avec des pré-realease de Tensorflow 2.4 (https://machinelearning.apple.com/updates/ml-compute-training-on-mac). 

ps : les diapositives qui reprennent les exemples sont ici : https://drive.google.com/file/d/1iDkoNf5Oc6emFU1Jm78Bb4A7YP7HcVDm/view?usp=sharing

Pour toute question : patrice.bellot@univ-amu.fr
