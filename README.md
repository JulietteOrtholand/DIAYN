# Projet DIAYN

## Description des fichiers : 

### Fichiers des modèles :

* a2c : contient le code code du modèle d'acteur critique utilisé,
* diayn :  contient le code du modèle diayn,
* navigation2D : contient le code qui permet la navigation dans un espace à deux dimension,
* neural_network : contient un code permettant de générer un réseau de neurone,
* showskills : contient un code qui permet de visualiser les compétences apprises par le modèle diayn suivant les environnements.

### Fichiers de test :

* test_moutain_car : contient les tests qui entraine un modèle diayn, puis récupère les poids pour les utiliser dans un modèle acteur critique afin d'améliorer les performances. Les résultats sont stockés dans une hiérarchie de dossier qui suivant le stade de lancement est crée ou réutilisée. Les résultats sont comparé avec une baseline d'actor critic.
* test_parameters : contient les tests qui nous ont permis de choisir les paramètres gamma et alpha. 
* test_stabilisation : contient les tests qui ont permis de mesurer la sensibilité du modèle diayn à la seed aléatoire.


