# Projet DIAYN

## Description des fichiers : 

### Fichiers des mod�les :

* a2c : contient le code code du mod�le d'acteur critique utilis�,
* diayn :  contient le code du mod�le diayn,
* navigation2D : contient le code qui permet la navigation dans un espace � deux dimension,
* neural_network : contient un code permettant de g�n�rer un r�seau de neurone,
* showskills : contient un code qui permet de visualiser les comp�tences apprises par le mod�le diayn suivant les environnements.

### Fichiers de test :

* test_moutain_car : contient les tests qui entraine un mod�le diayn, puis r�cup�re les poids pour les utiliser dans un mod�le acteur critique afin d'am�liorer les performances. Les r�sultats sont stock�s dans une hi�rarchie de dossier qui suivant le stade de lancement est cr�e ou r�utilis�e. Les r�sultats sont compar� avec une baseline d'actor critic.
* test_parameters : contient les tests qui nous ont permis de choisir les param�tres gamma et alpha. 
* test_stabilisation : contient les tests qui ont permis de mesurer la sensibilit� du mod�le diayn � la seed al�atoire.


