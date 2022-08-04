

![Streamlit](https://streamlit.io/images/brand/streamlit-logo-primary-colormark-lighttext.png)

# Streamlit tutoriel
|----------------------------------------------------------------------------------------------------------------------|

Bonjour :wave:, et bienvenue dans ce projet d'affichage streamlit.

Il s'agit ici d"une description des fichiers de ce projet dans le but d'une meilleure compréhension 
de leur utilisation.

Ce travail intervient dans un contexte précis et une problématique particulière.

En effet, il s'agit d'une société financière, **Prêt à dépenser** qui propose des crédits à la consommation pour des
personnes ayant peu ou pas du tout d'historique de prêt. Le jeu de données mise à notre disposition pour ce projet est
disponible sur Kaggle (Voir la source des données au bas de page).

## Notre mission 

- Mise en oeuvre d'un outil de **"scoring crédit"** pour calculer la probabilité de défaut de paiement de chaque client
   (Développement d'un modèle de classification);
- S'assurer de comprendre le modèle par une interprétabilité de ce dernier afin de garder une 
  transparence sur les prises de décision d'octroi de crédit;
- Développement d'un dashboard interactif utile pour les chargés de clients et aussi pour les clients.


## Pour ce répertoire
Vous trouverez ici la description des fichiers contenus dans le dossier **fichier_api** et leurs différentes utilités.

| Nom fichier                     | Description                                                                                                                                                                                                                                                                                                           |
|---------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| joblib_lgbm_beta_3_Model.pkl    | C'est le modèle prédctif conçu en interne pour prédire les défauts de paiement obtenu après sauvegarde via la librairie joblib                                                                                                                                                                                        |
| logo_pret_a_depenser.png        | il s'agit d'un logo d'entreprise mettre en première page de dashboard streamlit                                                                                                                                                                                                                                       |
| fichier-test1000-api.csv        | C'est un fichier test de 1000 individus issu de données initiales de jeu de test ayant subi l'étape de preprocessing. il contient les variables indépendantes.                                                                                                                                                        |
| data_predict.csv                | On construit ce fichier pour simplifier le problème et réduire le temps d'affichage de l'application de dashboard                                                                                                                                                                                                     |
| data_tr_api                     | Fichier contenant les variables indépendantes issu du jeu d'entrainement du modèle, contient également 1000 instances pour réduire le temps de calcul de l'explicateur TreeExplainer utilisé pour définir les valeurs de Shap. Le renommage du fichier avec ajout de **.csv** en fin de nom conduit à un fichier csv. |

## Installation de Streamlit
Dans votre notebook ou un éditeur de code comme **PyCharm Community Edition** , vous pouvez installer streamlit avec la commmande **pip install streamlit**
 

## Rôle implicite du code my_streamlit_dashboard_code.py
Ce code permet de visualiser un dashboard en local comme sur le web. Il s'agit d'un dashboard interactif qui est capable 
d'afficher:

1. le score de défaut de paiement d'un client dont l'identifiant est connu;
2. les contributions de chaque feature dans la prise de décision local du modèle;
3. les contributions de chaque feature dans la prise de décision globale du modèle;
4. les distributions statistiques de caractéristiques  tout affichant informations descriptives relatives au client 
5. considéré à l’ensemble des clients ou à un groupe de clients similaires;
6. l'impact de la modification des valeurs d'une feature sur l'orientation du modèle;
7. Analyse bivariées grâce à des graphiques de nuage de points.

Ces opérations sont associées à des fonctions définies

## Exécution Streamlit

Avec PyCharm par exemple, vous pouvez l'exécuter en local en tapant la commande : 
- ```code
  streamlit run my_streamlit_dashboard_code.py
  ```
dans votre terminal puis l'onglet  **Command Prompt**


## Déploiement sur Heroku
Pour un déploiement efficace sur le cloud Heroku, d'autres fichiers sont réquis en plus du code d'application .py:

| Nom fichier      | Description                                                                                                                                                                                                                          |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| requirements.txt | Fichier contenant toutes les librairies utile. au bon fonctionnement de l'application                                                                                                                                                |
| setup.sh         | Il s'agit d'un fichier shell utilsé dans le fichier Procfile ci-dessous.                                                                                                                                                             |
| Procfile         | Procfile spécifie à peu près les commandes une fois que vous exécutez l'application sur Heroku. Nous spécifions le fichier shell que nous avons créé ci-dessus, puis appelons Streamlit pour exécuter my_streamlit_dashboard_code.py |

Pour étoffer vos connaissances dans le déploiement de votre application streamlit sur Heroku vous pourrez suivre ces
liens:

- https://towardsdatascience.com/a-quick-tutorial-on-how-to-deploy-your-streamlit-app-to-heroku-874e1250dadd
- https://www.youtube.com/watch?v=IWWu9M-aisA&t=379s


## Sources de données

| Nom de fichier                        | lien de la Source                                       |
|---------------------------------------|---------------------------------------------------------|
| Fichiers de données de base du projet | https://www.kaggle.com/c/home-credit-default-risk/data  |


