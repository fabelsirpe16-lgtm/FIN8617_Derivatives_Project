

**FIN8617 – Produits dérivés**  

**Pricing d’options, couverture delta, volatilité \& simulation Monte Carlo**



**1. Description du projet**

Ce projet s’inscrit dans le cadre du cours FIN8617 – Produits dérivés, visant à appliquer les modèles fondamentaux de valorisation des options et des stratégies de couverture.  

L’objectif est d’utiliser MATLAB pour analyser la dynamique du S\&P 500, comparer volatilité implicite et historique, pricer des options européennes et américaines avec différents modèles et étudier les performances d’une stratégie de delta hedging.



**2. Contenu du dépôt GitHub**

Le dépôt comprend :



Code/

\- TP\_FIN8617.m : Script MATLAB principal contenant :

&nbsp; • Importation et nettoyage des données

&nbsp; • Analyse volatilité implicite vs historique

&nbsp; • Pricing Black–Scholes d’un put ATM

&nbsp; • Calcul des Grecs (Delta, Gamma, Vega, Theta)

&nbsp; • Delta hedging : portefeuille de réplication \& PnL

&nbsp; • Pricing d’une option américaine via arbre binomial

&nbsp; • Simulation Monte Carlo (digitale, barrière)

&nbsp; • Ajustement pour investisseur canadien (FX USD/CAD)



Data/

\- Data\_TP.xlsx : Données utilisées



Report/

\- TP\_FIN8617.pdf : Rapport complet (version PPT convertie)



**3. Résumé technique**

✔ Analyse volatilité  

✔ Pricing Black–Scholes  

✔ Grecs complets  

✔ Delta hedging + PnL  

✔ Arbre binomial (option américaine vs européenne)  

✔ Monte Carlo digital \& barrière  

✔ Ajustement en CAD  



**4. Structure du projet à publier sur GitHub**



FIN8617\_Derivatives\_Project/

│

├── Code/

│   └── TP\_FIN8617.m

│

├── Data/

│   └── Data\_TP.xlsx

│

├── Report/

│   └── TP\_FIN8617.pdf

│

└── README.md



**5. Instructions pour exécuter le code MATLAB**

1\. Ouvrir MATLAB  

2\. Ajouter le dossier Code/ au path : addpath('Code')

3\. Vérifier que Data\_TP.xlsx est bien dans Data/

4\. Lancer le script : run('TP\_FIN8617.m')



**6. Compétences démontrées**

• Modélisation : Black–Scholes, Binomial  

• Couverture : Delta hedging  

• Simulation : Monte Carlo  

• Analyse du risque en CAD  

• MATLAB – manipulation et modélisation financière  



**7. Auteur**

Projet réalisé par Fabel Sirpe.





