# Prévision de la consommation d'électricité en France

Deux parties, sur les données publiques de RTE :

1. un modèle additif généralisé (GAM) journalier en Python, avec une page qui explique chaque jour (novembre 2026) ;
2. une application R Shiny de prévision mensuelle, 2012-2024 (novembre 2024).

## Partie 1 : GAM journalier en Python (novembre 2026)

Page : `docs/index.html` (GitHub Pages). Code : `gam/`.

- Données ODRÉ (Licence Ouverte v2.0) : consommation brute (2016 – sept. 2026), températures quotidiennes
  régionales (2016 – oct. 2026), éCO2mix national consolidé pour la prévision J-1 de RTE (jusqu'à août 2026).
- Modèle (pyGAM) : s(température lissée) + s(jour de l'année, cyclique) + f(type de jour, férié en semaine = niveau propre)
  + f(pont) + f(fêtes de fin d'année) + f(niveau annuel). Apprentissage 2016 – août 2025, validation des choix
  sur sept. 2024 – août 2025 (`validate.py`), test sept. 2025 – août 2026 (365 jours).
- Résultat (MAPE) : GAM 2,16 % avec la température observée, 2,65 % avec la température de la veille ;
  prévision J-1 de RTE 1,24 % une fois retiré son décalage de niveau (-1,04 GW, mesuré sur les 12 mois avant
  le test), 2,36 % brute ; naïf 52 semaines 5,99 %. RTE reste nettement meilleur : le GAM est un modèle
  explicatif, pas un outil opérationnel. Les termes calendaires ont été ajoutés après lecture des
  erreurs du test : le 2,16 % est un peu optimiste.
- Correction par les erreurs connues : une prévision J-1 connaît les erreurs jusqu'à J-2 ; on ajoute k fois
  leur moyenne lissée (k et le lissage choisis sur l'année de validation). GAM avec la température de la veille :
  2,65 % -> 2,25 % (mieux que RTE brut, moins bien que RTE corrigé) ; avec la température observée : 2,16 % -> 1,56 %.
  L'idée vient aussi des erreurs du test (autocorrélation 0,77 à un jour).
- La page décompose chaque jour : base + température + saison + jour de la semaine + férié + pont + fêtes
  = prévision, puis réalisé et erreur (graphique D3), et liste les 10 plus grosses erreurs. Limites détaillées
  sur la page (résidus autocorrélés, extrapolation au-delà de 27,8 °C, concurvité température/saison).

Reproduire :

```bash
cd gam
# télécharger les trois exports CSV dans gam/data_raw/ (voir les URL dans build_data.py)
uv run python build_data.py   # -> gam/data/daily.csv
uv run python validate.py     # -> docs/data/validation.json
uv run python fit.py          # -> docs/data/*.json et docs/data/data.js
```

## Partie 2 : application R Shiny, prévision mensuelle (2012-2024)

Application R Shiny pour analyser et prévoir la consommation d'électricité de la France métropolitaine
(hors Corse), à partir des données publiques de RTE. Projet de la spécialisation en statistique appliquée,
novembre 2024.

### Données

- Source : jeu « Consommation quotidienne brute » (RTE / GRTgaz), plateforme ODRÉ :
  https://odre.opendatasoft.com/explore/dataset/consommation-quotidienne-brute/
- Licence : Licence Ouverte v2.0 (Etalab).
- Contenu utilisé : la consommation brute d'électricité, un point toutes les 30 minutes, en MW.
- Les données ne sont pas dans le dépôt. Téléchargez l'export CSV (séparateur `;`) et enregistrez-le sous
  `consommation-quotidienne-brute.csv` à côté de `app.R`. Au premier lancement, l'application crée
  `data_nettoyees.Rdata` (cache).

#### Unités

L'application additionne les valeurs en MW du pas de 30 minutes, par jour puis par mois.
Les axes indiquent donc « somme des valeurs MW au pas de 30 min ».
Pour obtenir une énergie en MWh, multipliez cette somme par 0,5 h.
Ce facteur constant ne change ni les modèles ni le MAPE.

### Méthode

1. Agrégation mensuelle, série de janvier 2012 à avril 2024.
2. Découpage dans le temps :
   - apprentissage : jusqu'à avril 2022 ;
   - validation : mai 2022 à avril 2023 ;
   - test : mai 2023 à avril 2024 (12 mois).
3. Modèles :
   - baseline saisonnière naïve (SNaive) : la valeur du même mois de l'année précédente ;
   - SARIMA(1,0,1)(0,1,2)[12], ordre fixé ;
   - SARIMA(2,0,1)(1,1,1)[12], ordre choisi à partir des graphiques ACF et PACF.
4. Prévision glissante (rolling forecast) sur la période de test, avec un horizon réglable.
5. Métriques : RMSE et MAPE. Analyse des résidus (histogramme, ACF, test de Ljung-Box).

### Lancer l'application

```r
install.packages(c("shiny", "shinythemes", "dplyr", "lubridate", "plotly", "forecast", "tseries",
                   "ggfortify", "scales", "xts", "Metrics", "MuMIn"))
shiny::runApp("app.R")
```

La liste complète des paquets est en tête de `app.R`. Les étiquettes de mois utilisent les locales
`en_US.UTF-8` et `fr_FR.UTF-8` ; elles doivent exister sur la machine.

### Limites

- Série mensuelle : trop grossière pour l'exploitation au jour le jour.
- Pas de variable de température ni de calendrier (jours fériés, vacances).
- Le test porte sur 12 points mensuels : les métriques varient beaucoup d'une période à l'autre.
