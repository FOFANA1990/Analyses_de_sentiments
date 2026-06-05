# Explication détaillée du Notebook — Analyse de Sentiment IMDb
**Auteur :** Fofana Boubacar

---

## Vue d'ensemble

Ce notebook réalise une **analyse de sentiment complète** sur le dataset IMDb de 50 000 critiques de films. Il couvre tout le pipeline NLP : du chargement brut des données jusqu'à la comparaison de modèles supervisés et non supervisés, en passant par un prétraitement rigoureux et une analyse exploratoire riche en visualisations.

---

## PARTIE 1 — Traitement des données textuelles

### Cellule 1 : Import Kaggle
```python
import kagglehub
lakshmi25npathi_imdb_dataset_of_50k_movie_reviews_path = kagglehub.dataset_download(...)
```
Cette cellule est spécifique à l'environnement Kaggle. Elle télécharge automatiquement le dataset IMDb (50 000 critiques étiquetées `positive` ou `negative`) via la bibliothèque `kagglehub`. En local, cette étape est remplacée par un simple `pd.read_csv()`.

### Cellule 2 : Imports standards
```python
import numpy as np
import pandas as pd
import os
```
Import des bibliothèques fondamentales : `numpy` pour le calcul numérique, `pandas` pour la gestion des données tabulaires, `os` pour lister les fichiers disponibles dans l'environnement Kaggle.

### Cellule 3 : Chargement du CSV
```python
textes = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
textes.head()
```
Chargement du fichier CSV dans un DataFrame nommé `textes`. Le dataset contient deux colonnes :
- `review` : le texte brut de la critique
- `sentiment` : le label réel (`positive` ou `negative`)

---

## PARTIE 2 — Prétraitement et Nettoyage

### Cellule 4 : Fonction `nettoyer_texte`
Cette fonction centrale applique 9 étapes de nettoyage dans l'ordre :

| Étape | Action | Exemple |
|-------|--------|---------|
| 1 | Mise en minuscules | `"Great Movie"` → `"great movie"` |
| 2 | Conversion emojis | `"😊"` → `":smiling_face:"` puis supprimé |
| 3 | Suppression balises HTML | `"<br />"` → `" "` |
| 4 | Suppression retours à la ligne | `"\n"` → `" "` |
| 5 | Suppression accents (normalisation Unicode) | `"é"` → `"e"` |
| 6 | Suppression des chiffres | `"2023"` → `" "` |
| 7 | Suppression de la ponctuation | `"great!"` → `"great"` |
| 8 | Suppression des underscores | `"good_movie"` → `"good movie"` |
| 9 | Suppression des espaces multiples | `"  "` → `" "` |

La colonne `review_cleaned` est créée pour stocker les textes nettoyés.

### Cellule 5 : Tokenisation en phrases (NLTK)
```python
nltk.tokenize.sent_tokenize(review_text)
```
Chaque critique est découpée en **phrases** grâce au tokeniseur de phrases de NLTK (`punkt`). Cette liste de phrases (`sentences`) sert d'étape intermédiaire.

### Cellule 6 : Tokenisation simple par espace
```python
def tokenizer_nltk_simple(texte):
    return texte.split()
```
Première approche de tokenisation : découpage naïf sur les espaces. Rapide mais moins précise que `word_tokenize`.

### Cellule 7 : Tokenisation avancée avec suppression des stopwords
```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
```
`word_tokenize` est plus sophistiqué que `.split()` : il gère les contractions (`don't` → `do`, `n't`) et la ponctuation. Les **stopwords** (mots vides : `the`, `a`, `is`, `in`…) sont ensuite éliminés car ils n'apportent pas d'information de sentiment. La colonne `review_token` est mise à jour.

### Cellule 8 : Lemmatisation
```python
lemmatizer.lemmatize(word, pos='v')  # verbe
lemmatizer.lemmatize(word, pos='n')  # nom
lemmatizer.lemmatize(word, pos='r')  # adverbe
```
La **lemmatisation** réduit les mots à leur forme canonique (lemme) :
- `"running"` → `"run"` (verbe)
- `"dogs"` → `"dog"` (nom)
- `"better"` → `"well"` (adverbe)

Elle est appliquée successivement pour les trois catégories grammaticales. La colonne `review_lemma` stocke le résultat.

### Cellule 9 : Sauvegarde
```python
textes.to_csv('imdb_clean.csv', index=False)
```
Le DataFrame enrichi est sauvegardé pour éviter de refaire tout le prétraitement à chaque exécution.

---

## PARTIE 3 — Analyse des données textuelles

### 3.1 Distribution des sentiments
```python
sentiment_counts = textes_clean['sentiment'].value_counts()
```
Comptage simple des labels. Le dataset est **parfaitement équilibré** : 25 000 avis positifs et 25 000 avis négatifs. Cet équilibre est fondamental pour l'entraînement des modèles supervisés.

### 3.2 Longueur moyenne des commentaires
```python
textes_clean['nb_mots'] = textes_clean['review_token'].apply(len)
nb_mots_moyen = textes_clean['nb_mots'].mean()
```
Calcule le nombre moyen de tokens par critique après nettoyage et suppression des stopwords.

---

###  Graphique 1 — Nuage de mots : Avis POSITIFS

**Ce que montre ce graphique :**  
Un nuage de mots où la taille de chaque mot est proportionnelle à sa fréquence dans les critiques positives.

**Interprétation :**  
Les termes `film`, `movie`, `one` dominent, ce qui reflète le sujet général du corpus. Les mots évaluatifs (`good`, `great`, `nice`) et les termes narratifs (`story`, `character`, `scene`, `performance`) sont très présents. Les verbes cognitifs (`think`, `know`, `see`, `feel`) traduisent un discours **subjectif et expérientiel** : les spectateurs racontent leur vécu émotionnel. Ce nuage révèle un corpus de critiques **détaillées et enthousiastes**, où l'analyse narrative et le jugement positif sont entrelacés.

---

###  Graphique 2 — Nuage de mots : Avis NÉGATIFS

**Ce que montre ce graphique :**  
Même logique que le nuage positif, mais construit uniquement sur les 25 000 critiques négatives.

**Interprétation :**  
Les mots dominants (`movie`, `film`, `character`, `story`, `scene`) sont similaires au nuage positif, ce qui confirme que le vocabulaire général est partagé. On note cependant la présence d'adjectifs négatifs (`bad`, `boring`) et de termes sur la réalisation (`director`, `actor`). Les verbes subjectifs (`see`, `say`, `make`, `look`, `think`) sont également présents. La proximité des deux nuages montre que le **sentiment n'est pas dans les mots eux-mêmes, mais dans leur contexte**, ce qui explique la limite des approches lexicales simples.

---

###  Graphique 3 — Bar Plot : 20 mots les plus fréquents (corpus entier)

**Ce que montre ce graphique :**  
Un diagramme en barres horizontales des 20 termes les plus fréquents après nettoyage et tokenisation.

**Interprétation :**  
`movie` et `film` écrasent les autres termes en fréquence. Les mots évaluatifs positifs (`good`, `great`) apparaissent dès le top 10, mais les mots négatifs sont absents du classement. Cela suggère un **biais naturel du langage** : les sentiments positifs sont exprimés de façon plus explicite et répétée. Des mots très courants mais peu informatifs (`like`, `one`, `time`) confirment l'importance du prétraitement contextuel (TF-IDF, embeddings) plutôt que de la simple fréquence brute.

---

###  Graphique 4 — Histogramme : Distribution de la longueur des commentaires

**Ce que montre ce graphique :**  
Un histogramme en 20 classes montrant combien de critiques contiennent X mots (après tokenisation).

**Interprétation :**  
La distribution est **fortement asymétrique à droite** (queue longue) :
- La majorité des critiques contient entre **0 et 300 mots** (distribution concentrée).
- Quelques rares critiques atteignent **plusieurs milliers de mots**.
Cette hétérogénéité est typique des corpus de critiques en ligne : la plupart sont concises, mais certaines sont très développées. Elle peut impacter les modèles bag-of-words (TF-IDF), qui traitent différemment les documents courts et longs.

---

###  Graphique 5 — Bar Plot : Top 10 des UNIGRAMMES

**Ce que montre ce graphique :**  
Les 10 mots les plus fréquents de l'ensemble du corpus.

**Interprétation :**  
Très similaire au graphique 3. `movie` et `film` dominent. `good` est le seul mot clairement chargé positivement dans ce top 10. Les termes `time`, `story`, `really`, `would` sont **contexte-dépendants** : leur polarité dépend du reste de la phrase. Cela illustre la limite des modèles purement basés sur les fréquences et justifie l'usage des n-grammes et des modèles de contexte.

---

###  Graphique 6 — Bar Plot : Top 10 des BIGRAMMES

**Ce que montre ce graphique :**  
Les 10 paires de mots consécutifs les plus fréquentes.

**Interprétation :**  
Les bigrammes sont bien plus informatifs que les mots seuls :
- **Positifs clairement identifiables :** `much better`, `one best`
- **Négatifs clairement identifiables :** `waste time`, `low budget`
- **Contextuels :** `special effects`, `year old`
- **Emphatiques (ambivalents) :** `ever seen` (peut précéder "best" ou "worst")

Cette analyse montre que les **bigrammes capturent la polarité** mieux que les unigrammes, ce qui justifie leur intégration dans les modèles TF-IDF ou dans les réseaux de neurones.

---

###  Graphique 7 — Bar Plot : Top 10 des TRIGRAMMES

**Ce que montre ce graphique :**  
Les 10 séquences de 3 mots consécutifs les plus fréquentes.

**Interprétation :**  
Les trigrammes révèlent des **structures émotionnelles complètes** :
- `movie ever seen`, `films ever seen`, `film ever seen` : emphase superlative (polarité forte)
- `worst movies ever`, `one worst movies` : charge **négative très explicite**
- `new york city`, `world war ii` : purement contextuels (lieu / époque)

Les trigrammes negatifs comme `worst movies ever` sont directement exploitables par un classificateur. Ils démontrent qu'un **contexte élargi** (3 mots) capture mieux l'intensité émotionnelle que les unigrammes ou bigrammes.

---

### 3.3 Vectorisation TF-IDF
```python
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(textes_clean['review_lemma'])
```
Le **TF-IDF** (Term Frequency – Inverse Document Frequency) transforme chaque critique en un vecteur de 1000 dimensions. Les mots fréquents dans une critique mais rares dans le corpus obtiennent un score élevé, ce qui pénalise les mots banals (`movie`, `film`) au profit des mots discriminants.

---

## PARTIE 4 — Modélisation du sentiment

### 4.1 Approches supervisées

#### SVM (LinearSVC)
```python
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
svm_model = LinearSVC()
svm_model.fit(x_train, y_train)
```
Le SVM à noyau linéaire est très efficace sur les données textuelles TF-IDF car celles-ci sont de **haute dimension et linéairement séparables**. Le split 80/20 est standard pour évaluer la généralisation.

#### Random Forest
```python
rf_model = RandomForestClassifier(n_estimators=200, stratify=True)
```
200 arbres de décision sont entraînés en parallèle (`n_jobs=-1`). Le paramètre `stratify` garantit que les proportions positive/négative sont identiques dans le train et le test. Plus robuste au surapprentissage que le SVM, mais plus lent.

---

### 4.2 Approches non supervisées

#### TextBlob
```python
TextBlob(x).sentiment.polarity  # score entre -1 et 1
```
TextBlob utilise un dictionnaire de polarité. Score > 0 → `positive`, ≤ 0 → `negative`.

#### VADER
```python
analyzer.polarity_scores(x)['compound']  # score entre -1 et 1
```
VADER est conçu pour les textes courts et informels (réseaux sociaux, critiques). Il intègre des règles spéciales pour les majuscules, la ponctuation répétée et les emojis.

---

###  Graphique 8 — Matrice de confusion : Sentiment réel vs VADER

**Ce que montre ce graphique :**  
Une grille 2×2 comparant les prédictions VADER aux vrais labels.

| | Prédit Négatif | Prédit Positif |
|---|---|---|
| **Réel Négatif** | 13 070  | 11 930  |
| **Réel Positif** | 3 397  | 21 603  |

**Interprétation :**  
VADER détecte bien les avis **positifs** (21 603 vrais positifs, seulement 3 397 faux négatifs). En revanche, il **rate près de la moitié des avis négatifs** (11 930 faux positifs). Cela reflète un biais optimiste : VADER interprète souvent des critiques négatives nuancées comme positives. Ce biais est inhérent aux approches lexicales qui ne captent pas la négation contextuelle ni l'ironie.

---

###  Graphique 9 — Matrice de confusion : Sentiment réel vs TextBlob

**Ce que montre ce graphique :**  
Même grille pour TextBlob.

| | Prédit Négatif | Prédit Positif |
|---|---|---|
| **Réel Négatif** | 10 903  | 14 097  |
| **Réel Positif** | 1 287  | 23 713  |

**Interprétation :**  
TextBlob est encore **plus biaisé vers le positif** que VADER : 14 097 faux positifs contre 11 930 pour VADER. Son rappel pour les positifs est excellent (23 713 vrais positifs, seulement 1 287 faux négatifs), mais au prix d'une détection médiocre des négatifs. Utile pour détecter les avis très positifs, mais non fiable pour les avis négatifs.

---

###  Graphique 10 — Matrice de confusion : VADER vs TextBlob (accord inter-outils)

**Ce que montre ce graphique :**  
Compare les prédictions des deux outils entre eux (sans référence au label réel).

| | TextBlob : Négatif | TextBlob : Positif |
|---|---|---|
| **VADER : Négatif** | 9 272 | 7 195 |
| **VADER : Positif** | 2 918 | 30 615 |

**Interprétation :**  
Les deux outils s'accordent massivement sur les **avis positifs** (30 615 accords), mais divergent davantage sur les avis négatifs. Quand VADER dit négatif, TextBlob dit souvent positif (7 195 désaccords). Cela confirme que **TextBlob est plus "optimiste"** que VADER. Cette comparaison suggère qu'un ensemble des deux outils (vote majoritaire) pourrait améliorer la précision.

---

###  Graphique 11 — Bar Plot groupé : Sentiment réel vs VADER vs TextBlob

**Ce que montre ce graphique :**  
Trois barres côte à côte pour chaque classe (négatif / positif), comparant le nombre de prédictions de chaque méthode au nombre réel.

**Interprétation :**  
- **Données réelles :** parfaitement équilibrées (25 000 / 25 000)
- **VADER :** ~16 500 négatifs / ~33 500 positifs → déséquilibre marqué
- **TextBlob :** ~12 000 négatifs / ~38 000 positifs → déséquilibre encore plus fort

Ce graphique synthétise visuellement le **biais positif systématique** des deux méthodes lexicales. Dans un contexte applicatif (détection d'insatisfaction client, modération), ce biais serait problématique. Il justifie l'utilisation de modèles supervisés ou d'un rééquilibrage des prédictions.

---

## Synthèse et Recommandations

| Méthode | Accuracy | Points forts | Limites |
|---------|----------|-------------|---------|
| **SVM** | Élevée (supervisé) | Très efficace sur TF-IDF | Nécessite des données étiquetées |
| **Random Forest** | Élevée (supervisé) | Robuste, parallélisable | Plus lent, peu interprétable |
| **VADER** | 0,69 | Équilibré, adapté aux textes courts | Biais positif, pas de contexte |
| **TextBlob** | 0,69 | Excellent rappel positif | Très biaisé positif |

**Pour aller plus loin :**
- Utiliser `BERT` ou `RoBERTa` (modèles Transformer) pour capturer le contexte
- Ajouter la négation comme feature (`not good` ≠ `good`)
- Tester un ensemble VADER + TextBlob avec vote majoritaire
- Appliquer une pondération des classes pour corriger le biais positif
