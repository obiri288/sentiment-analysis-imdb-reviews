# Stimmungsanalyse von IMDb-Filmkritiken

Dieses Repository enthält ein Machine-Learning-Projekt zur Stimmungsanalyse (Sentiment Analysis) von 50.000 Filmkritiken aus dem IMDb-Datensatz. Das Ziel war es, ein Modell zu trainieren, das automatisch erkennt, ob eine Kritik eine **positive** oder **negative** Stimmung ausdrückt.

Es wurden zwei verschiedene Methoden zur Text-Vektorisierung verglichen: **Bag-of-Words** und die fortschrittlichere **TF-IDF**-Methode. Das trainierte `LogisticRegression`-Modell erreichte mit TF-IDF eine **Genauigkeit von ca. 89%**.

---

## Inhaltsverzeichnis
1.  [Projekt-Workflow](#projekt-workflow)
2.  [Ergebnisse und Vergleich](#ergebnisse-und-vergleich)
3.  [Verwendete Technologien](#verwendete-technologien)
4.  [Setup und Ausführung](#setup-und-ausführung)
5.  [Autor](#autor)
6.  [Lizenz](#lizenz)

---

## Projekt-Workflow

Das Projekt folgt einem typischen Workflow für Natural Language Processing (NLP)-Klassifikationsaufgaben:

**1. Datenbeschaffung und -inspektion**
* Einlesen des "IMDb Dataset of 50K Movie Reviews" von Kaggle mit `pandas`.
* Behebung eines `ParserError` durch den Wechsel zur `python`-Engine in `read_csv`, um Formatierungsprobleme in der Textdatei zu bewältigen.
* Analyse der Datenstruktur: 50.000 Reviews, aufgeteilt in 25.000 positive und 25.000 negative Kritiken (ein perfekt ausbalancierter Datensatz).

**2. Aufteilung der Daten**
* Die Daten wurden in ein Trainingsset (80%) und ein Testset (20%) aufgeteilt, um eine faire Modellevaluierung zu gewährleisten.

**3. Text-Vektorisierung (Methodenvergleich)**
Da Machine-Learning-Modelle Text nicht direkt verarbeiten können, wurden zwei Methoden zur Umwandlung von Text in numerische Vektoren angewendet und verglichen:

* **Methode A: Bag-of-Words (BoW)**
    * Mittels `CountVectorizer` aus `scikit-learn` wurde für jede Kritik ein Vektor erstellt, der die Häufigkeit jedes Wortes zählt. Es wurde ein Vokabular von den 5.000 häufigsten Wörtern verwendet.

* **Methode B: TF-IDF (Term Frequency-Inverse Document Frequency)**
    * Mittels `TfidfVectorizer` wurde ein Vektor erstellt, der nicht nur die Worthäufigkeit in einer Kritik (TF) berücksichtigt, sondern auch, wie selten oder häufig ein Wort im gesamten Datensatz ist (IDF). Dies gewichtet wichtige, aber seltene Wörter stärker.

**4. Modelltraining und Evaluierung**
* Ein `LogisticRegression`-Modell wurde jeweils auf den BoW-vektorisierten und den TF-IDF-vektorisierten Daten trainiert.
* Die Leistung beider Ansätze wurde auf dem Testset mithilfe von Genauigkeit (Accuracy) und dem Klassifikationsbericht bewertet.

---

## Ergebnisse und Vergleich

Der Vergleich zeigt, dass die fortschrittlichere TF-IDF-Methode zu einer besseren Modellleistung führt.

| Vektorisierungs-Methode        | Modell                 | Genauigkeit (Accuracy) |
| ------------------------------ | ---------------------- | ---------------------- |
| Bag-of-Words (`CountVectorizer`) | Logistische Regression | `~86-88%`                |
| TF-IDF (`TfidfVectorizer`)       | Logistische Regression | `~88-90%` (Besser ✅)      |

*(Bitte ersetze die Werte durch deine exakten Ergebnisse aus deinem Notebook)*

Die Ergebnisse zeigen, dass die Berücksichtigung der Worthäufigkeit im gesamten Dokumentenkorpus (TF-IDF) dem Modell hilft, relevantere Muster zu erkennen und somit genauere Vorhersagen zu treffen.

---

## Verwendete Technologien
* **Sprache:** Python 3.x
* **Bibliotheken:**
    * `Pandas`: Für das Datenmanagement.
    * `Scikit-learn`: Für Datenaufteilung (`train_test_split`), Text-Vektorisierung (`CountVectorizer`, `TfidfVectorizer`), das Modell (`LogisticRegression`) und Evaluierungsmetriken.
* **Umgebung:** Jupyter Notebook / Google Colaboratory.

---

## Setup und Ausführung

1.  **Klone dieses Repository:**
    ```bash
    git clone [https://github.com/obiri288/sentiment-analysis-imdb-reviews.git](https://github.com/obiri288/sentiment-analysis-imdb-reviews.git)
    cd sentiment-analysis-imdb-reviews
    ```
    
2.  **Installiere die Abhängigkeiten:**
    ```bash
    pip install pandas scikit-learn
    ```

3.  **Führe das Notebook aus:**
    Öffne die `.ipynb`-Datei in einer Jupyter-Umgebung. Die Daten (`IMDB Dataset.csv`) müssen sich im selben Verzeichnis befinden oder der Pfad muss angepasst werden.

---

## Autor

* **OBIRI BORDOM**
* GitHub: `https://github.com/obiri288`

---

## Lizenz

Dieses Projekt ist unter der **MIT Lizenz** lizenziert.
