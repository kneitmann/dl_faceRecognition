# Project "Face Recognition"



## Meilenstein 1: Projektteam

<u>Deadline</u>: 31.10.

### Aufgabenstellung

1. Projektteam gewählt
   	- Teamgröße: maximal Zweiergruppen (also 1 oder 2 Personen) 
 2. Datenbasis vorhanden
    - Bilder sind geeignet für Training 
3. Softwarebasis vorhanden
   - Bilder/Ergebnisse können geladen, angezeigt, gespeichert werden
   - Code-/Verzeichnis-Struktur überlegt



### ToDos

- [x] Projektteam gewählt
- [x] Datenbasis zusammenstellen
- [x] Repo auf Github erstellen
- [x] Kanban-Board auf Github erstellen
- [x] Projektstruktur aufbauen
- [x] Daten/Bilder laden
- [x] Modell speichern
- [x] Modell laden
- [x] Tensorboard Integration



<u>Zusätzlich:</u>

- [x] Über Architekturen informieren (vor allem ResNet)
- [x] Erstes Netz aufbauen
  - [x] Auf wenige Daten overfitten
- [x] Grober Plan für Face Detection
- [x] ResNet101 und ResNet50 und ResNet152 anschauen und eins auswählen



### Notizen

#### Projektstruktur

|dl_faceRecognition/

​	|-- data/

​		|---- training/

​		|---- test/

​		|---- validation/

​	|-- src/

​	|-- test/

​	|-- log/



## Meilenstein 2: Gesichtserkennung (Detektion)

<u>Deadline</u>: 21.11.

### Aufgabenstellung

- Implementieren Sie eine Methode zur Gesichtserkennung
  - Wählen Sie eine geeignete Netzarchitektur (z.B. ResNet, MobileNet, …)
- Evaluation: Visualisierung & quantitative Bewertung der Ergebnisse
- Trainieren Sie Ihre Architektur
  - Variante 1: Nur auf eigenen Daten 
  - Variante 2: Transfer Learning 



### ToDos

- [x] ResNet50 Modell laden
- [x] Transfer Learning auf neues Modell
- [x] Programmargumente, wenn man altes Modell laden möchte?
- [x] Callback, um Modell zu speichern?
- [x] Code präsentabel und verständlich gestalten
- [x] Modell trainieren
- [x] Modell testen
- [x] Overfitting vermeiden
- [x] Ergebnisse auswerten
- [x] Präsentation erstellen



### Notizen





## Meilenstein 3: Erkennung von Merkmalen

<u>Deadline</u>: 19.12.

### Aufgabenstellung

- Implementieren Sie eine Methode zur Erkennung von  Merkmalen basierend auf einem Gesichtsbild (Maske, Alter) 

- Evaluation: Visualisierung & quantitative Bewertung der Ergebnisse

- Erweitern Sie Ihre Netzarchitektur aus Meilenstein 2 

- Trainieren Sie alle Merkmale gleichzeitig in einem Netz (Multi-Task Learning) 

  - Gesichtserkennung (aus Meilenstein 2) 
  - Maskenerkennung 
  - Altersbestimmung



### ToDos

- [x]  Auf MobileNet umsteigen
- [ ] Erstes Modell implementieren
  - [ ] Multi Task Learning  (3 Top Modelle auf Mobile Net Basis)
  - [ ] Erste Hyperparameter
- [ ] Modell Varianten/Vorgehensweise überlegen
- [x] Datenstruktur/Ordnerstruktur überlegen
- [ ] Testdaten/Vorgehensweise überlegen



### Notizen

#### Präsentation

- Mehr Anwendungsbeispiele
- Positive und negative Testbeispiele
- Trainingsdatensatz näher erläutern (Welche positiven und negativen Daten existieren, etc ..)

#### Sources

- [An Overview of Multi-Task Learning for Deep Learning (ruder.io)](https://ruder.io/multi-task/)
- [2009.09796.pdf (arxiv.org)](https://arxiv.org/pdf/2009.09796.pdf)
- [Multi-Task Learning for Classification with Keras | by Javier Martínez Ojeda | Towards Data Science](https://towardsdatascience.com/multi-task-learning-for-computer-vision-classification-with-keras-36c52e6243d2)
- [Multitask learning in TensorFlow with the Head API | by Simone Scardapane | Towards Data Science](https://towardsdatascience.com/multitask-learning-in-tensorflow-with-the-head-api-68f2717019df)
- [The Functional API (keras.io)](https://keras.io/guides/functional_api/#shared-layers)



## Meilenstein 4: Wiedererkennung (Recognition)

<u>Deadline</u>: 16.01.

### Aufgabenstellung

- Implementieren Sie eine Methode zur Wiedererkennung von Gesichtern
- Evaluation: Visualisierung & quantitative Bewertung der Ergebnisse
- Nutzen Sie ein Siamese Network für Ihr Training
