# Automatisierung des Relabeling-Prozesses durch Computer Vision

**Abstract**
Die vorliegende Projektarbeit beschäftigt sich mit der Automatisierung des Relabeling-Prozesses
durch Computer Vision im Logistikumfeld. Ziel des Projekts ist die Entwicklung eines Proof of
Concept (PoC) zur automatisierten Durchführung des Relabeling-Prozesses. Die entwickelte
Lösung umfasst mehrere Schritte: Labelerkennung, automatischer Bildausrichtung (Rotati-
on), Empfängeradresseerkennung und Postleitzahl-Extraktion (OCR). Zur Objektlokalisierung
und -klassifikation wurden zwei YOLO-Detektoren trainiert: einer zur Labelerkennung auf
dem Gesamtbild und einer zur Empfängeradresseerkennung auf dem Label; zur Korrektur von
Bildrotationen ein kompaktes CNN und für die Textextraktion wurde docTR eingesetzt.

Nach dem Training und der Evaluation der Modelle ergaben sich folgende Ergebnisse: das
YOLO-Modell zur Labelerkennung erreicht auf dem Testset eine Gesamtgenauigkeit von 91,5
% und einen durchschnittlichen IoU-Wert von 0,8845; das YOLO-Modell zur Empfänger-
adresseerkennung erzielt eine IoU-Wert von 0,8389 und eine Genauigkeit von 87,5 % . Das
Rotations-CNN erreicht auf dem Testset eine MSE von 0,012, was einer mittleren Winkeldiffe-
renz von 0,11° entspricht. Auf Pipeline-Ebene wurde die Postleitzahl in 55 von 62 Testbildern
korrekt extrahiert (=89 % Genauigkeit). Die durchschnittliche Verarbeitungszeit betrug lokal
5 Sekunden pro Bild. Für einen späteren Echtbetrieb wird eine Zielzeit von 1 Sekunde pro
Bild angestrebt, die durch den Einsatz GPU-beschleunigter Hardware erreichbar ist. Die bisher
erzielte Genauigkeit ist jedoch noch nicht ausreichend und soll durch weiteres Training und
Optimierung der Modelle verbessert werden.





**Code-Struktur**

Die Implementierung erfolgt über eine Pipeline aus mehreren Python-Skripten, die nacheinander verschiedene Verarbeitungsschritte auf Versandlabels durchführen:

1. **Training eines YOLO-Modells zur Label-Erkennung (`yolo_zur_labelerkennung.py`)**
   Dieses Skript trainiert ein YOLO-Modell, das Labels auf Gesamtbildern zuverlässig erkennen kann. Ein zusätzliches Skript (`yolo_zur_labelerkennung_testen.py`) ermöglicht die lokale Validierung des Modells und die Überprüfung der Erkennungsgenauigkeit.

2. **Ausschneiden der Labels (`cropping.py`)**
   Hier werden die erkannten Labels aus den Gesamtbildern ausgeschnitten, sodass nachfolgende Modelle ausschließlich auf den relevanten Label-Bereichen arbeiten.

3. **Erstellung von Trainingsdaten für das Rotation-CNN (`create_data_4_rotation_cnn.py`)**
   Dieses Skript rotiert 100 repräsentative Labels jeweils in 150 Rotationen, um Trainingsdaten für ein Convolutional Neural Network (CNN) zur Rotationserkennung zu erzeugen.

4. **Rotationserkennung mittels CNN (`cnn_zur_rotationerkennung.py`)**
   Ein CNN wurde trainiert, um die Rotation der Labels zu erkennen und diese vor der OCR-Erkennung korrekt auszurichten. In diesem Skript wird auch die Genauigkeit des Modells getestet.

5. **Erkennung der Empfängeradresse (`yolo_zur_empfängeradresse_erkennung.py`)**
   Ein zweites YOLO-Modell lokalisiert gezielt die Empfängeradresse auf den Labels. Die exakte Position der Adresse wird für die OCR-Erkennung bereitgestellt. Ein Testskript (`yolo_zur_empfängeradresse_erkennung_testen.py`) ermöglicht die Validierung der Erkennungsgenauigkeit.

6. **OCR zur Extraktion der Postleitzahl (`ocr.py`)**
   Dieses Skript liest die Empfängeradresse mit OCR aus und extrahiert gezielt die Postleitzahl.

Zur Vereinfachung der Nutzung wurde ein **Hauptskript (`mainskript3.py`)** entwickelt, das die gesamte Pipeline integriert und die einzelnen Schritte automatisch ausführt. Zur Effizienzsteigerung wurden die Labels direkt im RAM verarbeitet, und eine In-Memory-Zwischenspeicherung mittels NumPy-Arrays implementiert. Dadurch konnten wiederholte Festplattenzugriffe vermieden und die Verarbeitungsgeschwindigkeit deutlich erhöht werden. Das separate Skript zum Ausschneiden der Labels wird im Mainskript nicht verwendet, ist jedoch für die einzelnen Schritte innerhalb der Pipeline erforderlich. Die Pipeline wurde anschließend hinsichtlich **Genauigkeit** und **Geschwindigkeit (gesammt_pipline_testen.py)** getestet.



