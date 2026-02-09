
# CUDA Video Processing Pipeline

Pipeline di elaborazione video **real-time** basata su **CUDA** e **OpenCV**. Il programma acquisisce uno stream da webcam, lo converte in **grayscale** e applica una delle operazioni implementate su GPU, mostrando a schermo lo stream processato.

Operazioni disponibili (versioni base):

* **Edge Detection (Sobel + soglia)**
* **Crop (ROI)**
* **Scale**
* **Rotate**
* **Optical Flow** (tramite libreria / implementazione presente nel progetto)

---

## ⚙️ Utilizzo del progetto

Il progetto è composto da:

* **Pipeline principale (C++/OpenCV + CUDA)**: acquisizione video, trasferimenti Host↔Device, lancio kernel e visualizzazione.
* **Kernel CUDA**: implementazioni delle operazioni (file `.cu`) richiamate dal main.

### Funzionamento

All’avvio il programma:

1. Apre la webcam tramite OpenCV.
2. Legge un frame per determinare risoluzione e formato.
3. Converte i frame in **grayscale** (CV_8UC1).
4. Alloca i buffer su GPU e un buffer di output su CPU.
5. Entra nel loop real-time:

   * acquisisce frame
   * converte in grayscale
   * copia Host→Device
   * esegue l’operazione selezionata (kernel CUDA / funzione dedicata)
   * copia Device→Host
   * mostra l’output con `cv::imshow`

---

## 🧰 Comandi principali

### 🔨 Compilazione (un solo comando)

Dalla directory principale del progetto:

```bash
make
```

Questo comando:

* configura CMake (cartella `build/`)
* compila l’eseguibile `app`

> Output atteso: `build/app`

### ▶️ Esecuzione

```bash
./build/app
```

All’avvio ti verrà chiesto di selezionare l’operazione:

* `1` → Edge Detection (Sobel)
* `2` → Crop
* `3` → Scale
* `4` → Rotate
* `5` → Optical Flow

### ⌨️ Controlli runtime

* **ESC**: termina il programma
* (Solo Edge Detection) **w/s**: aumenta/diminuisce la soglia (threshold) durante l’esecuzione
  *(Nota: per leggere i tasti devi cliccare sulla finestra OpenCV “Result”)*

---

## 🧹 Pulizia dei file generati

Per rimuovere la cartella di build e i file generati:

```bash
make clean
```

---

## ✅ Dipendenze

* **CUDA Toolkit** (nvcc)
* **OpenCV 4** (installato e linkato via CMake)

Nel progetto `CMakeLists.txt` è impostato:

* `OpenCV_DIR="/usr/local/lib/cmake/opencv4"`

Se OpenCV è installato in un path diverso, aggiorna `OpenCV_DIR` nel `CMakeLists.txt`.

---

## 🔚 Terminazione

* Per uscire dall’app: premi **ESC** sulla finestra OpenCV.
* Se la tastiera non viene letta, clicca prima sulla finestra “Result” (focus).


