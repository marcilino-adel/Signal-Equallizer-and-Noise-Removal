# **🎿 Signal Equalizer Application**

A robust desktop application designed to modify frequency components of audio signals in real time. With multiple equalization modes and dynamic visualizations, the Signal Equalizer supports audio manipulation for both technical and creative purposes.

---

## **✨ Features**

### 🎹 **Equalization Modes**
1. **Uniform Range Mode**  
   - Divides the frequency range into **10 equal bands**, each controlled by a slider.  
   - Real-time updates of audio playback and output visualizations.

2. **Musical Instruments & Animals Mode**  
   - Adjust magnitudes of **specific musical instruments and animal sounds** in a mixed signal.  
   - Supports at least **three musical instruments** and **three animal sounds**.

3. **Music & Letters Mode**  
   - Adjust specific **instrument sounds** or **letter sounds** in audio.  
   - Supports at least **three instruments** and **three letters**.

4. **Noise Reduction Mode**  
   - Detect noise in the signal using a **rectangle tool**.  
   - Apply a **Wiener filter** to remove noise across the signal.

---

### 📊 **Dynamic Visualizations**
- **Fourier Transform**:  
  - Two frequency scales:  
    - Linear scale  
    - Audiogram scale  
  - **Seamless toggling** between scales without interrupting functionality.

- **Spectrograms**:  
  - Input and output spectrograms display **frequency-time intensity**.  
  - **Dynamic updates** for the output spectrogram when sliders are adjusted.

- **Linked Cine Viewers**:  
  - **Synchronized playback** of input and output signals with zoom and pan.  
  - Playback controls:  
    - **Play**, **Pause**, **Stop**,  
    - **Speed Adjustment**, **Zoom**, **Reset**.

---

## **💁‍♂️ Usage**

### **Switching Modes**  
- **Select a mode** via the mode selector in the UI.  
- Sliders and labels dynamically **adapt** based on the selected mode.

### **Noise Reduction**  
1. In **Noise Reduction Mode**, **draw a rectangle** on the spectrogram to isolate noise.  
2. Apply **Wiener filtering** to remove noise across the signal.

### **Spectrogram & Fourier Transform**  
- Use the **toggle button** to show or hide the spectrogram.  
- Seamlessly **switch between Linear and Audiogram scales**.

### **Playback Controls**  
- Use the **play, pause, stop** buttons to manage signal playback.  
- Adjust playback **speed** and **zoom** for finer signal inspection.

---

## **📁 Project Structure**

### **💧 Directories**

- **images/**: Screenshots .
- **data/**: Sound files.


### **🔂 Files**
- **README.md**: Project overview and setup instructions.
- **requirements.txt**: List of dependencies.
- **main.py**: Implementation of the Main Functio



```




### **⚙️ Installation**

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/marcilino-adel/Signal-Equallizer-and-Noise-Removal.git
cd Signal-Equallizer-and-Noise-Removal
```

### 2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3️⃣ **Run the Application**
```bash
python main.py
```

---

## **🎨 Screenshots**

### 🎿 **Equalizer Interface**  
![Equalizer Interface](https://github.com/marcilino-adel/Signal-Equallizer-and-Noise-Removal/blob/30c654719ec51ddbbd51f5abff8a6fe24d38694c/images/Signal%20Equalizer%201.png)

### 🔊 **Spectrogram & Signal Views**  
![Spectrogram and Signal Views](https://github.com/marcilino-adel/Signal-Equallizer-and-Noise-Removal/blob/30c654719ec51ddbbd51f5abff8a6fe24d38694c/images/Signal%20Equalizer%202.png)

---
## **🔒 License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **📈 Acknowledgments**
This application was developed as part of an academic project, inspired by real-world use cases in the **music**, **speech**, and **biomedical industries**.

---

## **🌐 Contributors**
- [Ziad Mohamed](https://github.com/Ziadmohammed200)  
- [Marcilino Adel](https://github.com/marcilino-adel)  
- [Ahmed Etman](https://github.com/AhmedEtma)  
- [Pavly Awad](https://github.com/PavlyAwad)  
- [Ahmed Rafat](https://github.com/AhmeedRaafatt)  






