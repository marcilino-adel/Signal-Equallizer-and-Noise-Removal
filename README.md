# **Signal Equalizer Application**

A robust desktop application designed to modify frequency components of audio signals in real time. With multiple equalization modes and dynamic visualizations, the Signal Equalizer supports audio manipulation for both technical and creative purposes.

---

## **Features**

### 🎹 **Equalization Modes**
1. **Uniform Range Mode**  
   - Divides the frequency range into 10 equal bands, each controlled by a slider.  
   - Includes a synthetic signal for validation, composed of pure frequencies.  

2. **Musical Instruments & Animals Mode**  
   - Adjust the magnitudes of specific musical instruments and animal sounds in a mixed signal.  
   - Supports at least three musical instruments and three animal sounds.

3. **Music & Letters Mode**  
   - Adjust specific instrument sounds or letter sounds.  
   - Works with a mix of at least three instrument sounds and three letter sounds.

4. **Noise Reduction Mode**  
   - Detect noise in the spectrogram using a rectangle tool.  
   - Removes noise using a Wiener filter across the entire signal.

---

### 📊 **Dynamic Visualizations**
- **Fourier Transform**:  
  - Supports two frequency scales:  
    - Linear scale  
    - Audiogram scale  
  - Toggle scales seamlessly without interrupting functionality.

- **Spectrograms**:  
  - Input and output spectrograms display frequency-time intensity.  
  - Output spectrogram updates dynamically when sliders are adjusted.

- **Linked Cine Viewers**:  
  - Input and output signal viewers are synchronized in time and zoom levels.  
  - Playback controls: play, stop, pause, speed adjustment, zoom, pan, and reset.

---

## **Screenshots**

Include relevant images, such as:
- The main application interface.
- Examples of the Fourier transform and spectrogram views.
- A comparison of sliders in different modes.

---

## **Installation**

### 1⃣ **Clone the Repository**
```bash
git clone https://github.com/your-username/signal-equalizer.git
cd signal-equalizer
```

### 2⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3⃣ **Run the Application**
```bash
python main.py
```

---

## **Usage**

### **Switching Modes**
- Select the desired mode via the mode selector in the UI.
- Slider labels and functions dynamically update for the selected mode.

### **Noise Reduction**
- In **Noise Reduction Mode**, draw a rectangle on the spectrogram to isolate the noisy region.  
- Apply Wiener filtering to remove noise across the signal.

### **Spectrogram & Fourier Transform**
- Toggle the spectrogram display via the provided button.  
- Switch between **Linear** and **Audiogram** scales for Fourier analysis.

### **Playback Controls**
- Use play/pause/stop to manage signal playback.  
- Adjust playback speed and zoom for finer signal inspection.

---

## **Validation**
- **Uniform Range Mode**:  
  - Test with synthetic signals composed of known frequencies to validate equalization behavior.  

- **Other Modes**:  
  - Validate by applying sliders to known mixes of instruments, animal sounds, or letters.

---

## **Project Structure**

```
├── src/                # Source code
│   ├── main.py         # Main application entry point
│   ├── equalizer_modes.py  # Equalizer logic for all modes
│   ├── ui_design.ui    # Qt Designer file for GUI
│   └── ...
├── assets/             # Synthetic signals and sample audio files
├── docs/               # Documentation and references
├── requirements.txt    # Python dependencies
└── README.md           # Project overview
```

---

## **Contributing**
Contributions are welcome! If you'd like to improve this project:
1. Fork the repository.  
2. Create a feature branch: `git checkout -b feature-name`.  
3. Commit your changes: `git commit -m "Add feature-name"`.  
4. Push to the branch: `git push origin feature-name`.  
5. Open a pull request.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**
This application was developed as part of an academic project, inspired by real-world use cases in the music, speech, and biomedical industries.

---

