# 🎵 ICA vs PCA Analysis Tool 🖼️

A comprehensive Streamlit application for **Blind Source Separation** comparing **Independent Component Analysis (ICA)** and **Principal Component Analysis (PCA)** for both audio and image data.

## Features

✨ **Audio Source Separation**
- Upload MP3 or WAV files
- Automatic audio mixture creation (synthetic mixtures for mono files)
- Separate components using ICA and PCA
- Download separated audio components
- View waveforms and spectrograms
- Compare RMS energy statistics

🖼️ **Image Source Separation**
- Upload up to 3 images (JPG, PNG, BMP, TIFF)
- Automatic image mixing using random mixing matrices
- Separate mixed images using ICA and PCA
- Download separated image components
- Visual comparison grid of all stages
- Performance metrics and statistics

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory:**
```powershell
cd C:\Documents\ICA
```

2. **Create and activate virtual environment:**
```powershell
python -m venv venv
venv\Scripts\activate.bat
```

3. **Install dependencies:**
```powershell
pip install numpy matplotlib streamlit librosa soundfile scipy scikit-image scikit-learn
```

## Usage

Run the Streamlit app:

```powershell
streamlit run main.py
```

The app will open in your default browser at `http://localhost:8501`

### Audio Analysis
1. Select "🎵 Audio Source Separation" from the dropdown
2. Upload an MP3 or WAV file
3. Adjust parameters (number of components, sample rate)
4. View original audio waveform and spectrogram
5. Wait for ICA and PCA separation
6. Compare results in different tabs
7. Download separated audio components

### Image Analysis
1. Select "🖼️ Image Source Separation" from the dropdown
2. Upload exactly 3 images
3. Adjust image processing size if needed
4. Wait for image mixing and separation
5. View results in comparison tabs
6. Download separated image components

## Methods Explained

### 🔵 ICA (Independent Component Analysis)
- **Best for:** Blind source separation, extracting independent signals
- **Principle:** Finds statistically independent components
- **Assumption:** Signals have non-Gaussian distributions
- **Result:** Components often resemble original sources
- **Note:** Order may differ from originals

### 🔴 PCA (Principal Component Analysis)
- **Best for:** Dimensionality reduction, data compression
- **Principle:** Finds orthogonal directions of maximum variance
- **Result:** Maximizes explained variance
- **Limitation:** May produce blurred/averaged components
- **Use case:** Better for noise reduction than source separation

## Project Structure

```
ICA/
├── main.py              # Main Streamlit application
├── venv/               # Virtual environment (created locally)
├── README.md           # This file
├── requirements.txt    # (Optional) Dependencies list
└── ica_outputs/        # Output folder for processed files
```

## Key Components

### Audio Functions
- `convert_mp3_to_wav()` - Convert MP3 to WAV format
- `create_audio_mixtures()` - Generate synthetic audio mixtures
- `process_audio_ica_pca()` - Apply ICA and PCA to audio
- `plot_audio_waveform()` - Visualize waveforms
- `plot_audio_spectrogram()` - Visualize spectrograms

### Image Functions
- `load_and_preprocess_image()` - Load and normalize images
- `create_image_mixtures()` - Generate mixed images
- `process_image_ica_pca()` - Apply ICA and PCA to images
- `normalize_component()` - Normalize output components

## System Requirements

- **RAM:** 2GB minimum (4GB+ recommended)
- **Disk Space:** 500MB
- **Python:** 3.8+
- **Browsers:** Modern browsers (Chrome, Firefox, Edge)

## Troubleshooting

### Module Import Errors
```powershell
# Ensure venv is activated
venv\Scripts\activate.bat

# Reinstall dependencies
pip install --upgrade pip setuptools wheel
pip install numpy matplotlib streamlit librosa soundfile scipy scikit-image scikit-learn
```

### Streamlit Not Starting
```powershell
# Check if streamlit is installed
pip show streamlit

# Reinstall if necessary
pip install --force-reinstall streamlit
```

### Audio Processing Errors
- Ensure audio file is not corrupted
- Try a different audio format
- Reduce sample rate to 22050 Hz

### Image Processing Errors
- Upload only 3 images
- Ensure images are RGB or grayscale
- Check image file size (max ~10MB recommended)

## Performance Tips

✅ **For Audio:**
- Use WAV files when possible (faster than MP3)
- Larger sample rates increase processing time
- 2-3 components recommended for speed

✅ **For Images:**
- Start with 256×256 resolution
- Ensure images are diverse for better separation
- Reduce image size for faster processing

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | Latest | Numerical computing |
| matplotlib | Latest | Visualization |
| streamlit | Latest | Web app framework |
| librosa | Latest | Audio processing |
| soundfile | Latest | Audio I/O |
| scipy | Latest | Signal processing |
| scikit-learn | Latest | Machine learning (ICA, PCA) |
| scikit-image | Latest | Image processing |

## Color Scheme

- **Primary Teal:** #17a2b8
- **Light Teal:** #5dccdb
- **Dark Teal:** #0d7f8f
- **Green (Success):** #28a745
- **Red (PCA):** #dc3545
- **Yellow (Warning):** #ffc107
- **Dark Text:** #2c3e50
- **Light Gray:** #6c757d

## License

This project is provided as-is for educational and research purposes.

## Notes

- ICA works best with multiple observed mixtures (multiple recordings/viewpoints)
- For single mono audio, synthetic mixtures are created for analysis
- Image separation quality depends on mixing matrix properties
- PCA components are ordered by variance, ICA by independence
