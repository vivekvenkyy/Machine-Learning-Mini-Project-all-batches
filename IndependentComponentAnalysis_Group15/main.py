"""
Streamlit Application: ICA vs PCA Analysis
Supports both Audio and Image Blind Source Separation

Run with: streamlit run app.py
"""

import os
import io
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# Audio processing imports
import librosa
import librosa.display
import soundfile as sf
from scipy.signal import butter, lfilter, fftconvolve

# Image processing imports
from skimage import io as skio
from skimage import color, transform

# ML imports
from sklearn.decomposition import FastICA, PCA

import warnings
warnings.filterwarnings('ignore')

# ================== CONFIGURATION ==================
st.set_page_config(
    page_title="ICA vs PCA Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with color theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-teal: #17a2b8;
        --light-teal: #5dccdb;
        --dark-teal: #0d7f8f;
        --accent-pink: #ff6b9d;
        --accent-yellow: #ffc107;
        --accent-green: #28a745;
        --bg-light: #f8f9fa;
        --text-dark: #2c3e50;
    }
    
    /* Hide default streamlit elements - partially hidden */
    footer {visibility: hidden;}
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
    }
    
    /* General spacing and text improvements */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
    }
    
    p {
        color: #495057;
        line-height: 1.6;
    }
    
    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #17a2b8 0%, #5dccdb 100%);
        padding: 4rem 2rem;
        border-radius: 20px;
        margin-bottom: 3rem;
        box-shadow: 0 10px 30px rgba(23, 162, 184, 0.3);
        text-align: center;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        letter-spacing: 1px;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: rgba(255,255,255,0.98);
        font-weight: 400;
        margin-top: 1rem;
        letter-spacing: 0.5px;
    }
    
    /* Mode selector card */
    .mode-selector-card {
        background: white;
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 3rem;
        color: #2c3e50;
    }
    
    /* Section headers */
    .section-header {
        color: #17a2b8;
        font-size: 2rem;
        font-weight: 700;
        margin: 3rem 0 2rem 0;
        padding: 1rem 0 0.75rem 0;
        border-bottom: 3px solid #17a2b8;
        letter-spacing: 0.5px;
    }
    
    .sub-section-header {
        color: #0d7f8f;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2.5rem 0 1.5rem 0;
        padding-top: 1rem;
        letter-spacing: 0.3px;
    }
    
    /* Info boxes */
    .info-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 5px solid #17a2b8;
        margin: 2rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        color: #2c3e50;
    }
    
    .info-card h4 {
        color: #0d7f8f;
        font-weight: 700;
        margin-bottom: 1rem;
        margin-top: 0;
    }
    
    .info-card p {
        color: #2c3e50;
        line-height: 1.8;
        margin-bottom: 0.75rem;
    }
    
    .info-card ul {
        color: #2c3e50;
        line-height: 1.9;
        padding-left: 1.5rem;
    }
    
    .info-card li {
        margin-bottom: 0.5rem;
    }
    
    .success-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #28a745;
        margin: 1.5rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        color: #155724;
        font-weight: 500;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ffc107;
        margin: 1.5rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        color: #664d03;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #17a2b8 0%, #5dccdb 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(23, 162, 184, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(23, 162, 184, 0.6);
    }
    
    /* Download button styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #28a745 0%, #5cb85c 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);
    }
    
    /* Selectbox styling */
    .stSelectbox [data-baseweb="select"] {
        background-color: white;
        border-radius: 10px;
        border: 2px solid #17a2b8;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #17a2b8;
    }
    
    .stFileUploader section > div {
        color: #2c3e50;
    }
    
    /* Metrics styling */
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #17a2b8;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    .metric-card h4 {
        color: #17a2b8;
        margin: 0 0 1rem 0;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .metric-card p {
        color: #2c3e50;
        margin: 0.75rem 0;
        font-weight: 500;
    }
    
    .metric-card p strong {
        color: #0d7f8f;
        font-weight: 700;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
        border: 2px solid #e0e0e0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #17a2b8 0%, #5dccdb 100%);
        color: white;
        border-color: #17a2b8;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 10px;
        border: 2px solid #17a2b8;
        font-weight: 600;
        color: #17a2b8;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #17a2b8 0%, #5dccdb 100%);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #17a2b8 !important;
    }
    
    /* Audio player styling */
    audio {
        width: 100%;
        border-radius: 10px;
    }
    
    /* Spacing utilities */
    .spacer-sm {
        margin: 1rem 0;
    }
    
    .spacer-md {
        margin: 1.5rem 0;
    }
    
    .spacer-lg {
        margin: 2.5rem 0;
    }
    
    /* Markdown text color */
    .markdown-text {
        color: #2c3e50;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)

# ================== AUDIO FUNCTIONS ==================

def convert_mp3_to_wav(input_file, target_sr=22050, mono=True):
    """Convert MP3 to WAV using librosa"""
    try:
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_wav_path = temp_wav.name
        temp_wav.close()
        
        y, sr = librosa.load(input_file, sr=target_sr, mono=mono)
        y = y.astype('float32')
        
        sf.write(temp_wav_path, y, target_sr, subtype='PCM_16')
        
        return temp_wav_path, y, sr
    except Exception as e:
        st.error(f"Error converting audio: {str(e)}")
        return None, None, None


def normalize_audio(x):
    """Normalize audio signal"""
    return x / (np.max(np.abs(x)) + 1e-9)


def bandpass_filter(signal, sr, lowcut, highcut, order=4):
    """Apply bandpass filter to signal"""
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    
    # Ensure frequencies are within valid range (0 < Wn < 1)
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))
    
    # Ensure low < high
    if low >= high:
        high = min(low + 0.1, 0.999)
    
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal)
    return y


def create_audio_mixtures(y, sr, n_components=2):
    """Create synthetic audio mixtures"""
    mixtures_list = []
    
    # Define frequency bands based on sample rate
    nyquist = sr / 2
    bands = [
        (50, min(8000, nyquist * 0.7)),
        (200, min(10000, nyquist * 0.9))
    ]
    
    for i in range(n_components):
        low, high = bands[i % len(bands)]
        
        if high >= nyquist:
            high = nyquist * 0.95
        if low >= high:
            low = high * 0.1
        
        filtered = bandpass_filter(y, sr, low, high, order=4)
        
        delay = int(0.001 * sr * i)
        if delay > 0:
            kernel = np.zeros(delay + 1)
            kernel[-1] = 1.0
            filtered = fftconvolve(filtered, kernel, mode='same')
        
        mixtures_list.append(filtered)
    
    mixtures = np.vstack(mixtures_list).T.astype(np.float64)
    mixtures = mixtures - np.mean(mixtures, axis=0)
    
    return mixtures


def plot_audio_waveform(signal, sr, title):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(signal, sr=sr, ax=ax, color='#17a2b8')
    ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_audio_spectrogram(signal, sr, title):
    """Plot audio spectrogram"""
    fig, ax = plt.subplots(figsize=(10, 4))
    S = np.abs(librosa.stft(signal, n_fft=2048, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, hop_length=512, 
                                    x_axis='time', y_axis='log', ax=ax, cmap='viridis')
    ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig


def process_audio_ica_pca(mixtures, sr, n_components=2):
    """Apply ICA and PCA to audio mixtures"""
    ica = FastICA(n_components=n_components, random_state=0, max_iter=2000)
    S_ica = ica.fit_transform(mixtures)
    
    S_ica_norm = np.zeros_like(S_ica)
    for i in range(S_ica.shape[1]):
        S_ica_norm[:, i] = normalize_audio(S_ica[:, i])
    
    pca = PCA(n_components=n_components)
    S_pca = pca.fit_transform(mixtures)
    
    S_pca_norm = np.zeros_like(S_pca)
    for i in range(S_pca.shape[1]):
        S_pca_norm[:, i] = normalize_audio(S_pca[:, i])
    
    return S_ica_norm, S_pca_norm, ica, pca


# ================== IMAGE FUNCTIONS ==================

def load_and_preprocess_image(uploaded_file, size=(256, 256)):
    """Load and preprocess image"""
    try:
        img = skio.imread(uploaded_file)
        
        if img.ndim == 3:
            img = color.rgb2gray(img)
        
        img = transform.resize(img, size, anti_aliasing=True)
        img = (img - img.min()) / (img.max() - img.min() + 1e-10)
        
        return img
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None


def normalize_component(x):
    """Normalize to [0, 1] range"""
    x = x - np.min(x)
    x = x / (np.max(x) + 1e-10)
    return x


def create_image_mixtures(originals, n_components=3):
    """Create mixed images"""
    S = np.array([img.flatten() for img in originals])
    
    np.random.seed(42)
    A = np.random.randn(n_components, n_components)
    
    X = np.dot(A, S)
    
    return X, A, S


def process_image_ica_pca(X, img_size=(256, 256), n_components=3):
    """Apply ICA and PCA to mixed images"""
    X_T = X.T
    
    ica = FastICA(n_components=n_components, random_state=0, max_iter=2000, tol=0.001)
    S_ica_T = ica.fit_transform(X_T)
    S_ica = S_ica_T.T
    
    pca = PCA(n_components=n_components)
    S_pca_T = pca.fit_transform(X_T)
    S_pca = S_pca_T.T
    
    ica_images = [normalize_component(s.reshape(img_size)) for s in S_ica]
    pca_images = [normalize_component(s.reshape(img_size)) for s in S_pca]
    
    return ica_images, pca_images, ica, pca


# ================== STREAMLIT UI ==================

def main():
    # Sidebar with deployment and help info
    with st.sidebar:
        st.markdown("### 📋 Information")
        
        with st.expander("🚀 Deployment Options", expanded=False):
            st.markdown("""
            **Deploy to Streamlit Cloud:**
            1. Push your repo to GitHub
            2. Go to [share.streamlit.io](https://share.streamlit.io)
            3. Sign in with GitHub
            4. Click "New app" and select your repository
            5. Your app will be live at: `https://[username]-[repo]-[branch].streamlit.app`
            
            **Deploy Locally:**
            ```bash
            streamlit run main.py
            ```
            """)
        
        with st.expander("❓ How to Use", expanded=False):
            st.markdown("""
            **Audio Analysis:**
            - Upload MP3 or WAV file
            - Adjust components (2-4)
            - View ICA and PCA results
            - Download separated audio
            
            **Image Analysis:**
            - Upload exactly 3 images
            - Choose image size (128-512px)
            - Compare separation results
            - Download separated images
            """)
        
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
            **ICA vs PCA Analysis Tool**
            
            This application demonstrates blind source separation using:
            - **ICA**: Independent Component Analysis
            - **PCA**: Principal Component Analysis
            
            Works with both audio and image data.
            """)
        
        st.markdown("---")
        st.markdown("Made with ❤️ using Streamlit")
    
    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🎵 ICA vs PCA Analysis 🖼️</div>
        <div class="hero-subtitle">Blind Source Separation for Audio and Images</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode Selector with Dropdown
    st.markdown('<div class="mode-selector-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🎯 Select Analysis Mode")
        analysis_mode = st.selectbox(
            "",
            ["🎵 Audio Source Separation", "🖼️ Image Source Separation"],
            index=0,
            label_visibility="collapsed"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Info card about methods
    st.markdown("""
    <div class="info-card">
        <h4 style="color: #0d7f8f; margin-top: 0; font-size: 1.2rem;">📚 About the Methods</h4>
        <p style="margin-bottom: 1rem; color: #2c3e50;"><strong>🔵 ICA (Independent Component Analysis):</strong> Separates mixed signals into independent components - ideal for blind source separation. Works best when signals have non-Gaussian distributions.</p>
        <p style="margin-bottom: 0; color: #2c3e50;"><strong>🔴 PCA (Principal Component Analysis):</strong> Finds orthogonal directions of maximum variance - better for dimensionality reduction and data compression.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Route to appropriate interface
    if analysis_mode == "🎵 Audio Source Separation":
        audio_analysis_ui()
    else:
        image_analysis_ui()


def audio_analysis_ui():
    """Audio analysis interface"""
    st.markdown('<h2 class="section-header">🎵 Audio Source Separation</h2>', unsafe_allow_html=True)
    
    # Parameters in columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📁 Upload Audio File (MP3 or WAV)",
            type=['mp3', 'wav'],
            help="Upload an audio file for ICA/PCA analysis"
        )
    
    with col2:
        st.markdown("### ⚙️ Parameters")
        n_components = st.slider("Components", 2, 4, 2, help="Number of components to separate")
        target_sr = st.selectbox("Sample Rate", [22050, 44100], index=0, help="Audio sample rate in Hz")
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        st.markdown('<div class="success-card">✅ <strong>File uploaded successfully!</strong></div>', 
                    unsafe_allow_html=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Convert if needed
            if uploaded_file.name.endswith('.mp3'):
                with st.spinner("🔄 Converting MP3 to WAV..."):
                    wav_path, y, sr = convert_mp3_to_wav(tmp_file_path, target_sr=target_sr, mono=True)
                    if wav_path is None:
                        return
                st.markdown('<div class="success-card">✅ Conversion complete!</div>', unsafe_allow_html=True)
            else:
                y, sr = librosa.load(tmp_file_path, sr=target_sr, mono=True)
            
            # Original audio
            st.markdown('<h3 class="sub-section-header">📊 Original Audio</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.audio(tmp_file_path)
                st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #17a2b8; margin: 0;">Audio Info</h4>
                    <p style="margin: 0.5rem 0; color: #2c3e50;"><strong>Sample Rate:</strong> {sr} Hz</p>
                    <p style="margin: 0; color: #2c3e50;"><strong>Duration:</strong> {len(y)/sr:.2f}s</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                fig_wave = plot_audio_waveform(y, sr, "Original Waveform")
                st.pyplot(fig_wave)
                plt.close()
            
            fig_spec = plot_audio_spectrogram(y, sr, "Original Spectrogram")
            st.pyplot(fig_spec)
            plt.close()
            
            # Create mixtures
            st.markdown('<h3 class="sub-section-header">🔀 Creating Mixtures</h3>', unsafe_allow_html=True)
            with st.spinner("Processing..."):
                mixtures = create_audio_mixtures(y, sr, n_components)
            
            st.markdown(f'<div class="success-card">✅ Created {n_components} mixture signals</div>', 
                       unsafe_allow_html=True)
            
            # Display mixtures
            with st.expander("📈 View Mixed Signals", expanded=False):
                for i in range(n_components):
                    st.markdown(f"#### Mixture {i+1}")
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = plot_audio_waveform(mixtures[:, i], sr, f"Mixture {i+1} Waveform")
                        st.pyplot(fig)
                        plt.close()
                    with col2:
                        fig = plot_audio_spectrogram(mixtures[:, i], sr, f"Mixture {i+1} Spectrogram")
                        st.pyplot(fig)
                        plt.close()
            
            # Apply ICA and PCA
            st.markdown('<h3 class="sub-section-header">🧮 Applying ICA and PCA</h3>', unsafe_allow_html=True)
            with st.spinner("🔄 Separating sources..."):
                S_ica, S_pca, ica_model, pca_model = process_audio_ica_pca(mixtures, sr, n_components)
            
            st.markdown('<div class="success-card">✅ <strong>Separation complete!</strong></div>', 
                       unsafe_allow_html=True)
            
            # Results tabs
            tab1, tab2, tab3 = st.tabs(["🔵 ICA Results", "🔴 PCA Results", "📊 Comparison"])
            
            with tab1:
                st.markdown("### ICA Separated Components")
                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
                
                for i in range(n_components):
                    st.markdown(f"#### 🎵 Component {i+1}")
                    
                    temp_ica = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    sf.write(temp_ica.name, S_ica[:, i], sr)
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.audio(temp_ica.name)
                        rms = np.sqrt(np.mean(S_ica[:, i]**2))
                        st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #17a2b8; margin: 0;">RMS Energy</h4>
                            <p style="font-size: 1.5rem; font-weight: bold; color: #0d7f8f; margin: 0.5rem 0 0 0;">{rms:.6f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
                        st.download_button(
                            label=f"⬇️ Download Component {i+1}",
                            data=open(temp_ica.name, 'rb').read(),
                            file_name=f"ica_component_{i+1}.wav",
                            mime="audio/wav"
                        )
                    
                    with col2:
                        fig = plot_audio_waveform(S_ica[:, i], sr, f"ICA Component {i+1}")
                        st.pyplot(fig)
                        plt.close()
                    
                    fig_spec = plot_audio_spectrogram(S_ica[:, i], sr, f"ICA Component {i+1} Spectrogram")
                    st.pyplot(fig_spec)
                    plt.close()
                    
                    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
                    if i < n_components - 1:
                        st.markdown("---")
            
            with tab2:
                st.markdown("### PCA Separated Components")
                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
                
                for i in range(n_components):
                    st.markdown(f"#### 🎵 Component {i+1}")
                    
                    temp_pca = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    sf.write(temp_pca.name, S_pca[:, i], sr)
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.audio(temp_pca.name)
                        rms = np.sqrt(np.mean(S_pca[:, i]**2))
                        st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #dc3545; margin: 0;">RMS Energy</h4>
                            <p style="font-size: 1.5rem; font-weight: bold; color: #c82333; margin: 0.5rem 0 0 0;">{rms:.6f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
                        st.download_button(
                            label=f"⬇️ Download Component {i+1}",
                            data=open(temp_pca.name, 'rb').read(),
                            file_name=f"pca_component_{i+1}.wav",
                            mime="audio/wav"
                        )
                    
                    with col2:
                        fig = plot_audio_waveform(S_pca[:, i], sr, f"PCA Component {i+1}")
                        st.pyplot(fig)
                        plt.close()
                    
                    fig_spec = plot_audio_spectrogram(S_pca[:, i], sr, f"PCA Component {i+1} Spectrogram")
                    st.pyplot(fig_spec)
                    plt.close()
                    
                    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
                    if i < n_components - 1:
                        st.markdown("---")
            
            with tab3:
                st.markdown("### 📊 Statistical Comparison")
                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🔵 ICA Statistics")
                    ica_rms = [np.sqrt(np.mean(S_ica[:, i]**2)) for i in range(n_components)]
                    for i, rms in enumerate(ica_rms):
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #17a2b8; margin: 0;">Component {i+1} RMS</h4>
                            <p style="font-size: 1.3rem; font-weight: bold; margin: 0.5rem 0 0 0;">{rms:.6f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### 🔴 PCA Statistics")
                    for i, var in enumerate(pca_model.explained_variance_ratio_):
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #dc3545; margin: 0;">Component {i+1} Variance</h4>
                            <p style="font-size: 1.3rem; font-weight: bold; margin: 0.5rem 0 0 0;">{var:.4f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Comparison chart
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                axes[0].bar(range(1, n_components+1), ica_rms, color='#17a2b8', alpha=0.8, edgecolor='#0d7f8f', linewidth=2)
                axes[0].set_title('ICA Component RMS Energy', fontsize=14, fontweight='bold', color='#2c3e50')
                axes[0].set_xlabel('Component', fontsize=12)
                axes[0].set_ylabel('RMS Energy', fontsize=12)
                axes[0].grid(True, alpha=0.3)
                
                axes[1].bar(range(1, n_components+1), pca_model.explained_variance_ratio_, 
                           color='#dc3545', alpha=0.8, edgecolor='#c82333', linewidth=2)
                axes[1].set_title('PCA Explained Variance Ratio', fontsize=14, fontweight='bold', color='#2c3e50')
                axes[1].set_xlabel('Component', fontsize=12)
                axes[1].set_ylabel('Variance Ratio', fontsize=12)
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        except Exception as e:
            st.error(f"❌ Error processing audio: {str(e)}")
        
        finally:
            try:
                os.unlink(tmp_file_path)
            except:
                pass


def image_analysis_ui():
    """Image analysis interface"""
    st.markdown('<h2 class="section-header">🖼️ Image Source Separation</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4 style="color: #0d7f8f; margin-top: 0; font-size: 1.1rem;">📌 Instructions</h4>
        <p style="margin-bottom: 0; color: #2c3e50;">Upload exactly 3 images for analysis. The system will mix them and then separate using ICA and PCA.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Parameters in columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "📁 Upload 3 Images (JPG, PNG, etc.)",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload exactly 3 images for ICA/PCA analysis"
        )
    
    with col2:
        st.markdown("### ⚙️ Parameters")
        img_size = st.selectbox("Image Size", [128, 256, 512], index=1, help="Processing image size")
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    if uploaded_files:
        if len(uploaded_files) != 3:
            st.markdown(f"""
            <div class="warning-card">
                ⚠️ <strong>Warning:</strong> Please upload exactly 3 images. You uploaded {len(uploaded_files)}.
            </div>
            """, unsafe_allow_html=True)
            return
        
        st.markdown('<div class="success-card">✅ <strong>3 images uploaded successfully!</strong></div>', 
                    unsafe_allow_html=True)
        
        try:
            # Load images
            st.markdown('<h3 class="sub-section-header">📸 Original Images</h3>', unsafe_allow_html=True)
            st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
            
            originals = []
            
            cols = st.columns(3)
            for i, (uploaded_file, col) in enumerate(zip(uploaded_files, cols)):
                img = load_and_preprocess_image(uploaded_file, size=(img_size, img_size))
                if img is None:
                    return
                originals.append(img)
                
                with col:
                    st.image(img, caption=f"Original {i+1}", width='stretch', use_column_width=True)
            
            st.markdown("<div style='margin: 2.5rem 0;'></div>", unsafe_allow_html=True)
            
            st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
            
            # Create mixtures
            st.markdown('<h3 class="sub-section-header">🔀 Creating Mixed Images</h3>', unsafe_allow_html=True)
            with st.spinner("🔄 Creating mixtures..."):
                X, A, S = create_image_mixtures(originals, n_components=3)
                mixed_images = [normalize_component(x.reshape(img_size, img_size)) for x in X]
            
            st.markdown('<div class="success-card">✅ Created 3 mixed images</div>', unsafe_allow_html=True)
            
            st.markdown('<h3 class="sub-section-header">🎭 Mixed Images</h3>', unsafe_allow_html=True)
            cols = st.columns(3)
            for i, (mixed_img, col) in enumerate(zip(mixed_images, cols)):
                with col:
                    st.image(mixed_img, caption=f"Mixed {i+1}", width='stretch', use_column_width=True)
            
            st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
            
            # Show mixing matrix
            with st.expander("🔢 View Mixing Matrix", expanded=False):
                st.write("**Mixing Matrix A:**")
                st.dataframe(A, width='stretch', use_container_width=True)
            
            # Apply ICA and PCA
            st.markdown('<h3 class="sub-section-header">🧮 Applying ICA and PCA</h3>', unsafe_allow_html=True)
            with st.spinner("🔄 Separating sources..."):
                ica_images, pca_images, ica_model, pca_model = process_image_ica_pca(
                    X, img_size=(img_size, img_size), n_components=3
                )
            
            st.markdown('<div class="success-card">✅ <strong>Separation complete!</strong></div>', 
                       unsafe_allow_html=True)
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["🔵 ICA Results", "🔴 PCA Results", "📊 Complete Comparison"])
            
            with tab1:
                st.markdown("### ICA Separated Components")
                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
                
                cols = st.columns(3)
                for i, (img, col) in enumerate(zip(ica_images, cols)):
                    with col:
                        st.image(img, caption=f"ICA Component {i+1}", width='stretch', use_column_width=True)
                        
                        # Convert to bytes for download
                        img_byte_arr = io.BytesIO()
                        plt.imsave(img_byte_arr, img, cmap='gray', format='png')
                        img_byte_arr.seek(0)
                        
                        st.download_button(
                            label=f"⬇️ Download",
                            data=img_byte_arr,
                            file_name=f"ica_component_{i+1}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                
                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #17a2b8; margin: 0;">ICA Statistics</h4>
                    <p style="margin: 0.5rem 0 0 0; color: #2c3e50;"><strong>Convergence:</strong> {ica_model.n_iter_} iterations</p>
                </div>
                """, unsafe_allow_html=True)
            
            with tab2:
                st.markdown("### PCA Separated Components")
                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
                
                cols = st.columns(3)
                for i, (img, col) in enumerate(zip(pca_images, cols)):
                    with col:
                        st.image(img, caption=f"PCA Component {i+1}", width='stretch', use_column_width=True)
                        
                        # Convert to bytes for download
                        img_byte_arr = io.BytesIO()
                        plt.imsave(img_byte_arr, img, cmap='gray', format='png')
                        img_byte_arr.seek(0)
                        
                        st.download_button(
                            label=f"⬇️ Download",
                            data=img_byte_arr,
                            file_name=f"pca_component_{i+1}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                
                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
                st.markdown("#### 📈 PCA Statistics")
                st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
                
                cols = st.columns(3)
                for i, (var, col) in enumerate(zip(pca_model.explained_variance_ratio_, cols)):
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #dc3545; margin: 0;">Component {i+1}</h4>
                            <p style="font-size: 1.3rem; font-weight: bold; margin: 0.5rem 0 0 0;">{var:.4f}</p>
                            <p style="font-size: 0.9rem; color: #666; margin: 0;">Variance Ratio</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab3:
                st.markdown("### 📊 Complete Visual Comparison")
                
                # Create comparison grid
                fig, axes = plt.subplots(4, 3, figsize=(14, 18))
                
                # Row 1: Originals
                for i in range(3):
                    axes[0, i].imshow(originals[i], cmap='gray', vmin=0, vmax=1)
                    axes[0, i].set_title(f'Original {i+1}', fontweight='bold', fontsize=12, color='#2c3e50')
                    axes[0, i].axis('off')
                
                # Row 2: Mixed
                for i in range(3):
                    axes[1, i].imshow(mixed_images[i], cmap='gray', vmin=0, vmax=1)
                    axes[1, i].set_title(f'Mixed {i+1}', fontweight='bold', fontsize=12, color='#2c3e50')
                    axes[1, i].axis('off')
                
                # Row 3: ICA
                for i in range(3):
                    axes[2, i].imshow(ica_images[i], cmap='gray', vmin=0, vmax=1)
                    axes[2, i].set_title(f'ICA Component {i+1}', fontweight='bold', fontsize=12, color='#17a2b8')
                    axes[2, i].axis('off')
                
                # Row 4: PCA
                for i in range(3):
                    axes[3, i].imshow(pca_images[i], cmap='gray', vmin=0, vmax=1)
                    axes[3, i].set_title(f'PCA Component {i+1}', fontweight='bold', fontsize=12, color='#dc3545')
                    axes[3, i].axis('off')
                
                # Add labels
                fig.text(0.02, 0.875, 'Original', fontsize=13, fontweight='bold', va='center', rotation=90, color='#2c3e50')
                fig.text(0.02, 0.625, 'Mixed', fontsize=13, fontweight='bold', va='center', rotation=90, color='#2c3e50')
                fig.text(0.02, 0.375, 'ICA', fontsize=13, fontweight='bold', va='center', rotation=90, color='#17a2b8')
                fig.text(0.02, 0.125, 'PCA', fontsize=13, fontweight='bold', va='center', rotation=90, color='#dc3545')
                
                plt.suptitle('ICA vs PCA for Blind Source Separation', fontsize=18, fontweight='bold', y=0.995, color='#2c3e50')
                plt.tight_layout(rect=[0.05, 0, 1, 0.99])
                
                st.pyplot(fig)
                plt.close()
                
                # Analysis
                st.markdown("### 📝 Detailed Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="info-card">
                        <h4 style="color: #0d7f8f; margin-top: 0; font-size: 1.1rem;">🔵 ICA (Independent Component Analysis)</h4>
                        <ul style="margin-bottom: 0; color: #2c3e50;">
                            <li><strong style="color: #17a2b8;">✅ Better for blind source separation</strong></li>
                            <li>Recovers statistically independent sources</li>
                            <li>Components should resemble original images</li>
                            <li>⚠️ Order may differ from originals</li>
                            <li>Non-Gaussian assumption</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="info-card">
                        <h4 style="color: #0d7f8f; margin-top: 0; font-size: 1.1rem;">🔴 PCA (Principal Component Analysis)</h4>
                        <ul style="margin-bottom: 0; color: #2c3e50;">
                            <li><strong style="color: #dc3545;">✅ Good for dimensionality reduction</strong></li>
                            <li>Finds orthogonal variance directions</li>
                            <li>Maximizes explained variance</li>
                            <li>⚠️ Often produces averaged/blurred components</li>
                            <li>⚠️ Not optimal for source separation</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Performance metrics
                st.markdown("### 📊 Performance Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #17a2b8; margin: 0;">ICA Iterations</h4>
                        <p style="font-size: 2rem; font-weight: bold; color: #0d7f8f; margin: 0.5rem 0;">{ica_model.n_iter_}</p>
                        <p style="font-size: 0.9rem; color: #666; margin: 0;">Convergence steps</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    total_var = sum(pca_model.explained_variance_ratio_)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #dc3545; margin: 0;">Total Variance</h4>
                        <p style="font-size: 2rem; font-weight: bold; color: #c82333; margin: 0.5rem 0;">{total_var:.4f}</p>
                        <p style="font-size: 0.9rem; color: #666; margin: 0;">PCA explained</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #28a745; margin: 0;">Components</h4>
                        <p style="font-size: 2rem; font-weight: bold; color: #218838; margin: 0.5rem 0;">3</p>
                        <p style="font-size: 0.9rem; color: #666; margin: 0;">Separated</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Error processing images: {str(e)}")
            import traceback
            st.error(traceback.format_exc())


if __name__ == "__main__":
    main()