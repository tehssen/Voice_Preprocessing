# Voice_Preprocessing

# Speech Preprocessing with Librosa

This repository contains a Python script for speech preprocessing, feature extraction, and visualization using the Librosa library. The code demonstrates loading audio samples, applying preprocessing steps, extracting MFCC (Mel-Frequency Cepstral Coefficients) features, and combining them into a tensor for potential use in machine learning tasks like speech recognition or classification.

## Description

The script performs the following key operations:
- **Audio Loading**: Loads two example audio files from Librosa's built-in examples (nutcracker and vibeace).
- **Preprocessing Pipeline**: Applies resampling to 16 kHz, normalization, silence trimming, and a simple smoothing filter (moving average).
- **Visualization**: Plots raw vs. processed waveforms for comparison.
- **Feature Extraction**: Computes MFCCs for each processed audio and visualizes them as spectrograms.
- **Feature Combination**: Trims MFCCs to the minimum length and stacks them into a NumPy tensor.
- **Bonus Experiment**: Adds Gaussian noise to one processed audio and visualizes the result.

This is a self-contained example script, ideal for educational purposes or as a starting point for audio processing projects.

## Requirements

- Python 3.7+
- Libraries:
  - `librosa` (for audio processing)
  - `numpy` (for numerical operations)
  - `matplotlib` (for plotting)

Install dependencies using pip:
```bash
pip install librosa numpy matplotlib
```

Note: Librosa may require additional system dependencies like FFmpeg for audio handling. Refer to the [Librosa installation guide](https://librosa.org/doc/latest/install.html) for details.

## Usage

1. Clone or download the script.
2. Run the script in a Python environment (e.g., Jupyter Notebook, Python interpreter, or IDE like VS Code).
   ```bash
   python speech_preprocessing.py
   ```
   Replace `speech_preprocessing.py` with the actual filename.

3. The script will:
   - Load and process the audio samples.
   - Display plots (waveforms and MFCC spectrograms). Ensure your environment supports matplotlib plotting (e.g., in Jupyter, use `%matplotlib inline`).
   - Print the shape and value range of the combined features tensor.
   - Show a bonus noisy audio waveform.

## Code Explanation

### Audio Loading
- Uses `librosa.ex()` to load example audio files internally (no external downloads required).
- Loads two samples: 'nutcracker' and 'vibeace', storing them as tuples of (audio array, sample rate).

### Preprocessing Functions
- `normalize(audio)`: Scales the audio to have a maximum absolute value of 1.
- `smoothing_filter(audio, kernel_size=5)`: Applies a simple moving average filter to smooth the audio signal.

### Preprocessing Pipeline
- Resamples each audio to 16 kHz using `librosa.resample`.
- Normalizes the audio.
- Trims silence using `librosa.effects.trim` (removes segments below 20 dB).
- Applies the smoothing filter.
- Stores processed audios in a list.

### Visualization
- Plots raw and processed waveforms side-by-side using `librosa.display.waveshow`.
- Displays MFCC spectrograms with `librosa.display.specshow`, using a 'magma' colormap.

### Feature Extraction
- Computes 13 MFCC coefficients for each processed audio using `librosa.feature.mfcc`.
- Visualizes each MFCC as a spectrogram.

### Feature Combination
- Trims all MFCC arrays to the minimum length across samples.
- Stacks them into a 3D NumPy array (shape: [num_samples, n_mfcc, time_steps]).
- Prints the tensor shape and value range.

### Bonus: Noise Experiment
- Generates Gaussian noise and adds it to the first processed audio.
- Visualizes the noisy waveform.

## Output

- **Plots**: Waveforms (raw/processed) and MFCC spectrograms. The bonus plot shows the noisy audio.
- **Console Output**:
  ```
  Features Tensor Shape: (2, 13, min_time_steps)
  Value Range: min_value max_value
  ```
  (Actual values depend on the audio samples.)

## Notes

- The script uses Librosa's example files, which are short clips. For real-world use, replace with your own audio files.
- MFCC parameters (e.g., n_mfcc=13) can be adjusted for different applications.
- The smoothing filter is basic; for advanced filtering, consider using SciPy or other libraries.
- This code is for demonstration and may need optimization for large datasets or production use.

## License

This code is provided as-is for educational purposes. Feel free to modify and distribute under an open-source license (e.g., MIT).
