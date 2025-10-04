import streamlit as st  # Untuk UI web (textbox, tombol, output)
import librosa  # Untuk analisis audio (chroma, key, chord)
import numpy as np  # Untuk perhitungan matriks
import yt_dlp  # Untuk download audio dari YouTube
import os  # Untuk handle file

# Judul dan instruksi app
st.title("Chord dan Nada Dasar Detector")
st.write("Masukkan link YouTube lagu favoritmu. App akan deteksi nada dasar dan chord progression. Cocok untuk lagu sederhana (pop/akustik). Proses 1-3 menit.")

# Textbox untuk input link YouTube
youtube_url = st.text_input("Link YouTube:", placeholder="https://www.youtube.com/watch?v=...")

# Tombol analisis
analyze_button = st.button("Analyze")

if analyze_button and youtube_url:
    with st.spinner("Downloading dan analisis... Tunggu sebentar!"):  # Loading indicator
        try:
            # Download audio dari YouTube (konfigurasikan untuk MP3 berkualitas sedang)
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': 'audio.%(ext)s',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            st.success("Audio berhasil didownload!")

            # Load audio
            audio_path = [f for f in os.listdir('.') if f.startswith('audio.')][0]
            y, sr = librosa.load(audio_path, sr=None)

            # Ekstrak chroma (fitur untuk nada/chord)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

            # Deteksi nada dasar
            key = librosa.key.estimate_key(chroma)
            key_name = librosa.key_to_string(key)
            st.subheader("Hasil Nada Dasar")
            st.write(f"**Nada Dasar**: {key_name}")

            # Template chord (tambahkan lebih banyak kalau perlu untuk akurasi)
            chord_templates = {
                'C': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
                'Cm': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
                'G': np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]),
                'Gm': np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]),
                'Am': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
                'F': np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),
                'Dm': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
                'D': np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]),
                'Em': np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
                # Tambah contoh: 'A': np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
            }

            # Deteksi chord
            chords = []
            for t in range(chroma.shape[1]):
                frame = chroma[:, t]
                best_chord = 'Unknown'
                max_corr = -1
                for chord, template in chord_templates.items():
                    frame_norm = frame / np.linalg.norm(frame) if np.linalg.norm(frame) > 0 else frame
                    template_norm = template / np.linalg.norm(template)
                    corr = np.dot(frame_norm, template_norm)
                    if corr > max_corr:
                        max_corr = corr
                        best_chord = chord
                chords.append(best_chord)

            # Tampilkan chord (ringkas, hanya saat berubah)
            times = librosa.times_like(chroma, sr=sr)
            st.subheader("Chord Progression")
            prev_chord = None
            chord_output = ""
            for t, chord in zip(times, chords):
                if chord != prev_chord:
                    chord_output += f"Waktu {t:.2f}s: Chord {chord}\n"
                    prev_chord = chord
            st.text(chord_output)

            # Cleanup file
            os.remove(audio_path)

        except Exception as e:
            st.error(f"Error: {e}")
            st.write("Cek link YouTube (harus publik). Coba lagi atau lagu lain.")
