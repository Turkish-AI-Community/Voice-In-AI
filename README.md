# 🎙️ Sıfırdan İleri Seviye: Ses Teknolojileri & AI ile Voice/Speech Eğitim Planı

> **Hedef:** Sesin fiziğinden başlayarak DSP, ses işleme, özellik mühendisliği, derin öğrenme tabanlı ses modelleri, ASR, TTS, VAD, Voice Cloning, Foundation Models ve MLOps'a kadar kapsamlı, açık kaynaklı, all-in-one eğitim reposu.

---

## 📁 Repo Yapısı

```
audio-ai-course/
│
├── README.md
├── requirements.txt
├── environment.yml
│
├── egitim/
│   ├── 01_ses_fizigi/
│   ├── 02_dsp_temelleri/
│   ├── 03_ses_isleme/
│   ├── 04_ozellik_muhendisligi/
│   ├── 05_klasik_ml_ses/
│   ├── 06_derin_ogrenme_ses/
│   ├── 07_vad/
│   ├── 08_asr_stt/
│   ├── 09_tts/
│   ├── 10_voice_cloning/
│   ├── 11_speaker_diarization/
│   ├── 12_ses_siniflandirma/
│   ├── 13_muzik_ses/
│   ├── 14_foundation_models/
│   ├── 15_multimodal_modeller/
│   └── 16_mlops_deployment/
│
├── modeller-makaleler/
│   ├── asr/
│   ├── tts/
│   ├── vad/
│   ├── voice_cloning/
│   ├── foundation_models/
│   ├── multimodal/
│   └── klasik_yontemler/
│
└── teknolojiler/
    ├── kutuphaneler/
    ├── framework_karsilastirma/
    ├── deployment_araclari/
    └── veri_setleri/
```

---

## 🎯 Eğitim Modülleri

---

### MODÜL 1 — Ses Fiziği ve Temel Kavramlar
**Klasör:** `egitim/01_ses_fizigi/`

#### Konular
- Ses nedir? Mekanik dalga, basınç dalgası
- Frekans, genlik, faz, dalga boyu
- İnsan işitme sistemi ve algı (psikоакустik giriş)
- Desibel (dB) ve ses şiddeti
- Mono, Stereo, Multichannel ses
- Dijital ses temelleri: örnekleme (sampling), kuantalama (quantization)
- Nyquist-Shannon örnekleme teoremi
- Bit derinliği ve dinamik aralık
- Ses formatları: WAV, MP3, FLAC, OGG, AAC — farklar ve kayıp/kayıpsız sıkıştırma

#### Dosyalar
```
01_ses_fizigi/
├── 01_ses_nedir.ipynb
├── 02_dijital_ses_temelleri.ipynb
├── 03_ses_formatlari.ipynb
└── assets/
```

#### İlgili Makaleler / Kaynaklar
- "The Science of Sound" — Rossing, Moore, Wheeler
- Julius O. Smith III — "Physical Audio Signal Processing"

---

### MODÜL 2 — DSP Temelleri (Dijital Sinyal İşleme)
**Klasör:** `egitim/02_dsp_temelleri/`

#### Konular
- Zaman domeninde sinyal analizi
- Fourier Dönüşümü (FT) — sezgisel ve matematiksel
- Ayrık Fourier Dönüşümü (DFT) ve FFT algoritması
- Frekans domeninde sinyal gösterimi
- Kısa Zamanlı Fourier Dönüşümü (STFT) ve Spektrogram
- Pencere fonksiyonları: Hann, Hamming, Blackman, Kaiser
- Konvolüsyon ve korelasyon
- FIR ve IIR filtreler
- Bant geçiren, alçak geçiren, yüksek geçiren filtreler
- Faz ve grup gecikmesi
- Z-dönüşümü (giriş seviyesi)
- Ters STFT ve ses sentezi

#### Dosyalar
```
02_dsp_temelleri/
├── 01_fourier_donusumu.ipynb
├── 02_fft_ve_spektrogram.ipynb
├── 03_pencere_fonksiyonlari.ipynb
├── 04_filtreler.ipynb
├── 05_stft_ve_ters_stft.ipynb
└── dsp_utils.py
```

#### İlgili Makaleler / Kaynaklar
- Oppenheim & Schafer — "Discrete-Time Signal Processing"
- Steven W. Smith — "The Scientist and Engineer's Guide to DSP"

---

### MODÜL 3 — Ses İşleme ve Manipülasyon
**Klasör:** `egitim/03_ses_isleme/`

#### Konular
- Ses okuma/yazma: librosa, soundfile, scipy.io.wavfile
- Ses görselleştirme: dalga formu, spektrogram, mel-spektrogram
- Ses kırpma, birleştirme, hız değiştirme
- Pitch shifting ve time stretching
- Gürültü ekleme ve azaltma (spectral subtraction, Wiener filter)
- Oda akustiği ve reverb — Room Impulse Response (RIR)
- Echo ve delay efektleri
- Normalizasyon ve ses dengeleme
- Veri artırma (data augmentation): SpecAugment, MixUp, noise injection
- Audio denoising: klasik yöntemler vs. derin öğrenme

#### Dosyalar
```
03_ses_isleme/
├── 01_ses_okuma_yazma.ipynb
├── 02_ses_manipulasyon.ipynb
├── 03_gurultu_azaltma.ipynb
├── 04_veri_artirma.ipynb
├── audio_utils.py
└── samples/
```

#### İlgili Kütüphaneler
- `librosa`, `soundfile`, `pydub`, `pedalboard`, `noisereduce`

---

### MODÜL 4 — Özellik Mühendisliği (Audio Feature Engineering)
**Klasör:** `egitim/04_ozellik_muhendisligi/`

#### Konular
- **Zaman domeninde özellikler:**
  - Zero Crossing Rate (ZCR)
  - RMS Enerji, Short-Time Energy
  - Autocorrelation
  - Temporal envelope

- **Frekans domeninde özellikler:**
  - Spektral centroid, bandwidth, rolloff, flatness
  - Spectral contrast
  - Harmonics-to-Noise Ratio (HNR)
  - Fundamental frequency (F0) — pitch tracking

- **Mel ölçeği ve insan işitme:**
  - Mel ölçeği neden var?
  - Mel Filtre Bankası
  - Mel Spektrogram
  - MFCC (Mel Frequency Cepstral Coefficients) — adım adım
  - Delta ve Delta-Delta MFCC

- **Gelişmiş özellikler:**
  - Chroma özellikleri (müzik uygulamaları)
  - Tonnetz
  - CQCC (Constant-Q Cepstral Coefficients)
  - RASTA-PLP
  - Gammatone özellikleri (biyolojik işitmeye dayalı)
  - Linear Prediction Coefficients (LPC) ve LPCC

- **Öğrenilmiş özellikler vs. elle tasarlanmış özellikler:**
  - CNN tabanlı özellik öğrenimi
  - wav2vec, HuBERT embeddings

#### Dosyalar
```
04_ozellik_muhendisligi/
├── 01_zaman_domeni_ozellikler.ipynb
├── 02_frekans_domeni_ozellikler.ipynb
├── 03_mel_mfcc.ipynb
├── 04_pitch_tracking.ipynb
├── 05_gelismis_ozellikler.ipynb
├── 06_ogrenilmis_vs_elle_tasarlanmis.ipynb
└── feature_extraction.py
```

---

### MODÜL 5 — Klasik ML ile Ses Uygulamaları
**Klasör:** `egitim/05_klasik_ml_ses/`

#### Konular
- Ses sınıflandırma için klasik pipeline
- Gaussian Mixture Models (GMM) — konuşmacı tanıma temeli
- Hidden Markov Models (HMM) — ASR'nin tarihsel temeli
- GMM-HMM hibrit sistemler
- Support Vector Machines (SVM) ile ses sınıflandırma
- k-NN ve karar ağaçları ses uygulamaları
- DTW (Dynamic Time Warping) — konuşma hizalama
- K-means ile ses kümelemeleme
- Değerlendirme metrikleri: accuracy, WER, EER, DER

#### Dosyalar
```
05_klasik_ml_ses/
├── 01_gmm_konusmacu_tanima.ipynb
├── 02_hmm_asr_temeli.ipynb
├── 03_svm_ses_siniflandirma.ipynb
├── 04_dtw_hizalama.ipynb
└── 05_degerlendirme_metrikleri.ipynb
```

---

### MODÜL 6 — Derin Öğrenme ile Ses
**Klasör:** `egitim/06_derin_ogrenme_ses/`

#### Konular
- **CNN tabanlı yaklaşımlar:**
  - 1D CNN ile ham ses işleme
  - 2D CNN ile spektrogram üzerinde işleme
  - VGG, ResNet adaptasyonları ses için
  - SoundNet, EnvNet mimarileri

- **RNN / LSTM / GRU tabanlı:**
  - Sekanslar için RNN'ler
  - LSTM ile konuşma tanıma
  - Bidirectional LSTM
  - CTC (Connectionist Temporal Classification) kaybı — ASR için kritik

- **Attention ve Transformer:**
  - Attention mekanizması — sezgisel açıklama
  - Self-attention ve Multi-head attention
  - Transformer mimarisi (encoder-decoder)
  - Ses için Transformer adaptasyonları

- **Conformer Mimarisi:**
  - CNN + Transformer hibrit
  - Yerel ve küresel bağlam
  - Conformer bloğu detayları

- **Diffusion modelleri ses için:**
  - Diffusion giriş: ileri süreç, ters süreç
  - DDPM, DDIM
  - Ses sentezi için diffusion

#### Dosyalar
```
06_derin_ogrenme_ses/
├── 01_cnn_ses.ipynb
├── 02_rnn_lstm_ses.ipynb
├── 03_ctc_loss.ipynb
├── 04_attention_transformer.ipynb
├── 05_conformer.ipynb
├── 06_diffusion_ses.ipynb
└── model_architectures/
    ├── conformer.py
    ├── ctc_model.py
    └── attention.py
```

#### Makale İncelemeleri (`modeller-makaleler/`)
- "Attention Is All You Need" — Vaswani et al. (2017)
- "Conformer: Convolution-augmented Transformer for Speech Recognition" — Gulati et al. (2020)
- "Deep Speech 2" — Baidu Research (2015)

---

### MODÜL 7 — VAD (Voice Activity Detection)
**Klasör:** `egitim/07_vad/`

#### Konular
- VAD nedir ve neden önemli?
- Enerji tabanlı VAD (basit eşik yöntemi)
- ZCR tabanlı VAD
- WebRTC VAD
- **Silero VAD** — PyTorch tabanlı, SOTA hafif model
- **pyannote.audio VAD**
- Streaming VAD (gerçek zamanlı)
- VAD değerlendirme: False Accept Rate, False Reject Rate
- Endpoint detection (konuşma bitiş tespiti)

#### Dosyalar
```
07_vad/
├── 01_klasik_vad.ipynb
├── 02_webrtc_vad.ipynb
├── 03_silero_vad.ipynb
├── 04_pyannote_vad.ipynb
├── 05_streaming_vad.ipynb
└── vad_utils.py
```

#### Makale İncelemeleri
- "Silero VAD" — snakers4 (2021)
- "ACAM: Attention-based Context-Aware Activity Model for VAD"

---

### MODÜL 8 — ASR / STT (Automatic Speech Recognition)
**Klasör:** `egitim/08_asr_stt/`

#### Konular
- **ASR'ye Giriş:**
  - ASR nedir, nasıl çalışır?
  - Akustik model, dil modeli, lexicon üçlüsü
  - WER (Word Error Rate) metriği
  - CER (Character Error Rate)

- **End-to-End ASR:**
  - CTC tabanlı modeller
  - Attention encoder-decoder
  - RNN-T (Transducer) mimarisi
  - Streaming vs. offline ASR

- **Whisper (OpenAI):**
  - Mimari detayları (encoder-decoder Transformer)
  - Eğitim verisi ve yaklaşımı
  - Multitask training (transcription, translation, language ID)
  - Türkçe performans analizi
  - faster-whisper, whisper.cpp ile optimizasyon
  - Fine-tuning Whisper

- **Wav2Vec 2.0 (Meta):**
  - Self-supervised pretraining
  - Contrastive learning
  - Fine-tuning ASR için
  - Mimari: feature encoder + context network

- **HuBERT (Meta):**
  - K-means clustering tabanlı pseudo-labels
  - Offline clustering + BERT-style masking
  - HuBERT vs wav2vec 2.0 karşılaştırma

- **Conformer tabanlı SOTA:**
  - ESPnet framework
  - NeMo (NVIDIA) ile Conformer-CTC, Conformer-Transducer

- **Türkçe ASR Özel:**
  - Türkçe veri setleri: Common Voice, Mozilla, OpenSTT-TR
  - Türkçe için fine-tuning stratejileri
  - Morfolojik zenginlik ve ASR zorluğu

#### Dosyalar
```
08_asr_stt/
├── 01_asr_giris_ctc.ipynb
├── 02_rnnt_transducer.ipynb
├── 03_whisper_inceleme.ipynb
├── 04_whisper_finetuning.ipynb
├── 05_wav2vec2_inceleme.ipynb
├── 06_wav2vec2_finetuning.ipynb
├── 07_hubert_inceleme.ipynb
├── 08_nemo_conformer.ipynb
├── 09_turkce_asr.ipynb
└── architectures/
    ├── whisper_arch.py
    ├── wav2vec2_arch.py
    └── rnnt.py
```

#### Makale İncelemeleri
- "Robust Speech Recognition via Large-Scale Weak Supervision" — Radford et al., OpenAI Whisper (2022)
- "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" — Baevski et al. (2020)
- "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units" — Hsu et al. (2021)
- "RNN-T for Latency-Controlled ASR" — He et al. (2019)

---

### MODÜL 9 — TTS (Text-to-Speech)
**Klasör:** `egitim/09_tts/`

#### Konular
- **TTS'ye Giriş:**
  - TTS bileşenleri: frontend, akustik model, vocoder
  - Text normalizasyon ve G2P (Grapheme-to-Phoneme)
  - Prosodi: ritim, vurgu, melodi

- **Akustik Modeller:**
  - Tacotron 2 — Attention tabanlı seq2seq
  - FastSpeech 2 — Non-autoregressive, hızlı
  - VITS — VAE + normalizing flows + GAN, end-to-end

- **Vocoder'lar:**
  - WaveNet — dilated causal convolutions
  - WaveGlow — normalizing flows
  - HiFi-GAN — GAN tabanlı yüksek kaliteli
  - BigVGAN — ölçeklendirilmiş GAN vocoder

- **Modern End-to-End TTS:**
  - VITS ve VITS2
  - NaturalSpeech (Microsoft)
  - StyleTTS 2

- **Türkçe TTS:**
  - Türkçe fonetik alfabe ve özellikler
  - Türkçe TTS için veri seti oluşturma
  - Fine-tuning stratejileri

- **Gerçek Zamanlı TTS:**
  - Streaming TTS
  - Edge deployment optimizasyonu

#### Dosyalar
```
09_tts/
├── 01_tts_giris_frontend.ipynb
├── 02_tacotron2_inceleme.ipynb
├── 03_fastspeech2_inceleme.ipynb
├── 04_vits_inceleme.ipynb
├── 05_vocoder_karsilastirma.ipynb
├── 06_hifigan_inceleme.ipynb
├── 07_turkce_tts.ipynb
├── 08_streaming_tts.ipynb
└── architectures/
    ├── tacotron2.py
    ├── fastspeech2.py
    ├── vits.py
    └── hifigan.py
```

#### Makale İncelemeleri
- "Tacotron 2: Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" — Shen et al. (2018)
- "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" — Ren et al. (2020)
- "VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End TTS" — Kim et al. (2021)
- "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" — Kong et al. (2020)
- "StyleTTS 2" — Li et al. (2023)

---

### MODÜL 10 — Voice Cloning (Ses Klonlama)
**Klasör:** `egitim/10_voice_cloning/`

#### Konular
- **Voice Cloning'e Giriş:**
  - Zero-shot, few-shot ve multi-shot klonlama
  - Speaker embedding (d-vector, x-vector)
  - Speaker encoder — mimarisi ve eğitimi

- **YourTTS:**
  - Çok dilli, zero-shot TTS
  - Mimari detayları

- **XTTS (Coqui):**
  - GPT tabanlı akustik model
  - Çok dilli destek, Türkçe
  - Gerçek zamanlı kullanım

- **Tortoise TTS:**
  - Diffusion tabanlı yüksek kaliteli klonlama
  - Autoregressive + diffusion yaklaşımı

- **OpenVoice (MyShell):**
  - Ses tonu transferi
  - Dil ve aksana bağımsız klonlama

- **ElevenLabs benzeri sistemlerin mimarisi:**
  - Emotion control, style control

- **Etik boyutlar:**
  - Deepfake ses tespiti
  - Watermarking
  - Anti-spoofing (ASVspoof veri seti)

#### Dosyalar
```
10_voice_cloning/
├── 01_speaker_embedding.ipynb
├── 02_xtts_inceleme.ipynb
├── 03_tortoise_inceleme.ipynb
├── 04_openvoice_inceleme.ipynb
├── 05_antispoof_deepfake_tespit.ipynb
└── architectures/
    ├── speaker_encoder.py
    └── voice_clone_pipeline.py
```

#### Makale İncelemeleri
- "Transfer Learning from Speaker Verification to Multispeaker TTS" — Jia et al. (2018) — SV2TTS
- "YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion" — Casanova et al. (2022)
- "XTTS: A Massively Multilingual Zero-Shot TTS"
- "Tortoise TTS" — James Betker (2023)
- "OpenVoice" — Qin et al. (2023)

---

### MODÜL 11 — Speaker Diarization (Konuşmacı Ayrıştırma)
**Klasör:** `egitim/11_speaker_diarization/`

#### Konular
- Diarization nedir? "Kim ne zaman konuştu?"
- DER (Diarization Error Rate) metriği
- Klasik pipeline: VAD → Segmentation → Embedding → Clustering
- x-vector ve d-vector konuşmacı temsilleri
- Kümeleme yöntemleri: AHC, spectral clustering, VBx
- **pyannote.audio:**
  - Segmentation modeli
  - Embedding modeli
  - End-to-end diarization
- **EEND (End-to-End Neural Diarization):**
  - Örtüşen konuşma problemi
  - EEND-EDA mimarisi
- Çok kanallı diarization (beamforming + diarization)
- Diarization + ASR entegrasyonu (word-level speaker attribution)

#### Dosyalar
```
11_speaker_diarization/
├── 01_klasik_diarization_pipeline.ipynb
├── 02_xvector_dvector.ipynb
├── 03_pyannote_diarization.ipynb
├── 04_eend_inceleme.ipynb
├── 05_diarization_asr_entegrasyon.ipynb
└── architectures/
    ├── eend.py
    └── speaker_embedding.py
```

#### Makale İncelemeleri
- "End-to-End Neural Diarization" — Fujita et al. (2019)
- "EEND-EDA: End-to-End Neural Diarization with Encoder-Decoder Attractors"
- "pyannote.audio: neural building blocks for speaker diarization" — Bredin et al. (2020)

---

### MODÜL 12 — Ses Sınıflandırma ve Olay Tespiti
**Klasör:** `egitim/12_ses_siniflandirma/`

#### Konular
- Ses olayı tespiti (Sound Event Detection — SED)
- Ortam sesi sınıflandırma (Environmental Sound Classification — ESC)
- Akustik sahne sınıflandırma (Acoustic Scene Classification — ASC)
- **PANNs (Pretrained Audio Neural Networks):**
  - AudioSet veri seti (527 sınıf)
  - CNN14 ve diğer PANN modelleri
- **DCASE Challenge (Detection and Classification of Acoustic Scenes and Events):**
  - DCASE görev kategorileri
  - Yarışma yaklaşımları
- **BirdNET:** Kuş sesi tanıma
- **YAMNet:** Google'ın MobileNet tabanlı ses sınıflandırıcısı
- **AudioCLIP:** CLIP'i sese uyarlamak
- Few-shot ses sınıflandırma
- Weakly supervised SED (zayıf etiket ile)
- Polyphonic SED — aynı anda birden fazla olay

#### Dosyalar
```
12_ses_siniflandirma/
├── 01_esc_asc_giris.ipynb
├── 02_panns_inceleme.ipynb
├── 03_yamnet_uygulamasi.ipynb
├── 04_audioclip_inceleme.ipynb
├── 05_polyphonic_sed.ipynb
├── 06_fewshot_siniflandirma.ipynb
└── architectures/
    ├── panns_cnn14.py
    └── sed_model.py
```

#### Makale İncelemeleri
- "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition" — Kong et al. (2020)
- "AudioCLIP: Extending CLIP to Audio" — Guzhov et al. (2021)
- "YAMNet" — Google (2019)

---

### MODÜL 13 — Müzik ve Ses Sentezi
**Klasör:** `egitim/13_muzik_ses/`

#### Konular
- Müzik bilgi geri getirme (MIR — Music Information Retrieval)
- Tempo ve beat tespiti
- Melodi ve akord tespiti
- Müzik kaynak ayrımı (Music Source Separation):
  - Demucs (Meta)
  - Spleeter (Deezer)
- Müzik üretimi — AI ile:
  - MusicGen (Meta)
  - AudioLDM
  - Stable Audio
- Vokal ayrımı ve karaoke uygulamaları
- Sinyal sentezi: FM, additive, subtractive

#### Dosyalar
```
13_muzik_ses/
├── 01_mir_temelleri.ipynb
├── 02_demucs_inceleme.ipynb
├── 03_musicgen_inceleme.ipynb
├── 04_audioldm_inceleme.ipynb
└── architectures/
    ├── demucs.py
    └── musicgen_overview.py
```

#### Makale İncelemeleri
- "Hybrid Transformers for Music Source Separation" — Rouard et al. (2022) — Demucs v4
- "Simple and Controllable Music Generation" — Copet et al., Meta (2023) — MusicGen
- "AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining" — Liu et al. (2023)

---

### MODÜL 14 — Foundation Models (Ses için Temel Modeller)
**Klasör:** `egitim/14_foundation_models/`

#### Konular
- **Ses Foundation Model Kavramı:**
  - Self-supervised learning paradigması
  - Pretraining + fine-tuning stratejisi

- **wav2vec 2.0 — Derinlemesine:**
  - Quantizer modülü
  - Contrastive loss
  - Downstream task fine-tuning

- **HuBERT — Derinlemesine:**
  - Iterative clustering
  - Masked unit prediction
  - Multi-layer feature analizi

- **WavLM:**
  - Masked speech denoising pretraining
  - Gürültülü ve temiz ses entegrasyonu
  - SUPERB benchmark SOTA

- **SPIRAL, data2vec ses:**
  - data2vec — unified pretraining

- **AudioPaLM (Google):**
  - Ses + metin foundation model
  - SpeechX gibi unified modeller

- **SUPERB Benchmark:**
  - Speech processing Universal PERformance Benchmark
  - Downstream görev değerlendirme çerçevesi
  - PR, ASR, KS, QbE, IC, SF, SID, ASV, SD görevleri

#### Dosyalar
```
14_foundation_models/
├── 01_ssl_paradigmasi.ipynb
├── 02_wav2vec2_derinlemesine.ipynb
├── 03_hubert_derinlemesine.ipynb
├── 04_wavlm_inceleme.ipynb
├── 05_superb_benchmark.ipynb
├── 06_audiopalm_inceleme.ipynb
└── architectures/
    ├── wav2vec2_full.py
    ├── hubert_full.py
    └── wavlm.py
```

#### Makale İncelemeleri
- "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing" — Chen et al. (2022)
- "data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language" — Baevski et al. (2022)
- "SUPERB: Speech processing Universal PERformance Benchmark" — Yang et al. (2021)
- "AudioPaLM: A Large Language Model That Can Speak and Listen" — Google (2023)

---

### MODÜL 15 — Multimodal Modeller
**Klasör:** `egitim/15_multimodal_modeller/`

#### Konular
- **Multimodal Ses Modelleri:**
  - Ses + metin: joint embedding
  - Ses + görsel: Audio-Visual Learning
  - Ses + dil + görsel: trimodal modeller

- **CLAP (Contrastive Language-Audio Pretraining):**
  - Ses ile metin eşleştirme
  - CLIP'ten sese adaptasyon
  - Zero-shot ses sınıflandırma

- **AudioCLIP:**
  - Görsel, ses, metin üçlü öğrenme

- **Gemini ve ses:**
  - Gemini'nin ses anlama yetenekleri
  - API kullanımı

- **GPT-4o ile ses:**
  - Native audio mode
  - Real-time conversation

- **Sesli Diyalog Sistemleri (Spoken Dialogue Systems):**
  - Cascade vs. end-to-end yaklaşımlar
  - ASR + LLM + TTS pipeline
  - Streaming diyalog mimarisi
  - dGSLM (Google) — speech language model

- **Voice LLM'ler:**
  - Voicebox (Meta)
  - SoundStorm (Google DeepMind)
  - AudioLM

#### Dosyalar
```
15_multimodal_modeller/
├── 01_clap_inceleme.ipynb
├── 02_audioclip_uygulama.ipynb
├── 03_cascade_vs_e2e_diyalog.ipynb
├── 04_gpt4o_ses_entegrasyon.ipynb
├── 05_audiollm_voicellm.ipynb
└── architectures/
    ├── clap.py
    └── spoken_dialogue_system.py
```

#### Makale İncelemeleri
- "CLAP: Learning Audio Concepts From Natural Language Supervision" — Elizalde et al. (2022)
- "AudioLM: A Language Modeling Approach to Audio Generation" — Borsos et al., Google (2022)
- "SoundStorm: Efficient Parallel Audio Generation" — Borsos et al., Google DeepMind (2023)
- "Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale" — Le et al., Meta (2023)

---

### MODÜL 16 — MLOps & Deployment
**Klasör:** `egitim/16_mlops_deployment/`

#### Konular
- **Ses Modeli Deployment:**
  - ONNX ile model export
  - TorchScript ve TorchServe
  - Triton Inference Server
  - FastAPI ile ses API'si oluşturma
  - WebSocket ile streaming ASR/TTS

- **Optimizasyon Teknikleri:**
  - Quantization (INT8, FP16)
  - Knowledge distillation (Distil-Whisper)
  - Pruning
  - Speculative decoding (ASR için)

- **Edge Deployment:**
  - whisper.cpp (CPU optimized)
  - faster-whisper (CTranslate2)
  - ONNX Runtime Mobile
  - Raspberry Pi, Jetson Nano üzerinde ASR

- **Real-time Streaming Pipeline:**
  - Chunk-based processing
  - VAD entegrasyonu
  - Latency optimizasyonu
  - WebRTC entegrasyonu

- **Ses Verisi Yönetimi:**
  - Veri toplama ve etiketleme
  - Kalite kontrol pipeline
  - Data versioning (DVC)

- **Monitoring:**
  - WER tracking production'da
  - Drift tespiti
  - A/B test ses modelleri

- **CI/CD ses projeleri için:**
  - Otomatik WER testi
  - Regresyon testi
  - Model registry

#### Dosyalar
```
16_mlops_deployment/
├── 01_model_export_onnx.ipynb
├── 02_fastapi_ses_api.ipynb
├── 03_streaming_asr_pipeline.ipynb
├── 04_quantization_distilasyon.ipynb
├── 05_edge_deployment.ipynb
├── 06_monitoring_wer.ipynb
└── deployment/
    ├── fastapi_app.py
    ├── triton_config/
    ├── docker/
    └── streaming_server.py
```

---

## 📚 Modeller-Makaleler Klasörü

```
modeller-makaleler/
│
├── asr/
│   ├── whisper_paper.md          # Detaylı okuma notları
│   ├── wav2vec2_paper.md
│   ├── hubert_paper.md
│   ├── rnnt_paper.md
│   ├── deepspeech2_paper.md
│   └── conformer_paper.md
│
├── tts/
│   ├── tacotron2_paper.md
│   ├── fastspeech2_paper.md
│   ├── vits_paper.md
│   ├── hifigan_paper.md
│   └── styletts2_paper.md
│
├── vad/
│   ├── silero_vad.md
│   └── pyannote_vad.md
│
├── voice_cloning/
│   ├── sv2tts_paper.md
│   ├── yourtts_paper.md
│   └── openvoice_paper.md
│
├── foundation_models/
│   ├── wavlm_paper.md
│   ├── data2vec_paper.md
│   └── superb_paper.md
│
├── multimodal/
│   ├── clap_paper.md
│   ├── audiollm_paper.md
│   └── soundstorm_paper.md
│
└── klasik_yontemler/
    ├── hmm_gmm.md
    ├── mfcc_history.md
    └── dtw.md
```

---

## 🛠️ Teknolojiler Klasörü

```
teknolojiler/
│
├── kutuphaneler/
│   ├── librosa_rehber.ipynb
│   ├── soundfile_rehber.ipynb
│   ├── torchaudio_rehber.ipynb
│   ├── huggingface_ses_rehber.ipynb
│   ├── pyannote_rehber.ipynb
│   ├── espnet_rehber.ipynb
│   └── nemo_rehber.ipynb
│
├── framework_karsilastirma/
│   ├── asr_framework_karsilastirma.md
│   ├── tts_framework_karsilastirma.md
│   └── diarization_framework_karsilastirma.md
│
├── deployment_araclari/
│   ├── faster_whisper_rehber.ipynb
│   ├── whisper_cpp_rehber.md
│   ├── triton_ses_rehber.ipynb
│   └── onnxruntime_ses_rehber.ipynb
│
└── veri_setleri/
    ├── veri_setleri_rehber.md
    ├── librispeech.md
    ├── common_voice.md
    ├── vctk.md
    ├── ljspeech.md
    └── turkce_veri_setleri.md
```

### Temel Kütüphaneler

| Kütüphane | Amaç | Notlar |
|---|---|---|
| librosa | Ses analizi, özellik çıkarma | Temel kütüphane |
| soundfile | Ses okuma/yazma | WAV, FLAC vb. |
| torchaudio | PyTorch ses kütüphanesi | GPU desteği |
| transformers (HF) | Pretrained modeller | Whisper, wav2vec2, vb. |
| pyannote.audio | Diarization, VAD | SOTA pipeline |
| ESPnet | ASR, TTS, SE | Araştırma odaklı |
| NVIDIA NeMo | ASR, TTS, NLP | Production odaklı |
| Coqui TTS | TTS, voice cloning | Açık kaynak XTTS |
| silero-vad | Hafif VAD | Edge uyumlu |
| faster-whisper | Hızlı ASR inference | CTranslate2 backend |
| pydub | Ses manipülasyon | Kolay API |
| pedalboard | Audio efektler | Spotify'dan |

---

## 🎬 YouTube Video Serisi Planı

### Seri 1: Ses Fiziği ve DSP (5 Video)
| # | Başlık | Modül |
|---|---|---|
| 1 | Sesin Yapısı: Frekans, Genlik, Örnekleme | Modül 1 |
| 2 | Fourier Dönüşümü Sezgisel: "Sesi Parçalara Ayırmak" | Modül 2 |
| 3 | FFT ve Spektrogram: Sesin Görsel Temsili | Modül 2 |
| 4 | Filtreler: Sesi Şekillendirmek | Modül 2 |
| 5 | STFT: Zamanla Değişen Sesi Analiz Etmek | Modül 2-3 |

### Seri 2: Özellik Mühendisliği (4 Video)
| # | Başlık | Modül |
|---|---|---|
| 6 | MFCC Nedir? Adım Adım Mel Cepstral Coefficients | Modül 4 |
| 7 | Mel Spektrogram: İnsan Kulağını Taklit Etmek | Modül 4 |
| 8 | Pitch Tracking ve Temel Frekans Tespiti | Modül 4 |
| 9 | Ses Veri Artırma Teknikleri: SpecAugment ve Daha Fazlası | Modül 3-4 |

### Seri 3: Derin Öğrenme Mimarileri (5 Video)
| # | Başlık | Modül |
|---|---|---|
| 10 | Sesle CNN: Spektrogram Üzerinde Görsel Öğrenme | Modül 6 |
| 11 | CTC Loss: Ses Tanıma için Kayıp Fonksiyonu | Modül 6 |
| 12 | Conformer Mimarisi: CNN + Transformer = ? | Modül 6 |
| 13 | RNN-T (Transducer): Streaming ASR'nin Temeli | Modül 8 |
| 14 | Diffusion Modelleri Ses için Nasıl Çalışır? | Modül 6 |

### Seri 4: ASR (6 Video)
| # | Başlık | Modül |
|---|---|---|
| 15 | Whisper Mimarisi Derinlemesine İnceleme | Modül 8 |
| 16 | Whisper ile Türkçe ASR ve Fine-Tuning | Modül 8 |
| 17 | wav2vec 2.0: Self-Supervised ile Konuşma Tanıma | Modül 8 |
| 18 | HuBERT: Öz-Gözetimli Ses Öğrenimi | Modül 8 |
| 19 | NVIDIA NeMo ile Production ASR | Modül 8 |
| 20 | Gerçek Zamanlı (Streaming) ASR Pipeline Kurmak | Modül 8 + 16 |

### Seri 5: TTS (5 Video)
| # | Başlık | Modül |
|---|---|---|
| 21 | Tacotron 2: Metinden Sese Klasik Yaklaşım | Modül 9 |
| 22 | FastSpeech 2: Non-Autoregressive TTS | Modül 9 |
| 23 | VITS: End-to-End Gerçekçi TTS | Modül 9 |
| 24 | HiFi-GAN Vocoder: Yüksek Kaliteli Ses Sentezi | Modül 9 |
| 25 | Türkçe TTS Fine-Tuning: Sıfırdan Türkçe Ses Sentezi | Modül 9 |

### Seri 6: Voice Cloning (4 Video)
| # | Başlık | Modül |
|---|---|---|
| 26 | Speaker Embedding: Bir Sesi Vektöre Sıkıştırmak | Modül 10 |
| 27 | XTTS ile Türkçe Ses Klonlama | Modül 10 |
| 28 | Tortoise TTS: Yüksek Kaliteli Zero-Shot Klonlama | Modül 10 |
| 29 | Deepfake Ses Tespiti ve Etik | Modül 10 |

### Seri 7: Diarization ve Sınıflandırma (4 Video)
| # | Başlık | Modül |
|---|---|---|
| 30 | VAD: Seste Konuşma Var mı? Silero ve pyannote | Modül 7 |
| 31 | Speaker Diarization: Kim Ne Zaman Konuştu? | Modül 11 |
| 32 | PANNs ile Ses Olayı Tespiti | Modül 12 |
| 33 | EEND: End-to-End Neural Diarization | Modül 11 |

### Seri 8: Foundation Models ve Multimodal (5 Video)
| # | Başlık | Modül |
|---|---|---|
| 34 | Ses Foundation Modelleri: SSL Paradigması | Modül 14 |
| 35 | WavLM: Full-Stack Ses İşleme için SSL | Modül 14 |
| 36 | CLAP: Ses ve Metni Birleştirmek | Modül 15 |
| 37 | AudioLM ve SoundStorm: Ses Dil Modelleri | Modül 15 |
| 38 | GPT-4o ile Sesli Uygulamalar Geliştirmek | Modül 15 |

### Seri 9: MLOps ve Deployment (4 Video)
| # | Başlık | Modül |
|---|---|---|
| 39 | Distil-Whisper: Bilgi Damıtma ile Hızlı ASR | Modül 16 |
| 40 | Ses API'si Kurmak: FastAPI + Streaming | Modül 16 |
| 41 | Edge'de ASR: whisper.cpp ve faster-whisper | Modül 16 |
| 42 | Ses Modellerinde Monitoring ve MLOps | Modül 16 |

**Toplam: 42 Video**

---

## 📊 Öğrenme Yolu (Learning Path)

```
Başlangıç
    │
    ▼
[Modül 1] Ses Fiziği
    │
    ▼
[Modül 2] DSP Temelleri
    │
    ▼
[Modül 3] Ses İşleme
    │
    ▼
[Modül 4] Özellik Mühendisliği
    │
    ├──────────────────────┐
    │                      │
    ▼                      ▼
[Modül 5]            [Modül 6]
Klasik ML            Derin Öğrenme
    │                      │
    └──────────┬───────────┘
               │
    ┌──────────┼──────────┐──────────┐
    │          │          │          │
    ▼          ▼          ▼          ▼
[Modül 7]  [Modül 8]  [Modül 9]  [Modül 12]
   VAD        ASR        TTS      Ses Sınıf.
               │          │
               └────┬─────┘
                    │
               [Modül 10]
             Voice Cloning
                    │
               [Modül 11]
             Diarization
                    │
    ┌───────────────┤
    │               │
    ▼               ▼
[Modül 13]   [Modül 14]
  Müzik      Foundation
              Models
                    │
               [Modül 15]
              Multimodal
                    │
               [Modül 16]
            MLOps & Deploy
```

---

## 🔧 Ortam Kurulumu

```bash
# Conda ortamı
conda create -n audio-ai python=3.10
conda activate audio-ai

# Core kütüphaneler
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install librosa soundfile transformers datasets
pip install pyannote.audio faster-whisper
pip install TTS  # Coqui TTS
pip install nemo_toolkit['asr']
pip install jupyter notebook ipywidgets
pip install matplotlib seaborn plotly
pip install fastapi uvicorn websockets
pip install onnxruntime onnx
```

---

## 📈 Değerlendirme Metrikleri Referans Tablosu

| Görev | Metrik | Açıklama |
|---|---|---|
| ASR | WER | Word Error Rate |
| ASR | CER | Character Error Rate |
| TTS | MOS | Mean Opinion Score (subjektif) |
| TTS | UTMOS | Otomatik MOS tahmini |
| TTS | WER on synth | Sentezlenen sesin anlaşılırlığı |
| VAD | FAR / FRR | False Accept/Reject Rate |
| Diarization | DER | Diarization Error Rate |
| Speaker Verif. | EER | Equal Error Rate |
| Ses Sınıf. | mAP | mean Average Precision |
| SED | PSDS | Polyphonic SED Detection Score |

---

## 🌐 Önerilen Veri Setleri

| Veri Seti | Görev | Dil | Boyut |
|---|---|---|---|
| LibriSpeech | ASR | İngilizce | 960 saat |
| Common Voice | ASR | Çok dilli | — |
| VCTK | TTS/VC | İngilizce | 110 konuşmacı |
| LJSpeech | TTS | İngilizce | 24 saat |
| VoxCeleb 1/2 | Speaker ID | İngilizce | 2000+ konuşmacı |
| AudioSet | Ses sınıf. | — | 2M klip |
| ESC-50 | Çevre sesi | — | 2000 klip |
| DCASE datasets | SED/ASC | — | — |
| ASVspoof | Anti-spoof | İngilizce | — |
| Mozilla CV TR | ASR | Türkçe | — |

---

*Bu plan açık kaynak topluluk katkılarına açıktır. Her modül bağımsız öğrenilebilecek şekilde tasarlanmıştır.*