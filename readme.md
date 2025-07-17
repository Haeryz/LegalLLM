# LegalLLM Indonesia: Fine-tuning Model untuk Analisis Hukum Indonesia

**Subtitle:** Pengembangan Model Bahasa Besar untuk Analisis Dokumen Hukum Indonesia  
**Autor:** Muhammad Hariz Faizul Anwar & Nizam Arif  
**Institusi:** Universitas Muhammadiyah Malang  
**Tanggal:** 17 Juli 2025

---

## Ringkasan Eksekutif

Proyek ini mengembangkan LLM yang dikhususkan untuk analisis dokumen hukum Indonesia melalui fine-tuning menggunakan framework Unsloth. Model berbasis Gemma-3 4B dilatih pada dataset putusan pengadilan Indonesia untuk menghasilkan analisis hukum yang mendalam dan terstruktur. Hasil evaluasi menunjukkan model mampu menghasilkan analisis yang komprehensif dengan exact n-gram overlap 26.94%, mengungguli model baseline dalam pemahaman konteks hukum Indonesia.

---

## Permasalahan

Analisis dokumen hukum Indonesia menghadapi beberapa tantangan:
- **Kompleksitas Bahasa Hukum:** Terminologi dan struktur bahasa hukum Indonesia yang kompleks
- **Keterbatasan Sumber Daya:** Kurangnya tools AI yang dioptimalkan untuk konteks hukum Indonesia
- **Efisiensi Waktu:** Proses analisis manual yang memakan waktu lama
- **Konsistensi Analisis:** Variasi kualitas analisis antar praktisi hukum
- **Aksesibilitas:** Keterbatasan akses terhadap expertise analisis hukum untuk masyarakat umum

---

## Gambaran Dataset

### Sumber Data
- **Sumber Utama:** Putusan Pengadilan Indonesia dari berbagai tingkatan
- **Kategori Kasus:** Perdagangan Manusia
- **Format:** JSONL (JavaScript Object Notation Lines)
- **Struktur Data:**
  ```json
  {
    "instruction": "Analisis mendalam putusan ini:",
    "input": "[Teks putusan pengadilan]",
    "output": "[Analisis hukum komprehensif]"
  }
  ```

### Spesifikasi Dataset
- **Ukuran:** Dataset ekstrak teks dari putusan pengadilan
- **Preprocessing:** 
  - Ekstraksi teks dari dokumen PDF putusan
  - Pembersihan dan normalisasi format
  - Strukturisasi ke format instruksi-input-output
- **Pembagian Data:** 85% training, 15% validation (seed=3407)
- **Lisensi:** Data publik dari Mahkamah Agung RI

### Link Data
- 📁 [Raw Data (Google Drive)](https://drive.google.com/drive/folders/14kWdffZru_ePZ4B58EzS3EO5cW-FLNRa?usp=sharing)

---

## Pipeline Fine-tuning

### Framework dan Tools
- **Framework Utama:** Unsloth (optimized LoRA fine-tuning)
- **Base Model:** `unsloth/gemma-3-1b-it` (Gemma-3 4B Instruct)
- **Teknik:** LoRA (Low-Rank Adaptation) dengan r=16
- **Hardware:** Google Colab Tesla T4 (gratis)

### Hyperparameters
```python
# Konfigurasi LoRA
r = 16
lora_alpha = 16
lora_dropout = 0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]

# Training Parameters
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
warmup_steps = 5
num_train_epochs = 3
learning_rate = 2e-4
optim = "adamw_8bit"
weight_decay = 0.01
lr_scheduler_type = "linear"
seed = 3407
max_seq_length = 2048
```

### Kriteria Training
- **Epochs:** 3 epochs penuh
- **Stopping Criteria:** Berdasarkan jumlah epochs
- **Optimasi:** Response-only training (hanya training pada output assistant)
- **Chat Template:** Gemma-3 format

---

## Artifacts Model

### Model Checkpoints
- 🤗 **Hugging Face Repository:** [legal-gemma3-id-16bit](https://huggingface.co/Haeryz/legal-gemma3-id-16bit)
- **Format Tersedia:**
  - 16-bit merged model (production ready)
  - LoRA adapters (untuk further fine-tuning)
  - GGUF quantized versions (q4_k_m, q8_0, q5_k_m)

### Spesifikasi Teknis
- **Base Model Size:** ~1B parameters
- **LoRA Adapter Size:** ~16M parameters (1.6% dari base model)
- **Final Model Size:** ~2GB (16-bit), ~1GB (4-bit quantized)
- **Memory Requirements:** 
  - Training: ~8GB VRAM (dengan gradient checkpointing)
  - Inference: ~4GB VRAM (16-bit), ~2GB VRAM (4-bit)

---

## Desain Evaluasi

### Metrik Evaluasi
- **BLEU Score:** Mengukur kemiripan n-gram dengan referensi
- **ROUGE-L Score:** Mengukur longest common subsequence
- **Exact N-gram Overlap:** Persentase overlap kata/frasa dengan training data
- **Evaluasi Kualitatif:** Analisis manual kualitas output

### Validasi Split
- **Training Set:** 85% dari total data
- **Validation Set:** 15% dari total data
- **Random Seed:** 3407 (untuk reproducibility)

### Parameter Inferensi Standar
```python
generation_params = {
    "max_new_tokens": 512,
    "min_new_tokens": 20,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id
}
```

---

## Tabel Hasil

### Perbandingan Model Quantitative

| Model | Type | Size | BLEU | ROUGE-L | N-gram Overlap |
|-------|------|------|------|---------|----------------|
| **Legal-Gemma3-ID** | Fine-tuned | 1B | - | - | **26.94%** |
| google/gemma-2b-it | Baseline | 2B | - | - | - |
| Qwen2-0.5B-Instruct | Tiny | 0.5B | - | - | - |

### Contoh Output Kualitatif

**Input:** Analisis mendalam putusan Pengadilan Negeri Denpasar tentang penggelapan dalam jabatan oleh Nyoman Suharyeni

**Legal-Gemma3-ID (Model Kami) Output:**
```
Berikut adalah analisis mendalam mengenai putusan tersebut:

1. **Kebenaran Putusan:**
   - Penyelidikan dan Bukti: Putusan didasarkan pada penyelidikan menyeluruh
   - Pengakuan Terdakwa: Nyoman Suharyeni mengakui melakukan penggelapan
   - Kerugian Material: Kerugian mencapai Rp 2,5 miliar

2. **Peran Terdakwa:**
   - Pelanggaran Tugas: Melanggar tanggung jawab sebagai kasir
   - Kesalahan Akuntabilitas: Tidak menjalankan tugas dengan benar
   - Konsekuensi Hukum: Tindak pidana serius dengan sanksi berat

3. **Implikasi Hukum:**
   - Pasal 374 Ayat 1 KUHP: Penggelapan dalam jabatan
   - Kehilangan Kepercayaan: Dampak terhadap reputasi perusahaan
```

**Baseline Model (Gemma-2B-IT):** 
```
Laporan: 
- Denpasar, [Tanggal]
- Nama: Nyoman Suharyeni
- Nomor HP: [Nomor HP]

Pernyataan:

1. Putusan Pengadilan Negeri Denpasar tidak mengizinkan Nyoman Suharyeni untuk melakukan fraud.
2. Nyoman Suharyeni melakukan fraud selama 3 tahun.
3. Nyoman Suharyeni merugikan Rp 2,5 M.
4. Denpasar memiliki kebijakan hukum yang memblintai setiap orang yang melakukan fraud.

Dampak:

1. Nyoman Suharyeni tidak dapat melakukan pekerjaan di tempat kerja.
2. Nyoman Suharyeni kehilangan pendapatan dan pengangguran.
3. Nyoman Suharyeni tidak dapat menerima tunai dan hadiah dari tempat kerja.

Penting:

1. Putusan ini hanya berlaku untuk kasus yang memenuhi pernyataan di atas.
2. Nyoman Suharyeni memiliki kesempatan untuk mengajukan permintaan hukum jika dia ingin.

Rekomendasi:

1. Nyoman Suharyeni dapat mengajukan permohonan hukum untuk membatalkan putusan pengadilan.
2. Nyoman Suharyeni dapat mengajukan tuntuhan hukum dengan tunai dan hadiah sebagai denda.
3. Denpasar dapat meningkatkan kebijakan hukum yang memblintai setiap orang yang melakukan fraud.
```

**Tiny Model (Qwen2-0.5B):**
```
Sekretaris Daerah Kecamatan Taman Jawa, Taman Jawa

Mengenai penegakan hukum atas permintaan tindakan hukum di hadapan penduduk masyarakat, yaitu pengadilan ngerah, tentunya ada pertanyaan yang harus dipertanyakan. Apakah pengadilan ngerah dapat mengevaluasi permintaan tindakan hukum tersebut? Apa status dan kebijaksanaan dari hukuman pengadilan ngerah?

Pada dasarnya, hukuman pengadilan ngerah adalah saran atau bantuan yang diberikan oleh pihak atas kepada seseorang untuk mengurangi atau memeriksa perbuatan seseorang. Hukuman ini biasanya ditujukan untuk membantu orang tersebut memperbaiki perilaku mereka atau memberikan kesempatan untuk melanjutkan pekerjaannya.

Secara umum, hukuman pengadilan ngerah sering digunakan dalam kasus-kasus yang berdampak negatif terhadap kesejahteraan manusia. Misalnya, jika seseorang melakukan kejahatan seperti penculikan, hukuman pengadilan ngerah bisa dianggap sebagai cara yang tepat untuk menggantikan keputusan hukum dengan hukuman.

Namun, penting untuk diingat bahwa hukuman pengadilan ngerah bukanlah solusi semua masalah, tetapi lebih kepada bagaimana menerima dan menghormati hak asasi manusia lainnya. 

Berdasarkan perspektif hukum dan hukuman pengadilan ngerah, pengadilan ngerah tidak boleh menyatakan bahwa hukuman itu merupakan "saran" atau "bantuan" bagi seseorang untuk mengurangi atau memeriksa perbuatan seseorang. Jika ada kekhawatiran tentang hukuman tersebut, maka sebaiknya dianjurkan untuk bertemu langsung dengan dokter atau ahli hukum untuk mendapatkan saran yang lebih jelas.

Kesimpulan: Penyebutan hukuman pengadilan ngerah sebagai saran atau bantuan bagi seseorang untuk mengurangi atau memeriksa perbuatan seseorang adalah sebuah perkemb
```
---

## Keterbatasan & Risiko

### Keterbatasan
- **Coverage Domain:** Terbatas pada jenis kasus tertentu dalam training data
- **Bias Data:** Kemungkinan bias dari sumber data putusan pengadilan
- **Ukuran Model:** Model 4B parameter memiliki keterbatasan dibanding model yang lebih besar
- **Konteks Panjang:** Keterbatasan dalam menangani dokumen yang sangat panjang

### Risiko
- **Halusinasi:** Model dapat menghasilkan informasi hukum yang tidak akurat
- **Bias Sistemik:** Kemungkinan bias dalam analisis berdasarkan pola training data
- **Ketergantungan Konteks:** Performa dapat menurun untuk kasus di luar domain training
- **Keterbatasan Komputasi:** Memerlukan hardware dengan spesifikasi minimal untuk inference

### Disclaimer
⚠️ **PENTING:** Model ini adalah alat bantu analisis dan tidak menggantikan konsultasi hukum profesional. Selalu verifikasi output dengan ahli hukum yang kompeten.

---

## Panduan Penggunaan

### Quick Start - One Command Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model dan tokenizer
model_name = "Haeryz/legal-gemma3-id-16bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Contoh penggunaan
prompt = """Analisis mendalam putusan ini:

Putusan Pengadilan Negeri Jakarta
Nomor: 123/Pid/2024/PN Jkt
Terdakwa: [Nama Terdakwa]
Dakwaan: [Pasal dan Dakwaan]
Ringkasan: [Ringkasan Kasus]

Jawaban:"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.1
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Google Colab Links
- 🔧 [Fine-tuning Notebook](https://colab.research.google.com/drive/1nHWwPOAc5E5H-0O6I985ftvr_hJzy3ey?usp=sharing)
- 📊 [Evaluation Notebook](https://colab.research.google.com/drive/1vzoGxEQMKi1PKQhTyyCyqLHE8Z97K8DY?usp=sharing)

### Requirements
```
transformers>=4.36.0
torch>=2.0.0
peft
bitsandbytes
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
```

---

## Reproducibility

### Struktur Repository
```
LegalLLM/
├── readme.md                 # Dokumentasi proyek
├── Data/
│   ├── Conversation.jsonl    # Data conversational (dalam pengembangan)
│   └── extracted_text.jsonl  # Data training utama
└── Notebooks/
    ├── Eval.ipynb            # Notebook evaluasi model
    └── Paper_Hukum_Gemma,LLama,Mistral.ipynb  # Training notebook
```

### Seed Values
- **Training Seed:** 3407
- **Data Split Seed:** 3407
- **Random State:** 3407

### Environment
- **Platform:** Google Colab
- **GPU:** Tesla T4 (15GB VRAM)
- **Python:** 3.10+
- **CUDA:** 11.8+

### Reproduksi Training
1. Buka [Fine-tuning Colab](https://colab.research.google.com/drive/1nHWwPOAc5E5H-0O6I985ftvr_hJzy3ey?usp=sharing)
2. Mount Google Drive dan akses data dari [Raw Data folder](https://drive.google.com/drive/folders/14kWdffZru_ePZ4B58EzS3EO5cW-FLNRa?usp=sharing)
3. Jalankan semua cell secara berurutan
4. Model akan otomatis disimpan ke Hugging Face Hub

---

## Rencana Pengembangan

### Short-term (1-3 bulan)
- **Ekspansi Dataset:** Menambah variasi jenis kasus hukum
- **Multi-turn Conversations:** Implementasi chat format untuk interaksi yang lebih natural
- **Model Comparison:** Evaluasi sistematis dengan model lain (Llama, Mistral)

### Medium-term (3-6 bulan)
- **Larger Model:** Fine-tuning pada model yang lebih besar (7B-13B parameters)
- **Domain Specialization:** Model khusus untuk bidang hukum tertentu (pidana, perdata, dll)
- **Evaluation Framework:** Sistem evaluasi otomatis yang lebih komprehensif

### Long-term (6-12 bulan)
- **RLHF Implementation:** Reinforcement Learning from Human Feedback untuk meningkatkan kualitas
- **Production Deployment:** API dan web interface untuk penggunaan praktis
- **Integration:** Integrasi dengan sistem manajemen kasus hukum
- **Multi-modal:** Mendukung input dokumen PDF dan gambar

---

## Appendices

### A. Contoh Prompt dan Output

**Prompt Template:**
```
Analisis mendalam putusan ini:

[DETAIL PUTUSAN]

Jawaban:
```

### B. Error Logs
Lihat notebook evaluasi untuk detail error handling dan troubleshooting.

### C. Lisensi
- **Model:** Apache 2.0 (following base Gemma license)
- **Data:** Creative Commons (sesuai ketentuan data publik MA-RI)
- **Code:** MIT License

### D. Acknowledgments
- **Unsloth Team:** Framework optimized fine-tuning
- **Google:** Gemma base models
- **Hugging Face:** Model hosting dan tools
- **Mahkamah Agung RI:** Sumber data putusan pengadilan

---

### Contact & Support

- **GitHub Issues:** [Repository Issues](https://github.com/Haeryz/LegalLLM/issues)
- **Model Page:** [Hugging Face Model](https://huggingface.co/Haeryz/legal-gemma3-id-16bit)
- **Email:** muh4mm4dh4r1z@gmail.com

---

*Terakhir diupdate: Juli 2025*