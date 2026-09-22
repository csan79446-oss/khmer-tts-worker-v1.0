# 🚀 សៀវភៅណែនាំ Deploy AI Worker ទៅកាន់ RunPod Serverless (Khmer TTS / VoxCPM ពិតប្រាកដ)

សៀវភៅណែនាំនេះនឹងបង្ហាញពីរបៀបយក **AI Worker** នៅក្នុង folder `runpod_worker/` ទៅកាន់ **RunPod Serverless** ដើម្បីដំណើរការ **ម៉ូដែល VoxCPM ពិតប្រាកដ** (OpenBMB VoxCPM2 — Voice Cloning & Voice Design) ភ្ជាប់ជាមួយកម្មវិធី Khmer TTS V2។

> [!IMPORTANT]
> Worker ប្រើម៉ូដែល VoxCPM ពិត — វាត្រូវការ **torch >= 2.5** (រូបភាព Docker បានកំណត់រួចរាល់), **Container Disk >= 25GB** ហើយគួរប្រើ **RunPod Network Volume** ដើម្បីរក្សាទុក model weights មិនឱ្យទាញឡើងវិញរាល់ពេល cold start។

---

## 📋 តម្រូវការជាមុន (Prerequisites)

1. **Docker Desktop** (ឬ Docker CLI)
2. គណនី **Docker Hub** ([hub.docker.com](https://hub.docker.com))
3. គណនី **RunPod** មាន Credit ([runpod.io](https://runpod.io))

---

## ជំហានទី ១៖ ធ្វើតេស្តសាកល្បង Worker ក្នុងម៉ាស៊ីន (Local Test)

```powershell
cd c:\App\Khmer_TTS_V2
python runpod_worker\test_local.py
```

* ប្រសិនបើជោគជ័យ អ្នកនឹងឃើញ `[✓] Local worker test PASSED!` និង file [test_output.wav](test_output.wav)។
* ក្នុងម៉ាស៊ីនកុំព្យូទ័រ (ដែលមិនមាន package `voxcpm`), តេស្តនេះដំណើរការតាម **Edge-TTS fallback** ដើម្បីផ្ទៀងផ្ទាត់ payload/response contract។ នៅលើ RunPod (ក្នុង Docker), VoxCPM ពិតនឹងដំណើរការជាមួយ GPU។

---

## ជំហានទី ២៖ Build Docker Image & Push ទៅ Docker Hub

```powershell
cd c:\App\Khmer_TTS_V2\runpod_worker
docker login
docker build -t YOUR_DOCKERHUB_USERNAME/khmer-tts-worker:v2.0 .
docker push YOUR_DOCKERHUB_USERNAME/khmer-tts-worker:v2.0
```

> [!TIP]
> Repository ត្រូវតែ **Public** (ឬបន្ថែម Docker Credentials ក្នុង RunPod ប្រសិនបើ Private)។

---

## ជំហានទី ៣៖ បង្កើត Network Volume (ណែនាំខ្លាំង)

1. ចូល **RunPod Console** ➡️ **Storage** ➡️ **Network Volume** ➡️ **New Volume**
2. តម្លៃ៖ **Size `30GB`**, Region ដូច Endpoint របស់អ្នក (ឧ. US, EU...)
3. Volume នេះនឹងរក្សាទុក model weights (~៥GB+) ដូច្នេះ cold start លឿន និងមិនទាញឡើងវិញរាល់ពេល Worker ថ្មីចាប់ផ្តើម។

---

## ជំហានទី ៤៖ បង្កើត Serverless Template

1. **Serverless** ➡️ **Templates** ➡️ **New Template**
2. បំពេញ៖
   * **Template Name**: `Khmer-TTS-Worker`
   * **Container Image**: `YOUR_DOCKERHUB_USERNAME/khmer-tts-worker:v2.0`
   * **Container Disk**: `25GB` (ឬ `30GB`)
   * **Volume Mount**: 
     * អាចជ្រើសរើស `/runpod-volume` (Default របស់ RunPod) ឬ `/workspace`
     * > [!NOTE]
     * > Code របស់ Worker ត្រូវបានរក្សាទុកដាច់ដោយឡែកក្នុង `/app` ដូច្នេះការ mount volume ទៅ `/runpod-volume` ឬ `/workspace` នឹង **មិនបាត់បង់ ឬ overwrite code** ឡើយ! Worker នឹងស្វែងរក Model និង HuggingFace cache ដោយស្វ័យប្រវត្តិ។
   * **Environment Variables** (កំណត់ស្រេចក្នុង Docker image — អាចប្តូរតាមចិត្ត)：

     | Variable | តម្លៃលំនាំដើម | អត្ថន័យ |
     | :--- | :--- | :--- |
     | `VOXCPM_MODEL_ID` | `openbmb/VoxCPM2` | Model repo ID ឬ local checkpoint directory |
     | `VOXCPM_DEVICE` | `auto` | `auto` (ជ្រើស `cuda` អូតូបើមាន GPU, else `cpu`) |
     | `VOXCPM_TIMESTEPS` | `10` | Diffusion timesteps (4-30, លំនាំដើម 10 លឿននិងច្បាស់) |
     | `VOXCPM_DENOISER` | `0` | `1` = ដាក់ denoiser សម្រាប់ reference audio |
     | `HF_HOME` | `/workspace/models/hf_cache` | ទីតាំង cache weights នៅលើ Network Volume |
     | `MODEL_PATH` | `/workspace/models` | ទីតាំង local checkpoints (ស្វែងរក auto ក្នុង `/runpod-volume` ផងដែរ) |
3. ចុច **Save Template**។

---

## ជំហានទី ៥៖ បង្កើត Serverless Endpoint

1. **Serverless** ➡️ **Endpoints** ➡️ **New Endpoint** ➡️ ជ្រើស Template `Khmer-TTS-Worker`
2. **GPU Types** (VoxCPM2 ជា foundation model 2B — ត្រូវការ VRAM >= 16GB)៖
   * **RTX 4090 (24GB)** (ណែនាំខ្លាំងបំផុត — លឿនបំផុត និងតម្លៃសមរម្យ)
   * **A40 (48GB)** ឬ **RTX 3090 (24GB)**
3. **Execution Timeout**: កំណត់យ៉ាងតិច `180s` (សម្រាប់ Request វែង)
4. **Active Workers (Min)**: `0` (សន្សំសំចៃ) | **Max Workers**: `2` | **Idle Timeout**: `60s`
5. ចុច **Create Endpoint**។

---

## ជំហានទី ៦៖ យក API Key និង Endpoint ID

1. **Endpoint URL**: `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID`
2. **API Key**: **Settings** ➡️ **API Keys** ➡️ **+ API Key** (`rpa_...`)

---

## ជំហានទី ៧៖ ភ្ជាប់ជាមួយកម្មវិធី Khmer TTS V2

1. បើកកម្មវិធី៖ `python main.py`
2. **Settings** ⚙️ ➡️ Tab **"VoxCPM & RunPod"**
3. បំពេញ **RunPod API Key** និង **RunPod Endpoint URL** រួច **Test Connection**
4. ដោះធីក **"Enable Mock VoxCPM Mode"** ដើម្បីប្រើ Serverless ពិត។
5. **Save Changes**។

> ការប្រើ Engine៖
> * **VoxCPM** (Serverless) — ជម្រើសម៉ូដែល AI ពិត, មាន Voice Cloning (reference audio), Voice Design (prompt presets), emotion, temperature, speed។
> * **Edge-TTS** (Local, ឥតគិតថ្លៃ) — ភ្លាមៗ, សំឡេង km-KH-PisethNeural / km-KH-SreymomNeural, មិនមាន cloning។
> រាល់ការឆ្លើយតបពី Worker មាន field `engine` ដែលប្រាប់ថាសំឡេងបង្កើតដោយម៉ូដែលណា។

---

## 🛠️ ការដោះស្រាយបញ្ហា (Troubleshooting)

| បញ្ហា | មូលហេតុ | ដំណោះស្រាយ |
| :--- | :--- | :--- |
| **Authentication Failed (401/403)** | API Key មិនត្រឹមត្រូវ | ពិនិត្យ API Key ឡើងវិញ |
| **Endpoint Not Found (404)** | Endpoint ID ខុស | URL ត្រូវមានទម្រង់ `https://api.runpod.ai/v2/{ENDPOINT_ID}` |
| **Job Timeout / Cold Start យឺត** | ទាញ model weights លើកដំបូង | ប្រើ **Network Volume** + `HF_HOME`; បង្កើន Timeout ក្នុង Settings ទៅ 300s |
| **CUDA OOM** | VRAM មិនគ្រប់គ្រាន់សម្រាប់ VoxCPM2 (2B) | ប្រើ GPU >= 16GB ឬ 24GB (RTX 3090/4090) |
| **`no kernel image available` / Worker unhealthy** | GPU ជំនាន់ Blackwell (B200, sm_100) — PyTorch 2.5.1+CUDA12.4 មិនគាំទ្រ | **ជម្រើស A (ងាយ):** Endpoint → Edit → GPU Types → deselect B200 → ប្រើ RTX 4090/A100/H100 ។ **ជម្រើស B:** Rebuild Docker image ដោយប្រើ `FROM nvcr.io/nvidia/pytorch:25.04-py3` |
| **តេស្តឃើញ `engine: Edge-TTS (fallback)`** | `voxcpm` មិនទាន់បាន install / weights មិនទាន់ទាញ | ពិនិត្យ worker logs; រង់ចាំ cold start ដំបូងបំពេញ |
| **Build បរាជ័យ `pip install ... exit code: 1`** | (១) NGC 25.04 កំណត់ `PIP_CONSTRAINT=/etc/pip/constraint.txt` ដែលចាក់សោរ package ទាំងអស់ → gradio 6 / datasets 3 / funasr / modelscope resolve មិនបាន ។ (២) `pip install torchaudio` បង្ខំឱ្យ pip ដូរផ្ទាំង CUDA stack ទាំងមូល (NGC torch ជា pre-release `2.7.0a0+<hash>`) | Dockerfile ថ្មីសរសេរ constraint ឡើងវិញ ដោយចាក់សោរតែ `torch`/`torchaudio` ដែលមានក្នុង image ។ **មិនត្រូវបន្ថែម `pip install torchaudio`/`torch` ក្នុង Dockerfile ឡើយ** — rebuild ជាមួយ commit ថ្មី |
| **Build បរាជ័យ ឬ warning ពី `torchcodec`** | VoxCPM ប្រកាស `torchcodec` ដោយគ្មានកំណត់ version ប៉ុន្តែ **មិនប្រើវាក្នុង code** ទេ — pip ទាញ 0.16 ដែលសម្រាប់ torch ≥ 2.11 | `requirements.txt` ចាក់សោរ `torchcodec==0.5` (កំណែចុងក្រោយដែលសម្រាប់ torch 2.7) |
| **`voxcpm` ប្តូរឥរិយាបថដោយខ្លួនឯង បន្ទាប់ពី rebuild** | ពីមុន requirements ទាញ `voxcpm` ពី branch `main` (ផ្លាស់ប្តូររាល់ថ្ងៃ) | `requirements.txt` ចាក់សោរ tag ជាក់លាក់ `git+...VoxCPM.git@2.0.3` |
| **សំឡេងខ្មែរមិនធម្មជាតិ** | Khmer អាចមិនមែនជាភាសា officially supported ក្នុង VoxCPM2 | សាកល្បង `VOXCPM_TIMESTEPS` ខ្ពស់ជាង (15-20), ប្រើ reference audio, ឬ LoRA fine-tune |

---

## 💡 ការបន្ថែម Custom Model Weights (Checkpoints)

ប្រសិនបើអ្នកមាន weights fine-tuned (`.safetensors`) នៃម៉ូដែល VoxCPM Khmer៖
1. ដាក់ទៅក្នុង Network Volume  e.g. `/workspace/models/VoxCPM-Khmer/`
2. កំណត់ `VOXCPM_MODEL_ID=VoxCPM-Khmer` (ឬទុក — worker រកឃើញ local checkpoint ដោយស្វ័យប្រវត្តិតាម `MODEL_PATH`)។
