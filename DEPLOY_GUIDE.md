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
2. តម្លៃ៖ **Size `30GB`**, Region ដូច Endpoint របស់អ្នក។
3. Volume នេះនឹង缓存 model weights (~៥GB+) ដូច្នេះ cold start លឿន និងមិនទាញឡើងវិញ។

---

## ជំហានទី ៤៖ បង្កើត Serverless Template

1. **Serverless** ➡️ **Templates** ➡️ **New Template**
2. បំពេញ៖
   * **Template Name**: `Khmer-TTS-Worker`
   * **Container Image**: `YOUR_DOCKERHUB_USERNAME/khmer-tts-worker:v2.0`
   * **Container Disk**: `25GB` (ឬ `30GB`)
   * **Volume Mount**: ភ្ជាប់ Network Volume ទៅ `/workspace`
   * **Environment Variables** (កំណត់រួចក្នុង image រួចហើយ — កែបានតាមត្រូវការ)：

     | Variable | តម្លៃលំនាំដើម | អត្ថន័យ |
     | :--- | :--- | :--- |
     | `VOXCPM_MODEL_ID` | `openbmb/VoxCPM2` | Model repo ឬ local checkpoint |
     | `VOXCPM_DEVICE` | `auto` | `cuda` / `cpu` / `auto` |
     | `VOXCPM_TIMESTEPS` | `10` | Diffusion steps (4-30, ច្រើន = គុណភាពល្អ តែយឺត) |
     | `VOXCPM_DENOISER` | `0` | `1` = ដាក់ denoiser សម្រាប់ reference audio |
     | `HF_HOME` | `/workspace/models/hf_cache` | ត្រូវនៅលើ Volume! |
     | `MODEL_PATH` | `/workspace/models` | Local checkpoint dir |
3. ចុច **Save Template**។

---

## ជំហានទី ៥៖ បង្កើត Serverless Endpoint

1. **Serverless** ➡️ **Endpoints** ➡️ **New Endpoint** ➡️ ជ្រើស Template `Khmer-TTS-Worker`
2. **GPU Types** (VoxCPM2 ជា model 2B — ត្រូវការ VRAM >= 16GB)៖
   * **RTX 4090 (24GB)** ឬ **RTX 3090** ឬ **A40** (ណែនាំ)
3. **Active Workers (Min)**: `0` | **Max Workers**: `2` | **Idle Timeout**: `60s`
4. ចុច **Create Endpoint**។

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
| **CUDA Out of Memory** | VRAM មិនគ្រប់គ្រាន់សម្រាប់ VoxCPM2 (2B) | ប្រើ GPU >= 16GB ឬ 24GB (RTX 3090/4090) |
| **តេស្តឃើញ `engine: Edge-TTS (fallback)`** | `voxcpm` មិនទាន់បាន install / weights មិនទាន់ទាញ | ពិនិត្យ worker logs; រង់ចាំ cold start ដំបូងបំពេញ |
| **សំឡេងខ្មែរមិនធម្មជាតិ** | Khmer អាចមិនមែនជាភាសា officially supported ក្នុង VoxCPM2 | សាកល្បង `VOXCPM_TIMESTEPS` ខ្ពស់ជាង (15-20), ប្រើ reference audio, ឬ LoRA fine-tune |

---

## 💡 ការបន្ថែម Custom Model Weights (Checkpoints)

ប្រសិនបើអ្នកមាន weights fine-tuned (`.safetensors`) នៃម៉ូដែល VoxCPM Khmer៖
1. ដាក់ទៅក្នុង Network Volume  e.g. `/workspace/models/VoxCPM-Khmer/`
2. កំណត់ `VOXCPM_MODEL_ID=VoxCPM-Khmer` (ឬទុក — worker រកឃើញ local checkpoint ដោយស្វ័យប្រវត្តិតាម `MODEL_PATH`)។
