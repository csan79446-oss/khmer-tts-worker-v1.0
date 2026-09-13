# 🚀 សៀវភៅណែនាំ Deploy AI Worker ទៅកាន់ RunPod Serverless (Khmer TTS / VoxCPM)

សៀវភៅណែនាំនេះនឹងបង្ហាញពីរបៀបយក **AI Worker** នៅក្នុង folder [runpod_worker/](file:///c:/App/Khmer_TTS_V2/runpod_worker) ទៅកាន់ **RunPod Serverless** ដើម្បីដំណើរការម៉ូដែល AI បង្កើតសំឡេងខ្មែរ (Voice Cloning & Natural Narration) ភ្ជាប់ជាមួយកម្មវិធី [Khmer_TTS_V2](file:///c:/App/Khmer_TTS_V2)។

---

## 📋 តម្រូវការជាមុន (Prerequisites)

1. **Docker Desktop** (ឬ Docker CLI) ដំឡើងលើកុំព្យូទ័ររបស់អ្នក
2. គណនី **Docker Hub** (ឥតគិតថ្លៃនៅ [hub.docker.com](https://hub.docker.com))
3. គណនី **RunPod** (ចុះឈ្មោះ និងបញ្ចូល Credit នៅ [runpod.io](https://runpod.io))

---

## ជំហានទី ១៖ ធ្វើតេស្តសាកល្បង Worker ក្នុងម៉ាស៊ីន (Local Test)

មុននឹងធ្វើការ Build និង Push ទៅ Cloud សូមធ្វើតេស្តសាកល្បង Handler នៅក្នុង Terminal របស់អ្នក៖

```powershell
cd c:\App\Khmer_TTS_V2
python runpod_worker/test_local.py
```

* ប្រសិនបើដំណើរការជោគជ័យ អ្នកនឹងឃើញសារ `[✓] Local worker test PASSED!`
* វានឹងបង្កើត file សំឡេងគំរូ [test_output.wav](file:///c:/App/Khmer_TTS_V2/runpod_worker/test_output.wav) ដើម្បីឱ្យអ្នកអាចបើកស្ដាប់បានភ្លាមៗ។

---

## ជំហានទី ២៖ Build Docker Image & Push ទៅកាន់ Docker Hub

1. បើក PowerShell ឬ Terminal រួចចូលទៅកាន់ folder `runpod_worker`៖
   ```powershell
   cd c:\App\Khmer_TTS_V2\runpod_worker
   ```

2. Login ចូលគណនី Docker Hub របស់អ្នក៖
   ```powershell
   docker login
   ```

3. Build Docker Image (សូមប្ដូរ `YOUR_DOCKERHUB_USERNAME` ទៅជា Username ពិតប្រាកដរបស់អ្នក)៖
   ```powershell
   docker build -t YOUR_DOCKERHUB_USERNAME/khmer-tts-worker:v1.0 .
   ```

4. Push Image ទៅកាន់ Docker Hub Repository៖
   ```powershell
   docker push YOUR_DOCKERHUB_USERNAME/khmer-tts-worker:v1.0
   ```

> [!TIP]
> ត្រូវប្រាកដថា Docker Hub Repository របស់អ្នកត្រូវបានកំណត់ជា **Public** (ឬប្រសិនបើ Private ត្រូវបន្ថែម Docker Credentials នៅក្នុង RunPod)។

---

## ជំហានទី ៣៖ បង្កើត Serverless Template នៅលើ RunPod Console

1. ចូលទៅកាន់គេហទំព័រ **[RunPod Console](https://www.runpod.io/console/serverless)**
2. ចុចលើម៉ឺនុយ **Serverless** ➡️ **Templates** ➡️ **New Template**
3. បំពេញព័ត៌មានដូចខាងក្រោម៖
   * **Template Name**: `Khmer-TTS-Worker`
   * **Container Image**: `YOUR_DOCKERHUB_USERNAME/khmer-tts-worker:v1.0`
   * **Container Disk**: `20 GB` (ឬ `30 GB` ប្រសិនបើមាន Model Weights ធំ)
   * **Environment Variables**:
     * `MODEL_PATH`: `/workspace/models`
4. ចុច **Save Template**។

---

## ជំហានទី ៤៖ បង្កើត Serverless Endpoint

1. ចូលទៅកាន់ **Serverless** ➡️ **Endpoints** ➡️ ចុច **New Endpoint**
2. ជ្រើសរើស **Template**: ជ្រើសរើស `Khmer-TTS-Worker` ដែលទើបបង្កើត
3. កំណត់ **GPU Types** (រើសយក GPU ដែលស័ក្តិសម និងសន្សំសំចៃ)៖
   * **RTX 4090 (24GB VRAM)** ឬ **RTX 3090** (ល្បឿនលឿន និងតម្លៃសមរម្យ)
   * ឬ **NVIDIA A40 / A4000**
4. កំណត់ **Active Workers & Scaling**៖
   * **Active Workers (Min)**: `0` (ដើម្បីកុំឱ្យអស់លុយពេលអត់ប្រើ)
   * **Max Workers**: `2` (ឬតាមតម្រូវការ)
   * **Idle Timeout**: `60` វិនាទី (ដើម្បីឱ្យ GPU បិទដោយស្វ័យប្រវត្តិនៅពេលគ្មានការងារ)
5. ចុច **Create Endpoint**។

---

## ជំហានទី ៥៖ យក API Key និង Endpoint ID

នៅពេល Endpoint បង្កើតរួចរាល់ អ្នកនឹងទទួលបាន៖
1. **Endpoint ID**: ជាកូដសម្គាល់ (ឧទាហរណ៍: `v2/abcd1234efgh`)
   * Endpoint URL ពេញលេញ: `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID`
2. **API Key**: ចូលទៅកាន់ Settings របស់ RunPod ➡️ **API Keys** ➡️ ចុច **+ API Key** រួចចម្លង Key នោះទុក (`rpa_...`)។

---

## ជំហានទី ៦៖ ភ្ជាប់ជាមួយកម្មវិធី Khmer TTS V2

1. បើកកម្មវិធី [Khmer_TTS_V2](file:///c:/App/Khmer_TTS_V2)៖
   ```powershell
   python main.py
   ```
2. ចូលទៅកាន់ម៉ឺនុយ **Settings** (រូបសញ្ញា Gear ⚙️) ➡️ ជ្រើសរើស Tab **"VoxCPM & RunPod"**
3. បំពេញ៖
   * **RunPod API Key**: បិទភ្ជាប់ API Key របស់អ្នក
   * **RunPod Endpoint URL**: `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID`
   * ដោះធីក **"Enable Mock VoxCPM Mode"** (ប្រសិនបើចង់ប្រើ Serverless ពិតប្រាកដ)
4. ចុចប៊ូតុង **"Test Connection"**
   * ប្រសិនបើជោគជ័យ កម្មវិធីនឹងបង្ហាញ Status ពណ៌បៃតង៖ `[OK] Endpoint reachable and authenticated.`
5. ចុច **Save Changes**។

---

## 🛠️ ការដោះស្រាយបញ្ហា (Troubleshooting)

| បញ្ហា | មូលហេតុដែលអាចកើតមាន | ដំណោះស្រាយ |
| :--- | :--- | :--- |
| **Authentication Failed (401/403)** | API Key មិនត្រឹមត្រូវ | ពិនិត្យមើល RunPod API Key ក្នុង Settings ឡើងវិញ |
| **Endpoint Not Found (404)** | Endpoint ID ខុស | ពិនិត្យ URL ត្រូវតែមានទម្រង់ `https://api.runpod.ai/v2/{ENDPOINT_ID}` |
| **Job Timeout / Cold Start** | GPU ចំណាយពេលទាញ Image ដំបូង | បង្កើន `RunPod Timeout` ក្នុង Settings ទៅ 180s ឬ 300s សម្រាប់ការហៅលើកដំបូង |
| **CUDA Out of Memory** | VRAM មិនគ្រប់គ្រាន់ | ជ្រើសរើស GPU ដែលមាន VRAM ចាប់ពី 16GB ឬ 24GB ឡើងទៅ (RTX 3090 / 4090) |

---

## 💡 ការបន្ថែម Custom Model Weights (Checkpoints)

ប្រសិនបើអ្នកមាន Checkpoint Weights ផ្ទាល់ខ្លួន (ឧទាហរណ៍ `.pt` ឬ `.safetensors` នៃម៉ូដែល VoxCPM Khmer)៖
1. អ្នកអាចដាក់ Folder `models/` ចូលទៅក្នុង `runpod_worker/models/` មុនពេល `docker build`
2. ឬប្រើ **RunPod Network Volume** ភ្ជាប់ទៅកាន់ `/workspace/models` ដើម្បីកុំឱ្យ Docker Image មានទំហំធំពេក។
