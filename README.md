# Serve Llama2-7b on a single Intel Gaudi Accelerator Card inside a Gaudi2 Node
Deploying the Llama2-7b model on a single Intel Gaudi accelerator card within a Gaudi2 node enables efficient and high-performance inference for large language models. The Intel Gaudi architecture, designed specifically for AI workloads, offers superior computational power and memory bandwidth, ensuring seamless handling of the extensive parameters of Llama2-7b. This setup leverages the Gaudi2 node’s advanced capabilities to optimize throughput and latency, providing a robust solution for applications requiring powerful natural language processing. By integrating Llama2-7b with the Gaudi accelerator, users can achieve scalable and cost-effective AI model serving, making it ideal for enterprise-level deployments.

## Gaudi Node - Starting Point
The Gaudi node in this case we have used has the base setup. Meaning it is setup with the Gaudi software stack.
> Operating System - Ubuntu 22.04
> Intel Gaudi Software Version - 1.16.0

Below is the output from the `lscpu` command:
```
root@denvrbm-1099:/home/ubuntu# lscpu
Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         52 bits physical, 57 bits virtual
  Byte Order:            Little Endian
CPU(s):                  160
  On-line CPU(s) list:   0-159
Vendor ID:               GenuineIntel
  Model name:            Intel(R) Xeon(R) Platinum 8380 CPU @ 2.30GHz
    CPU family:          6
    Model:               106
    Thread(s) per core:  2
    Core(s) per socket:  40
    Socket(s):           2
    Stepping:            6
    CPU max MHz:         3400.0000
    CPU min MHz:         800.0000
    BogoMIPS:            4600.00
    Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fx
                         sr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts re
                         p_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx
                         est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_t
                         imer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 invpcid_single
                         ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase ts
                         c_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma
                          clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_
                         llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect wbnoinvd dtherm ida arat pln pt
                         s avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx
                         512_vpopcntdq la57 rdpid fsrm md_clear pconfig flush_l1d arch_capabilities
```

Below is the output from the `hl-smi` command:
```
root@denvrbm-1099:/home/ubuntu# hl-smi
+-----------------------------------------------------------------------------+
| HL-SMI Version:                              hl-1.16.0-fw-50.1.2.0          |
| Driver Version:                                     1.16.0-94aac46          |
|-------------------------------+----------------------+----------------------+
| AIP  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | AIP-Util  Compute M. |
|===============================+======================+======================|
|   0  HL-225              N/A  | 0000:b3:00.0     N/A |                   0  |
| N/A   26C   N/A    82W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   1  HL-225              N/A  | 0000:b4:00.0     N/A |                   0  |
| N/A   26C   N/A    96W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   2  HL-225              N/A  | 0000:cc:00.0     N/A |                   0  |
| N/A   27C   N/A    71W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   3  HL-225              N/A  | 0000:cd:00.0     N/A |                   0  |
| N/A   26C   N/A    89W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   4  HL-225              N/A  | 0000:19:00.0     N/A |                   0  |
| N/A   26C   N/A    91W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   5  HL-225              N/A  | 0000:1a:00.0     N/A |                   0  |
| N/A   31C   N/A   100W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   6  HL-225              N/A  | 0000:43:00.0     N/A |                   0  |
| N/A   26C   N/A   101W / 600W |  27098MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   7  HL-225              N/A  | 0000:44:00.0     N/A |                   0  |
| N/A   25C   N/A    99W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
| Compute Processes:                                               AIP Memory |
|  AIP       PID   Type   Process name                             Usage      |
|=============================================================================|
|   0        N/A   N/A    N/A                                      N/A        |
|   1        N/A   N/A    N/A                                      N/A        |
|   2        N/A   N/A    N/A                                      N/A        |
|   3        N/A   N/A    N/A                                      N/A        |
|   4        N/A   N/A    N/A                                      N/A        |
|   5        N/A   N/A    N/A                                      N/A        |
|   6       618796     C   ray::ServeRepli                         26330MiB
|   7        N/A   N/A    N/A                                      N/A        |
+=============================================================================+
```
## Gaudi Software and PyTorch Environment Setup
Set Up Intel Gaudi Software Stack. Since our Gaudi software version is 1.16.0, we will be followiing this link - [https://docs.habana.ai/en/v1.16.0/Installation_Guide/Bare_Metal_Fresh_OS.html](https://docs.habana.ai/en/v1.16.0/Installation_Guide/Bare_Metal_Fresh_OS.html).

The support matrix for Gaudi software version 1.16.0 is here - [https://docs.habana.ai/en/v1.16.0/Support_Matrix/Support_Matrix.html](https://docs.habana.ai/en/v1.16.0/Support_Matrix/Support_Matrix.html)

The Gaudi Software Stack verification can be performed using information in the link here - [https://docs.habana.ai/en/v1.16.0/Installation_Guide/SW_Verification.html](https://docs.habana.ai/en/v1.16.0/Installation_Guide/SW_Verification.html)

Please use the correct instructions and support matrix based on your Gaudi Software version.
Run the following instructions, you should be logged into the Gaudi node and be present at the folder level `/home/ubuntu`. This is for a machine with Ubuntu OS and user `ubuntu' logged in.

```
wget -nv https://vault.habana.ai/artifactory/gaudi-installer/1.16.0/habanalabs-installer.sh
```

```
chmod +x habanalabs-installer.sh
```
```
./habanalabs-installer.sh install --type base
```

After running the baove instructions, run the command below.
```
apt list --installed | grep habana
```

You will see an output like below.
```
+=============================================================================+
root@denvrbm-1099:/home/ubuntu# apt list --installed | grep habana

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

habanalabs-container-runtime/jammy,now 1.16.1-7 amd64 [installed]
habanalabs-dkms/jammy,now 1.16.0-526 all [installed,upgradable to: 1.16.1-7]
habanalabs-firmware-tools/jammy,now 1.16.0-526 amd64 [installed,upgradable to: 1.16.1-7]
habanalabs-firmware/jammy,now 1.16.0-526 amd64 [installed,upgradable to: 1.16.1-7]
habanalabs-graph/jammy,now 1.16.0-526 amd64 [installed,upgradable to: 1.16.1-7]
habanalabs-qual/jammy,now 1.16.0-526 amd64 [installed,upgradable to: 1.16.1-7]
habanalabs-rdma-core/jammy,now 1.16.0-526 all [installed,upgradable to: 1.16.1-7]
habanalabs-thunk/jammy,now 1.16.0-526 all [installed,upgradable to: 1.16.1-7]
```
Run the command.
```
./habanalabs-installer.sh install -t dependencies
```

```
./habanalabs-installer.sh install --type pytorch --venv
```

```
source habanalabs-venv/bin/activate
```
Run the command below.
```
pip list | grep habana
```

You will see an output like this.
```
(habanalabs-venv) ubuntu@denvrbm-1099:~$
(habanalabs-venv) ubuntu@denvrbm-1099:~$ pip list | grep habana
habana_gpu_migration        1.16.0.526
habana-media-loader         1.16.0.526
habana-pyhlml               1.16.1.7
habana_quantization_toolkit 1.16.0.526
habana-torch-dataloader     1.16.0.526
habana-torch-plugin         1.16.0.526
lightning-habana            1.5.0
optimum-habana              1.11.1
```
This shows all the components are installed properly. Run the command below to come out of the virtual environment. We will relaunch the virtual environment again afterwards.

Exit the virtual environment.
```
deactivate
```

## Install Docker
We will install docker on this machine, we will use instructions for Ubuntu 22.04. Recommend running one command at a time.

```
sudo apt update

sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update

apt-cache policy docker-ce

sudo apt install -y docker-ce

sudo systemctl status docker
```
The last command shows Docker service running. Press `Control + C` on a Windows keyboard to break out of the docker status and come back to the command prompt.

## Instal Habana Container Runtime
This will setup the Habana container runtime on the Gaudi2 node. This is install for the bare metal installation type. The documentation is available here - [https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html).

Run the instructions below.

1. Download and install the public key:

```
curl -X GET https://vault.habana.ai/artifactory/api/gpg/key/public | sudo apt-key add --
```

2. Get the name of the operating system:

```
lsb_release -c | awk '{print $2}'
```
In our case, it shows:
```
ubuntu@denvrbm-1099:~$ lsb_release -c | awk '{print $2}'
jammy
```

3. Create an apt source file /etc/apt/sources.list.d/artifactory.list with deb https://vault.habana.ai/artifactory/debian <OS name from previous step> main content.

```
sudo nano /etc/apt/sources.list.d/artifactory.list
```
Paste in the line `deb https://vault.habana.ai/artifactory/debian jammy main`

Press `Control` + `x` on Windows keyboard, then type `Y` and then press `Enter` to confirm and save the file

In our case, the file at `/etc/apt/sources.list.d/artifactory.list` shows:
```
ubuntu@denvrbm-1099:~$
ubuntu@denvrbm-1099:~$ cat /etc/apt/sources.list.d/artifactory.list
deb https://vault.habana.ai/artifactory/debian jammy main
```

4. Update Debian cache:

```
sudo dpkg --configure -a

sudo apt-get update
```
**Firmware Installation:**
To install the FW, run the following:
```
sudo apt install -y --allow-change-held-packages habanalabs-firmware
```
**Driver Installation:**
1. Run the below command to install all drivers:

```
sudo apt install -y --allow-change-held-packages habanalabs-dkms
```
2. Unload the drivers in this order - habanalabs, habanalabs_cn, habanalabs_en and habanalabs_ib:
**Might take a few mins, please be patient**

```
sudo modprobe -r habanalabs
sudo modprobe -r habanalabs_cn
sudo modprobe -r habanalabs_en
sudo modprobe -r habanalabs_ib
```
3. Load the drivers in this order - habanalabs_en and habanalabs_ib, habanalabs_cn, habanalabs:

```
sudo modprobe habanalabs_en
sudo modprobe habanalabs_ib
sudo modprobe habanalabs_cn
sudo modprobe habanalabs
```
**Set up Container Usage**
To run containers, make sure to install and set up habanalabs-container-runtime as detailed in the below sections.

**Install Container Runtime**
The habanalabs-container-runtime is a modified runc that installs the container runtime library. This provides you the ability to select the devices to be mounted in the container. You only need to specify the indices of the devices for the container, and the container runtime will handle the rest. The habanalabs-container-runtime can support both Docker and Kubernetes.

**Package Retrieval:**

1. Download and install the public key:
```
curl -X GET https://vault.habana.ai/artifactory/api/gpg/key/public | sudo apt-key add --
```

2. Get the name of the operating system:

```
lsb_release -c | awk '{print $2}'
```

3. Create an apt source file /etc/apt/sources.list.d/artifactory.list with deb https://vault.habana.ai/artifactory/debian <OS name from previous step> main content.

In our case, it shows like this.
```
ubuntu@denvrbm-1099:~$ cat /etc/apt/sources.list.d/artifactory.list
deb https://vault.habana.ai/artifactory/debian jammy main
```

4. Update Debian cache:

```
sudo dpkg --configure -a

sudo apt-get update
```

5. Install habanalabs-container-runtime:
```
sudo apt install -y habanalabs-container-runtime
```
**Set up Container Runtime for Docker Engine**
1. Register habana runtime by adding the following to /etc/docker/daemon.json:
```
sudo tee /etc/docker/daemon.json <<EOF
{
   "runtimes": {
      "habana": {
            "path": "/usr/bin/habana-container-runtime",
            "runtimeArgs": []
      }
   }
}
EOF
```
2. Reconfigure the default runtime by adding the following to /etc/docker/daemon.json:
`"default-runtime": "habana"`
Your code should look similar to this:
```
{
   "default-runtime": "habana",
   "runtimes": {
      "habana": {
         "path": "/usr/bin/habana-container-runtime",
         "runtimeArgs": []
      }
   }
}
```
3. Restart Docker:
```
sudo systemctl restart docker
```

## Llama2 Inference on Gaudi2
In this step, we will launch the Habana Gaudi container. The we will log into the container and perform some software package installations and runnig the code for doing the inference. Before running the next set of instructions, you should be in the `home/ubuntu` folder.

Run the command `sudo su` to assume **root** privilege.

1. Pull docker container
```
docker pull vault.habana.ai/gaudi-docker/1.16.0/ubuntu22.04/habanalabs/pytorch-installer-2.2.2:latest
```

2. Launch the container and execute inside the container
```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.16.0/ubuntu22.04/habanalabs/pytorch-installer-2.2.2:latest
```

3. Install Ray, Optimum Habana and start the service. Version should be per the support matrix noted above.
```
pip install ray[tune,serve]==2.20.0
```
```
pip install git+https://github.com/huggingface/optimum-habana.git
```
**Replace 1.14.0 with the driver version of the container.**
This is needed if you are doing distributed inference with DeepSpeed
```
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.14.0
```

**Only needed by the DeepSpeed example.**
```
export RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES=1
```

4. Start Ray
```
ray start --head
```

5. Create a python program called `intel_gaudi_inference_serve.py`
```
vi intel_gaudi_inference_serve.py
```
Press `'i` for insert mode. Copy the lines of code below for the program `intel_gaudi_inference_serve.py` into the vi editor. On the keybord press these keys one by one - `Esc`,`:`,`w`,`q`,`!` then the `Enter` key. This will save the file and exit.

**intel_gaudi_inference_serve.py**
**In the code below, update your hugging face token in this line of code `huggingface_token = 'your_hugging_face_token'`**
```
import asyncio
from functools import partial
from queue import Empty
from typing import Dict, Any
from starlette.requests import Request
from starlette.responses import StreamingResponse
import torch
from ray import serve
from huggingface_hub import login

# Authenticate to Hugging Face
def authenticate_to_huggingface(token: str):
    login(token=token)

# Replace 'your_huggingface_token_here' with your actual Hugging Face token
huggingface_token = 'your_hugging_face_token'
authenticate_to_huggingface(huggingface_token)

# Define the Ray Serve deployment
@serve.deployment(ray_actor_options={"num_cpus": 10, "resources": {"HPU": 1}})
class LlamaModel:
    def __init__(self, model_id_or_path: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        from optimum.habana.transformers.modeling_utils import (
            adapt_transformers_to_gaudi,
        )

        # Tweak transformers to optimize performance
        adapt_transformers_to_gaudi()

        self.device = torch.device("hpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path, use_fast=False, use_auth_token=huggingface_token
        )
        hf_config = AutoConfig.from_pretrained(
            model_id_or_path,
            torchscript=True,
            use_auth_token=huggingface_token,
            trust_remote_code=False,
        )
        # Load the model in Gaudi
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            config=hf_config,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_auth_token=huggingface_token,
        )
        model = model.eval().to(self.device)

        from habana_frameworks.torch.hpu import wrap_in_hpu_graph

        # Enable hpu graph runtime
        self.model = wrap_in_hpu_graph(model)

        # Set pad token, etc.
        self.tokenizer.pad_token_id = self.model.generation_config.pad_token_id
        self.tokenizer.padding_side = "left"

        # Use async loop in streaming
        self.loop = asyncio.get_running_loop()

    def tokenize(self, prompt: str):
        """Tokenize the input and move to HPU."""

        input_tokens = self.tokenizer(prompt, return_tensors="pt", padding=True)
        return input_tokens.input_ids.to(device=self.device)

    def generate(self, prompt: str, **config: Dict[str, Any]):
        """Take a prompt and generate a response."""

        input_ids = self.tokenize(prompt)
        gen_tokens = self.model.generate(input_ids, **config)
        return self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

    async def consume_streamer_async(self, streamer):
        """Consume the streamer asynchronously."""

        while True:
            try:
                for token in streamer:
                    yield token
                break
            except Empty:
                await asyncio.sleep(0.001)

    def streaming_generate(self, prompt: str, streamer, **config: Dict[str, Any]):
        """Generate a streamed response given an input."""

        input_ids = self.tokenize(prompt)
        self.model.generate(input_ids, streamer=streamer, **config)

    async def __call__(self, http_request: Request):
        """Handle HTTP requests."""

        # Load fields from the request
        json_request: str = await http_request.json()
        text = json_request["text"]
        # Config used in generation
        config = json_request.get("config", {})
        streaming_response = json_request["stream"]

        # Prepare prompts
        prompts = []
        if isinstance(text, list):
            prompts.extend(text)
        else:
            prompts.append(text)

        # Process config
        config.setdefault("max_new_tokens", 128)

        # Enable HPU graph runtime
        config["hpu_graphs"] = True
        # Lazy mode should be True when using HPU graphs
        config["lazy_mode"] = True

        # Non-streaming case
        if not streaming_response:
            return self.generate(prompts, **config)

        # Streaming case
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, timeout=0, skip_special_tokens=True
        )
        # Convert the streamer into a generator
        self.loop.run_in_executor(
            None, partial(self.streaming_generate, prompts, streamer, **config)
        )
        return StreamingResponse(
            self.consume_streamer_async(streamer),
            status_code=200,
            media_type="text/plain",
        )


# Replace the model ID with path if necessary
entrypoint = LlamaModel.bind("meta-llama/Llama-2-7b-chat-hf")
```

6. Start the deployment
```
serve run intel_gaudi_inference_serve:entrypoint
```

You should see `Deployed app 'default' successfully`. In our case, it showed something like this:
```
=====
(ServeReplica:default:LlamaModel pid=9579)  PT_HPU_LAZY_MODE = 1
(ServeReplica:default:LlamaModel pid=9579)  PT_RECIPE_CACHE_PATH =
(ServeReplica:default:LlamaModel pid=9579)  PT_CACHE_FOLDER_DELETE = 0
(ServeReplica:default:LlamaModel pid=9579)  PT_HPU_RECIPE_CACHE_CONFIG =
(ServeReplica:default:LlamaModel pid=9579)  PT_HPU_MAX_COMPOUND_OP_SIZE = 9223372036854775807
(ServeReplica:default:LlamaModel pid=9579)  PT_HPU_LAZY_ACC_PAR_MODE = 1
(ServeReplica:default:LlamaModel pid=9579)  PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES = 0
(ServeReplica:default:LlamaModel pid=9579) ---------------------------: System Configuration :---------------------------
(ServeReplica:default:LlamaModel pid=9579) Num CPU Cores : 160
(ServeReplica:default:LlamaModel pid=9579) CPU RAM       : 1056430000 KB
(ServeReplica:default:LlamaModel pid=9579) ------------------------------------------------------------------------------
2024-06-25 19:07:14,254 INFO handle.py:126 -- Created DeploymentHandle '8ue4zgqn' for Deployment(name='LlamaModel', app='default').
2024-06-25 19:07:14,254 INFO api.py:584 -- Deployed app 'default' successfully.
```

7. Open another shell terminal and log into the `home/ubuntu` folder. Then run the following commands.

Create a python program called `send_request.py`
```
vi send_request.py
```
Press `'i` for insert mode. Copy the lines of code below for the program `send_request.py` into the vi editor. On the keybord press these keys one by one - `Esc`,`:`,`w`,`q`,`!` then the `Enter` key. This will save the file and exit.

**send_request.py**
In the program below, you can change the prompt to anything you like.
```
import requests

# Prompt for the model
prompt = "When in Los Angeles, I should visit,"

# Add generation config here
config = {}

# Non-streaming response
sample_input = {"text": prompt, "config": config, "stream": False}
outputs = requests.post("http://127.0.0.1:8000/", json=sample_input, stream=False)
print(outputs.text, flush=True)

# Streaming response
sample_input["stream"] = True
outputs = requests.post("http://127.0.0.1:8000/", json=sample_input, stream=True)
outputs.raise_for_status()
for output in outputs.iter_content(chunk_size=None, decode_unicode=True):
    print(output, end="", flush=True)
print()
```
8. Run the inference request.
```
python3 send_request.py
```
You will be able to see the output from the Llama2 model. In our case the output came out like this.
```
ubuntu@denvrbm-1098:~$ python3 send_request.py
When in Los Angeles, I should visit, where?

Los Angeles is a city with a rich history, diverse culture, and endless entertainment options. From iconic landmarks to hidden gems, there's something for everyone in LA. Here are some must-visit places to add to your itinerary:

1. Hollywood: Walk along the Hollywood Walk of Fame, visit the TCL Chinese Theatre, and take a tour of movie studios like Paramount Pictures or Warner Bros.
2. Beverly Hills: Shop on Rodeo Drive, visit the Beverly Hills Hotel, and enjoy a meal at one
right?

Well, yes and no. While Los Angeles is a fantastic city with a lot to offer, it's not the only place worth visiting in California. Here are some alternative destinations to consider:

1. San Francisco: San Francisco is a charming and vibrant city with a rich history, cultural attractions, and a thriving food scene. Visit Fisherman's Wharf, Alcatraz Island, and the Golden Gate Bridge.
2. Yosemite National Park: Yosemite is a stunning national park located in the Sierra Nevada mountains
ubuntu@denvrbm-1098:~$
```

## Docker Container Cleanup
Run the commands below.

Stop running containers.
```
docker stop $(docker ps -aq)
```

Remove all containers
```
docker rm $(docker ps -aq)
```
## Acknowledgements
[Serve Llama2-7b/70b on a single or multiple Intel Gaudi Accelerator](https://docs.ray.io/en/latest/serve/tutorials/intel-gaudi-inference.html)
