> 请注意 以下为ai生成 我仅提供了过程命令 仅能保证这个过程在我这里成立

用两台 NVIDIA DGX Spark 本地部署 DeepSeek V4 Flash 0731：从开机、RoCE 到 vLLM TP=2 的完整踩坑记录

本文记录我使用 2 台 NVIDIA DGX Spark，通过 ConnectX-7 / RoCE 组成双机集群，并最终在本地跑起 DeepSeek-V4-Flash-0731 + DSpark + vLLM TP=2 的完整过程。

这不是一篇“理论上应该能跑”的安装教程，而是我实际部署时留下来的操作记录整理版。过程中我碰到了 DGX Spark 更新后 NVIDIA 驱动异常、Docker 状态损坏、Netplan 配置文件损坏、代理、模型同步等问题，所以也把这些修复过程保留下来。

本文整理于 2026 年 8 月 7 日。DGX Spark 的系统、驱动以及相关 vLLM 镜像更新都比较快，尤其是内核版本、驱动版本和仓库默认参数，后面很可能继续变化，因此不要机械照抄所有版本号。

为什么要用两台 DGX Spark？

DGX Spark 使用 NVIDIA GB10 Grace Blackwell Superchip，每台机器拥有 128 GB LPDDR5x 统一内存，并带有 ConnectX-7 高速网络接口。NVIDIA 官方规格给出的定位是：单机可以处理最高约 200B 参数级模型，双 Spark 配置则可以覆盖到约 405B 参数级模型。

DeepSeek-V4-Flash 本身是一个 MoE 模型。官方模型信息中给出的规模是 284B 总参数、13B 激活参数，并原生支持 1M token 上下文。

所以 V4 Flash 恰好是一个很适合双 DGX Spark 折腾的模型。

这里并不是把两台机器的内存变成一个真正意义上的“256 GB 单机统一内存池”，而是通过 Tensor Parallel，TP=2 将模型计算拆到两个节点。NVIDIA 自己的双 Spark 示例也是类似思路：两台机器通过高速链路交换中间结果，让一个超过单 Spark 容量的模型作为一个逻辑实例运行。

我最后使用的是 MiaAI-Lab 的：

DeepSeek-v4-Flash-DSpark-2x-DGX-Spark

目前仓库已经更新为针对 DeepSeek-V4-Flash-0731 的双 DGX Spark recipe，使用 vLLM TP=2、DSpark speculative decoding，以及实验性的 "nvfp4_ds_mla" KV Cache 路径；默认运行镜像是：

ghcr.io/anemll/dspark-vllm-gx10:0.1.1

我的最终拓扑大致如下：

                         普通局域网 / Wi-Fi
                                │
                         SSH / 下载 / API
                                │
                    ┌───────────┴───────────┐
                    │                                                            │
              DGX Spark #1                                      DGX Spark #2
                 Head                                                     Worker
                    │                                                            │
            192.168.100.10                                  192.168.100.11
                    │                                                            │
                    └──── ConnectX-7 ───────────┘
                         RoCE / NCCL
                         vLLM TP=2

              第二组接口也配置为：
            192.168.101.10 ↔ 192.168.101.11

              OpenAI-compatible API
                 Head :8888/v1

实际的 vLLM/NCCL 数据面我使用 "192.168.100.0/24" 这一组网络。

---

1. DGX Spark 第一次开机

DGX Spark 第一次开机其实很简单。

我直接接了 Type-C 键盘鼠标、HDMI 显示器，再接电源完成初始化。配置好无线网络之后让系统执行自己的更新流程。更新完成后机器会重启；如果没有自动重启，我建议自己重启一次。我的两台机器完成初始化以后 SSH 已经可以直接使用，所以之后显示器、鼠标和键盘就全部拔掉了，后面的工作都通过 SSH 完成。

NVIDIA 当前官方文档同样支持这种模式：第一次可以接显示器、键鼠完成初始化，之后直接把 DGX Spark 当网络设备，通过 SSH 使用即可。官方还特别提到，部分显示器在 USB-C/DisplayPort 下可能存在兼容问题，此时 HDMI 是更稳妥的方案。

先找到机器 IP：

ip a

然后从自己的电脑连接：

ssh dgx@<DGX_SPARK_IP>

之后整套部署基本都不再需要本地显示器。

---

2. 第一波坑：一台驱动异常，两台 Docker 都出了问题

我这批机器初始化之后并不是完全健康。

其中一台在系统更新过程中出现了 NVIDIA 驱动/内核包状态异常；两台机器的 Docker 也都出现了启动问题。

这里建议不要急着开始配双机。先保证两台机器分别满足：

nvidia-smi
docker info

都能正常执行，再往下做。

2.1 修 Docker

我的机器是新机器，没有需要保留的容器和镜像，因此处理方式比较直接：

sudo systemctl stop docker.service docker.socket

sudo rm -rf /var/lib/docker/buildkit
sudo rm -f /run/docker.pid /var/run/docker.pid

sudo systemctl reset-failed docker
sudo systemctl start docker

sudo systemctl is-active docker
sudo docker info

然后重启：

sudo reboot

回来以后重新确认：

sudo systemctl is-active docker
sudo docker info

这就是我当时实际使用的修复步骤。

注意："rm -rf" 不应该被当成通用 Docker 修复命令。 我的场景是刚初始化的新机器，本身没有任何需要保留的镜像、容器和 BuildKit 数据。如果机器已经有生产容器或者本地镜像，先搞清楚损坏位置并做好备份，不要照抄删除命令。

---

3. 修复更新后异常的 NVIDIA 驱动

其中一台机器的情况更麻烦：升级过程里驱动和内核软件包没有处于完整配置状态。

我先检查磁盘和 dpkg：

df -h / /boot /boot/efi 2>/dev/null || true

sudo dpkg --audit
sudo apt-get check || true

然后先把已经下载、但没有配置完成的软件包继续配置：

sudo dpkg --configure -a 2>&1 | tee ~/dpkg-configure.log

修依赖：

sudo apt-get --fix-broken install 2>&1 | tee ~/apt-fix-broken.log

再配置一次：

sudo dpkg --configure -a 2>&1 | tee ~/dpkg-configure-2.log

接着：

sudo apt-get update
sudo apt-get full-upgrade 2>&1 | tee ~/apt-full-upgrade.log

我当时出问题的版本对应：

nvidia-driver-580-open
linux-modules-nvidia-580-open-6.17.0-1029-nvidia
linux-modules-nvidia-580-open-nvidia-hwe-24.04

所以我检查的是：

dpkg-query -W \
  -f='${db:Status-Abbrev} ${Package} ${Version}\n' \
  nvidia-driver-580-open \
  linux-modules-nvidia-580-open-6.17.0-1029-nvidia \
  linux-modules-nvidia-580-open-nvidia-hwe-24.04 \
  2>/dev/null

以及对应内核：

ls -lh /boot/vmlinuz-6.17.0-1029-nvidia \
       /boot/initrd.img-6.17.0-1029-nvidia 2>/dev/null

检查 NVIDIA kernel module：

find /lib/modules/6.17.0-1029-nvidia \
  -type f -name 'nvidia*.ko*' -print 2>/dev/null

我的目标是让相关 package 状态重新变成 "ii"。

确认包完整以后：

sudo depmod -a 6.17.0-1029-nvidia
sudo update-initramfs -u -k 6.17.0-1029-nvidia
sudo update-grub
sudo reboot

这是我那台故障机器对应的真实版本。

这里最重要的一句话是：

不要把 "6.17.0-1029-nvidia" 当成 DGX Spark 永久固定版本。

以后照这篇文章部署，先看：

uname -r

再根据机器当前安装的软件包和内核版本处理。

DGX Spark 的 DGX OS 基于 Ubuntu 24.04 LTS，同时集成 NVIDIA 驱动、CUDA 及相关组件，所以普通 Ubuntu 的升级思路可以参考，但不要无脑替换 NVIDIA 提供的整套驱动栈。

两台机器都恢复正常以后，我统一又执行：

sudo apt update
sudo apt full-upgrade -y
sudo reboot

---

4. 如果机器需要代理

我的环境下载 GitHub、GHCR、APT 等资源时需要代理，所以这里也记录一下。

临时给 APT 指代理：

sudo apt \
  -o Acquire::http::Proxy="<PROXY_URL>" \
  -o Acquire::https::Proxy="<PROXY_URL>" \
  update

Snap：

sudo snap set system proxy.http="<PROXY_URL>"
sudo snap set system proxy.https="<PROXY_URL>"

snap get system proxy

升级：

sudo apt \
  -o Acquire::http::Proxy="<PROXY_URL>" \
  -o Acquire::https::Proxy="<PROXY_URL>" \
  full-upgrade -y

不需要以后取消 Snap 代理：

sudo snap unset system proxy.http
sudo snap unset system proxy.https
sudo snap get system proxy

这些是我部署时实际采用的代理处理方式。

后面 Git、Docker 还要分别处理，因为 Shell 的 "HTTP_PROXY" 并不意味着 Docker daemon 一定继承代理。

---

5. 配双机 ConnectX-7 / RoCE 网络

这是双 DGX Spark 最关键的一步。

NVIDIA 官方现在明确给出了 DGX Spark 的 ConnectX-7 拓扑：每台机器背后有两个 QSFP 端口，每个端口最高 200 Gb/s。更容易让人困惑的是，一个物理 QSFP 端口在 Linux 下会对应多个 Ethernet/RoCE 接口。

这里有个非常容易误判的细节。

我机器上的接口包含：

enp1s0f1np1
enP2p1s0f1np1

第二个里面的 大写 "P" 不是笔误。

NVIDIA 当前官方文档给出的 "ibdev2netdev" 映射里就是：

rocep1s0f1   -> enp1s0f1np1
roceP2p1s0f1 -> enP2p1s0f1np1

因此第一步最好先执行：

ip -br link
ibdev2netdev

确认自己机器实际使用的接口名。

Head 节点

我的 Head 节点配置：

sudo tee /etc/netplan/40-cx7.yaml >/dev/null <<'EOF'
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses:
        - 192.168.100.10/24
      dhcp4: false
      optional: true
    enP2p1s0f1np1:
      addresses:
        - 192.168.101.10/24
      dhcp4: false
      optional: true
EOF

sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan generate
sudo netplan apply

Worker 节点

第二台：

sudo tee /etc/netplan/40-cx7.yaml >/dev/null <<'EOF'
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses:
        - 192.168.100.11/24
      dhcp4: false
      optional: true
    enP2p1s0f1np1:
      addresses:
        - 192.168.101.11/24
      dhcp4: false
      optional: true
EOF

sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan generate
sudo netplan apply

这就是我的实际双节点地址规划。

至少先确认：

ping 192.168.100.11

Head 能直接找到 Worker。

Netplan 的一个坑

我还碰到过一次很诡异的 Netplan 错误。

最后发现 "/etc/netplan/" 里面存在一个损坏的 NetworkManager 自动生成 YAML。我当时的文件是：

BAD='/etc/netplan/90-NM-2edf06d6-a4d8-4336-95d9-d76c293f806f.yaml'

处理方法不是直接删，而是先挪走：

sudo mkdir -p /root/netplan-bak

sudo mv "$BAD" \
  "/root/netplan-bak/$(basename "$BAD").corrupt.$(date +%Y%m%d-%H%M%S)"

sudo netplan generate
sudo netplan try --timeout 60

如果 "netplan generate" 莫名其妙报 YAML 错误，非常值得把 "/etc/netplan/" 下面的所有配置文件都检查一遍，而不是只盯着自己刚写的 "40-cx7.yaml"。

---

6. 配置 Head → Worker SSH 免密

这个仓库需要 Head 自动操作 Worker，所以 SSH 免密是必须的。

在 Head：

mkdir -p ~/.ssh
chmod 700 ~/.ssh

test -f ~/.ssh/id_ed25519 || \
  ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519

先测试 Worker：

ssh -o StrictHostKeyChecking=accept-new \
  dgx@192.168.100.11 hostname

第一次会要求输入 Worker 的密码。

然后：

ssh-copy-id dgx@192.168.100.11

再做一次真正的无交互测试：

ssh -o BatchMode=yes dgx@192.168.100.11 '
hostname
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
docker version --format "{{.Server.Version}}"
'

如果这一步仍然需要密码，就不要继续部署。

NVIDIA 官方的双 Spark playbook 同样把“高速互联 + passwordless SSH”作为组成可用双节点环境的核心步骤。

---

7. 关闭 earlyoom

这是一个很容易忽略，但对 DGX Spark 跑大模型非常重要的步骤。

执行：

sudo systemctl stop earlyoom
sudo systemctl disable earlyoom

两台机器都做。

MiaAI-Lab 当前仓库明确建议在 DGX Spark Host 上关闭 "earlyoom"。原因是大上下文、高内存压力场景下，它可能提前把 vLLM Head 或 Worker 杀掉，即使这种内存压力只是暂时性的。

如果碰到“机器没有真的 OOM，但 vLLM 莫名其妙死了”的情况，这一点尤其值得检查。

---

8. 克隆双 DGX Spark 部署仓库

如果需要代理，我先设置 Shell：

export HTTP_PROXY="<PROXY_URL>"
export HTTPS_PROXY="<PROXY_URL>"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"

export NO_PROXY='localhost,127.0.0.1,192.168.31.0/24,192.168.100.0/24,192.168.101.0/24'
export no_proxy="$NO_PROXY"

然后：

cd /home/dgx

git clone --depth=1 \
  https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark.git

cd DeepSeek-v4-Flash-DSpark-2x-DGX-Spark

git rev-parse HEAD

我习惯记录一下 commit hash。这样以后仓库更新，至少知道这套环境到底是基于哪个版本部署出来的。我的原始部署也是按这个流程开始。

---

9. 配 Docker daemon 代理并准备 vLLM 镜像

Docker pull GHCR 镜像时，单纯设置当前 Shell 的 "HTTP_PROXY" 往往不够，所以我给 Docker daemon 单独配置代理：

sudo mkdir -p /etc/systemd/system/docker.service.d

sudo tee /etc/systemd/system/docker.service.d/90-deploy-proxy.conf >/dev/null <<'EOF'
[Service]
Environment="HTTP_PROXY=<PROXY_URL>"
Environment="HTTPS_PROXY=<PROXY_URL>"
Environment="NO_PROXY=localhost,127.0.0.1,::1,192.168.31.0/24,192.168.100.0/24,192.168.101.0/24"
EOF

sudo systemctl daemon-reload
sudo systemctl restart docker
sudo systemctl is-active docker

当前 recipe 使用：

IMG='ghcr.io/anemll/dspark-vllm-gx10:0.1.1'

下载：

docker pull "$IMG"

检查：

docker image inspect "$IMG" \
  --format 'ID={{.Id}} ARCH={{.Architecture}} SIZE={{.Size}}'

当前仓库 README 也将这个 Anemll 镜像作为默认 runtime，其中已经包含针对 GX10 / DGX Spark 的 vLLM 以及 DSpark、NVFP4 DS-MLA、b12x MoE 等支持。

先在 Head 验证 GPU 是否能进入容器：

docker run --rm \
  --gpus all \
  --entrypoint nvidia-smi \
  "$IMG"

如果这个命令都失败，就先修 Docker/NVIDIA Container Runtime，不要直接跑模型。

---

10. 不重复 pull：直接把 Docker 镜像同步到 Worker

因为两台机器镜像必须一致，而且我的外网速度并不总是理想，所以我最后直接把 Head 上已经下载好的镜像传给 Worker：

docker save "$IMG" |
ssh -o Compression=no \
  -c aes128-gcm@openssh.com \
  dgx@192.168.100.11 \
  'docker load'

然后比较 Image ID：

echo 'Head:'
docker image inspect "$IMG" --format '{{.Id}}'

echo 'Worker:'
ssh dgx@192.168.100.11 \
  "docker image inspect '$IMG' --format '{{.Id}}'"

最后单独在 Worker 上验证 GPU：

ssh dgx@192.168.100.11 \
  "docker run --rm --gpus all --entrypoint nvidia-smi '$IMG'"

双机部署前我非常建议做这个检查。

两边 Image ID 不一致，就先不要启动 TP=2。

---

11. 下载 DeepSeek-V4-Flash-0731

我的网络环境下 Hugging Face 下载不是最舒服，所以实际模型权重使用 ModelScope 下载。

先创建一个独立环境：

python3 -m venv /home/dgx/.venvs/modelscope
source /home/dgx/.venvs/modelscope/bin/activate

python -m pip install -U pip modelscope

本地模型目录：

MODEL_DIR='/home/dgx/.cache/huggingface/local/DeepSeek-V4-Flash-0731'

mkdir -p "$MODEL_DIR"

下载：

modelscope download \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local_dir "$MODEL_DIR" \
  --max-workers 1

我的最终本地路径因此是：

/home/dgx/.cache/huggingface/local/DeepSeek-V4-Flash-0731

当前 MiaAI-Lab recipe 也已经以 "DeepSeek-V4-Flash-0731" 为默认 agent-serving checkpoint。

---

12. 不要相信“下载完成”：检查所有 Safetensors 分片

几百 GB 级模型最怕的事情之一就是“命令看起来完成了，但少一个 shard”。

所以我额外写了一个完整性检查。

MODEL_DIR='/home/dgx/.cache/huggingface/local/DeepSeek-V4-Flash-0731'

python3 - "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index_path = root / "model.safetensors.index.json"
encoding_path = root / "encoding" / "encoding_dsv4.py"

if not index_path.exists():
    raise SystemExit(f"缺少 {index_path}")

index = json.loads(index_path.read_text())
shards = sorted(set(index["weight_map"].values()))
missing = [name for name in shards if not (root / name).is_file()]

print("权重分片数：", len(shards))
print("缺失分片数：", len(missing))
print("encoding_dsv4.py：", encoding_path.is_file())

if missing:
    print("\n".join(missing[:20]))
    raise SystemExit(1)
PY

这里不仅检查模型 shard，还检查：

encoding/encoding_dsv4.py

这不是多此一举。当前 0731 recipe 的 README 也专门提醒：两台机器都必须拥有完整模型 cache，并且 snapshot 中需要包含 "encoding/encoding_dsv4.py"；不完整 cache 或启动时再次在线下载，甚至可能把 Worker 磁盘打满，最终导致 TP=2 启动失败。

---

13. 把模型同步到 Worker

先创建目标目录：

ssh dgx@192.168.100.11 \
  'mkdir -p /home/dgx/.cache/huggingface/local/DeepSeek-V4-Flash-0731 && \
   chown -R dgx:dgx /home/dgx/.cache/huggingface'

然后：

rsync -aH --partial --info=progress2 \
  /home/dgx/.cache/huggingface/local/DeepSeek-V4-Flash-0731/ \
  dgx@192.168.100.11:/home/dgx/.cache/huggingface/local/DeepSeek-V4-Flash-0731/

同步以后，我建议把上一节的完整性检查脚本在 Worker 再跑一次。

不要只比较目录大小。

---

14. 最关键的 ".env.dspark"

我最初部署时使用的 ".env.dspark" 里有一项现在已经过时：

GPU_MEMORY_UTILIZATION=0.80

当前仓库已经不建议直接设置 "GPU_MEMORY_UTILIZATION"。启动脚本会根据是否启用 VL sidecar 自动选择两个 profile：

ENABLE_VL_SIDECAR=0
GPU_MEMORY_UTILIZATION_TEXT=0.835
GPU_MEMORY_UTILIZATION_VISION=0.80

其中默认的正式运行方式仍然是纯文本模式，也就是 "ENABLE_VL_SIDECAR=0"，实际使用 "GPU_MEMORY_UTILIZATION_TEXT=0.835"。只有实验性的 VL sidecar 开启时，才切到 "GPU_MEMORY_UTILIZATION_VISION=0.80"。

另外，仓库后续又增加了 "LONG_PREFILL_TOKEN_THRESHOLD=1024"，用于限制 chunked prefill 的单次 chunk 大小；默认 "MTP_NUM_TOKENS" 仍然是 5，"DEFAULT_THINKING" 现在推荐为 "max"，并继续使用普通 CUDA Graph，即 "VLLM_USE_BREAKABLE_CUDAGRAPH=0"。

所以我现在使用的核心参数更新为：

WORKER_HOST=dgx@192.168.100.11
WORKER_SCRIPT_DIR=/home/dgx/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark

MASTER_ADDR=192.168.100.10
MASTER_PORT=25000

NCCL_IB_HCA=rocep1s0f1
NCCL_SOCKET_IFNAME=enp1s0f1np1
TP_SOCKET_IFNAME=enp1s0f1np1
GLOO_SOCKET_IFNAME=enp1s0f1np1

WORKER_NCCL_IB_HCA=rocep1s0f1
WORKER_NCCL_SOCKET_IFNAME=enp1s0f1np1
WORKER_TP_SOCKET_IFNAME=enp1s0f1np1
WORKER_GLOO_SOCKET_IFNAME=enp1s0f1np1

NCCL_IB_GID_AUTO=1

VLLM_HOST_IP=192.168.100.10
WORKER_VLLM_HOST_IP=192.168.100.11

HF_CACHE=/home/dgx/.cache/huggingface
WORKER_HF_CACHE=/home/dgx/.cache/huggingface

HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HUB_DISABLE_XET=1

ABLITERATED=0

DSPARK_MODEL_OFFICIAL=/cache/huggingface/local/DeepSeek-V4-Flash-0731
DSPARK_REVISION=
DSPARK_ENCODING_FILE=/cache/huggingface/local/DeepSeek-V4-Flash-0731/encoding/encoding_dsv4.py

SERVED_MODEL_NAME=deepseek-v4-flash-0731

VLLM_HOST=0.0.0.0
VLLM_PORT=8888

DSPARK_VLLM_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1

MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=6
MAX_NUM_BATCHED_TOKENS=8192
LONG_PREFILL_TOKEN_THRESHOLD=1024

GPU_MEMORY_UTILIZATION_TEXT=0.835
GPU_MEMORY_UTILIZATION_VISION=0.80

MTP_NUM_TOKENS=5
DEFAULT_THINKING=max

ENABLE_VL_SIDECAR=0
PREPARE_VL_SIDECAR_MODEL=0

VLLM_USE_FLASHINFER_SAMPLER=1
VLLM_USE_BREAKABLE_CUDAGRAPH=0
VLLM_USE_B12X_MOE=1

这里还有一个仓库更新后很重要的变化：现在不应该再直接写 "DSPARK_MODEL="。启动脚本根据 "ABLITERATED" 自动选择 "DSPARK_MODEL_OFFICIAL" 或 "DSPARK_MODEL_ABLITERATED"。官方仓库默认使用 Hugging Face 模型 ID，并固定到测试过的 revision；如果把 "DSPARK_REVISION=" 显式留空，则表示不传 "--revision"。

我的模型不是按照标准 Hugging Face Hub snapshot 结构保存，而是之前通过 ModelScope 放在：

/home/dgx/.cache/huggingface/local/DeepSeek-V4-Flash-0731

Compose 会把：

/home/dgx/.cache/huggingface

挂载成容器里的：

/cache/huggingface

所以我仍然需要使用容器路径：

DSPARK_MODEL_OFFICIAL=/cache/huggingface/local/DeepSeek-V4-Flash-0731
DSPARK_REVISION=
DSPARK_ENCODING_FILE=/cache/huggingface/local/DeepSeek-V4-Flash-0731/encoding/encoding_dsv4.py

当前 Compose 在启动时还会把这个 "encoding_dsv4.py" 安装到 vLLM，并自动应用 Issue #21 的 encoder hotfix。

原文里 "GPU_MEMORY_UTILIZATION=0.80" 以及直接设置 "DSPARK_MODEL=/cache/..." 的写法，因此都应该删除。

后续仓库更新：先备份，再同步 "origin/main"

这套仓库后面更新得很快，所以我后来又完整走了一次更新流程。

这里最重要的是：".env.dspark" 本身已经在 ".gitignore" 中，所以创建 Git 备份分支并不能备份 ".env.dspark"，必须另外复制一份。

先停掉旧服务。如果服务本来没有运行，可以跳过：

cd ~/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark

./status-deepseek-v4-flash-dspark.sh
./stop-deepseek-v4-flash-dspark.sh

然后单独备份环境变量：

cd ~/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark

STAMP="$(date +%Y%m%d-%H%M%S)"
ENV_BACKUP=".env.dspark.bak.${STAMP}"

cp -a .env.dspark "$ENV_BACKUP"

echo "环境变量备份：$ENV_BACKUP"
ls -lh "$ENV_BACKUP"

接下来检查本地和远端到底分叉了多少：

git fetch origin main

echo "===== 分叉数量：左=本地独有，右=远端独有 ====="
git rev-list --left-right --count main...origin/main

echo "===== 分叉提交 ====="
git log --left-right --graph --decorate --oneline \
  --max-count=30 main...origin/main

我的本地仓库还有一些额外修改，所以在强制同步之前，我还保留一个 Git 分支：

BACKUP_BRANCH="backup-before-sync-${STAMP}"

git branch "$BACKUP_BRANCH" HEAD

echo "备份分支：$BACKUP_BRANCH"
git rev-parse "$BACKUP_BRANCH"

工作区里如果还有 tracked 或普通 untracked 文件，也先 stash。这里使用 "-u"，但它仍然不会代替前面对 ".env.dspark" 的单独备份：

STASH_NAME="before-sync-${STAMP}"

if [ -n "$(git status --porcelain)" ]; then
  git stash push -u -m "$STASH_NAME"
fi

git stash list

之后切回 "main"，直接对齐远端：

git switch main
git reset --hard origin/main

git status
git log -1 --oneline --decorate

echo "===== HEAD / origin/main ====="
git rev-parse HEAD
git rev-parse origin/main

如果前面确实创建了 stash，再把本地文件恢复回来：

STASH_REF="$(
  git stash list --format='%gd %s' |
  awk -v name="$STASH_NAME" 'index($0, name) {print $1; exit}'
)"

if [ -n "$STASH_REF" ]; then
  git stash apply "$STASH_REF"
fi

git status --short

ls -lh "$ENV_BACKUP"
ls -lh results/my-benchmark.json 2>/dev/null

这里我不会立刻无脑 "drop"。先确认没有冲突、本地 benchmark 等文件也都恢复正确，再执行：

if [ -n "$STASH_REF" ]; then
  git stash drop "$STASH_REF"
fi

截至我这次重新同步时，仓库在 8 月 11～12 日连续加入了 text-only 默认 profile、VL 开关、revision pin、Issue #22 长上下文修复、vLLM 0.27 性能 hotfix backport，以及 Issue #27 的 partial-prefill 修复，所以旧 ".env.dspark" 确实不能再完全照搬。

更新旧 ".env.dspark"

因为我的网络、Worker 地址、本地模型路径这些配置都已经调通，所以没有直接覆盖整个 ".env.dspark"，而是在备份以后更新仓库新版本需要的参数。

我现在用下面这段一次处理：

cd ~/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark

python3 - <<'PY'
from pathlib import Path
import re

p = Path(".env.dspark")
s = p.read_text()

# 新版本已经不应该由用户直接设置的旧参数。
for key in (
    "GPU_MEMORY_UTILIZATION",
    "DSPARK_MODEL",
):
    s = re.sub(
        rf"^{re.escape(key)}=.*\n?",
        "",
        s,
        flags=re.M,
    )

values = {
    # 当前 1M agent-serving profile
    "MAX_MODEL_LEN": "1048576",
    "MAX_NUM_SEQS": "6",
    "MAX_NUM_BATCHED_TOKENS": "8192",
    "LONG_PREFILL_TOKEN_THRESHOLD": "1024",

    # GPU profile
    "GPU_MEMORY_UTILIZATION_TEXT": "0.835",
    "GPU_MEMORY_UTILIZATION_VISION": "0.80",

    # DSpark
    "MTP_NUM_TOKENS": "5",
    "DEFAULT_THINKING": "max",

    # 当前正式默认仍然是 text-only
    "ENABLE_VL_SIDECAR": "0",
    "PREPARE_VL_SIDECAR_MODEL": "0",

    # Anemll 0.1.1 当前使用的 runtime knobs
    "VLLM_USE_FLASHINFER_SAMPLER": "1",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
    "VLLM_USE_B12X_MOE": "1",

    # 使用官方 0731，但模型来自我自己的本地目录
    "ABLITERATED": "0",
    "DSPARK_MODEL_OFFICIAL": "/cache/huggingface/local/DeepSeek-V4-Flash-0731",
    "DSPARK_REVISION": "",
    "DSPARK_ENCODING_FILE": "/cache/huggingface/local/DeepSeek-V4-Flash-0731/encoding/encoding_dsv4.py",
}

for k, v in values.items():
    pat = re.compile(rf"^{re.escape(k)}=.*$", re.M)
    line = f"{k}={v}"

    if pat.search(s):
        s = pat.sub(line, s)
    else:
        s = s.rstrip("\n") + "\n" + line + "\n"

p.write_text(s)
PY

这也替代了我之前分别使用几次 "sed" 修改 "GPU_MEMORY_UTILIZATION_*"、"DSPARK_MODEL_OFFICIAL"、"DSPARK_REVISION" 和 "DSPARK_ENCODING_FILE" 的做法。

当前仓库的 ".env.dspark.example" 还把 Anemll "0.1.1" 镜像进一步固定到了 manifest digest，目的是防止同一个 tag 将来被重新发布后内容发生变化。

不过我这里前面采用的是：

docker save "$IMG" | ssh ... 'docker load'

把 Head 的镜像直接复制到 Worker，并通过 Image ID 保证两边完全一致。因此已经部署好的环境继续使用：

DSPARK_VLLM_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1

也是可以工作的。

如果改成仓库 example 里的 "@sha256:..." 写法，需要先确认 Head 和 Worker 都能用这个完整 digest 引用执行 "docker image inspect"，否则新版启动脚本会在启动前直接停止。启动脚本现在会主动检查两台机器上配置的 image 是否存在。

更新以后先检查，不要直接启动

先确认旧变量真的已经清掉：

grep -E \
'^(ABLITERATED|DSPARK_MODEL_OFFICIAL|DSPARK_REVISION|DSPARK_ENCODING_FILE|MAX_MODEL_LEN|MAX_NUM_SEQS|MAX_NUM_BATCHED_TOKENS|LONG_PREFILL_TOKEN_THRESHOLD|GPU_MEMORY_UTILIZATION_TEXT|GPU_MEMORY_UTILIZATION_VISION|MTP_NUM_TOKENS|DEFAULT_THINKING|ENABLE_VL_SIDECAR|PREPARE_VL_SIDECAR_MODEL|VLLM_USE_FLASHINFER_SAMPLER|VLLM_USE_BREAKABLE_CUDAGRAPH|VLLM_USE_B12X_MOE)=' \
.env.dspark

echo "===== 下面两项应该没有输出 ====="
grep '^GPU_MEMORY_UTILIZATION=' .env.dspark || true
grep '^DSPARK_MODEL=' .env.dspark || true

再检查 Host 上的本地模型和 encoder：

test -d \
  /home/dgx/.cache/huggingface/local/DeepSeek-V4-Flash-0731 \
  && echo "Head model OK"

test -f \
  /home/dgx/.cache/huggingface/local/DeepSeek-V4-Flash-0731/encoding/encoding_dsv4.py \
  && echo "Head encoder OK"

Worker 也检查一次：

ssh dgx@192.168.100.11 '
test -d /home/dgx/.cache/huggingface/local/DeepSeek-V4-Flash-0731 &&
echo "Worker model OK"

test -f /home/dgx/.cache/huggingface/local/DeepSeek-V4-Flash-0731/encoding/encoding_dsv4.py &&
echo "Worker encoder OK"
'

现在仓库还提供了专门的配置检查脚本：

./validate-dspark-config.sh

它会按照实际启动脚本的逻辑解析 "ENABLE_VL_SIDECAR"、模型选择和 GPU utilization，并把最终渲染出来的 vLLM 参数打印出来。

我重点确认：

serve mode: text
checkpoint: /cache/huggingface/local/DeepSeek-V4-Flash-0731
revision: (default branch tip / unpinned)

max model len: 1048576
max num seqs: 6
max batched tokens: 8192
gpu memory utilization: 0.835
spec tokens: 5
breakable cudagraph: 0

没有问题以后再启动。

15. 启动 DeepSeek V4 Flash

更新后的正常启动仍然是一条：

cd ~/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark

./start-deepseek-v4-flash-dspark.sh

但现在这条脚本背后做的事情已经比我最初部署时多了很多。

它会自动根据 Head 和 Worker 的 RoCE 地址解析各自的 RoCEv2 GID index；检查两边 Docker image；把最新的 Compose、".env.dspark"、DSpark proposer 和相关 hotfix 文件同步到 Worker；分别验证 Head 和 Worker 的 Compose；然后按照 Worker first 的顺序启动双机服务。

新版本还会自动处理 Issue #22 的 "nvfp4_ds_mla" 长上下文 decode 修复，以及一组从 vLLM 0.27 backport 回当前 Anemll runtime 的 DeepSeek V4 性能 hotfix。补丁应用完成后，两边容器会统一重启一次，然后脚本等待 API ready，并自动执行一个最小 OpenAI-compatible chat 请求。

所以这些补丁不需要我再手工进入 Docker 修改。

启动完成后先看：

curl -fsS http://127.0.0.1:8888/v1/models |
python3 -m json.tool

确认服务状态：

./status-deepseek-v4-flash-dspark.sh

再跑一次仓库自己的完整 smoke test：

./smoke-deepseek-v4-flash-dspark.sh

如果有问题：

./logs-deepseek-v4-flash-dspark.sh

当前默认 text-only profile 下，仓库记录的配置是：

MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=6
MAX_NUM_BATCHED_TOKENS=8192
LONG_PREFILL_TOKEN_THRESHOLD=1024
GPU_MEMORY_UTILIZATION_TEXT=0.835
MTP_NUM_TOKENS=5
DEFAULT_THINKING=max
ENABLE_VL_SIDECAR=0
VLLM_USE_BREAKABLE_CUDAGRAPH=0

仓库在这套配置下记录的启动结果大约是 18 GiB 可用 KV Cache、约 2.5M token KV 容量，以及对单个 1,048,576-token request 约 2.4x 的理论最大并发；这些数字只应该作为启动 sanity check，实际值仍然以自己的 boot log 为准。

---

15. 启动 DeepSeek V4 Flash

所有东西准备好以后：

cd ~/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark

./start-deepseek-v4-flash-dspark.sh

当前仓库的启动逻辑是 Worker first：先准备和启动 Worker，再启动 Head，这样可以减少多节点初始化阶段的竞争问题。启动脚本还会检查双方的 Compose 配置和运行环境。

启动完成后：

curl -fsS http://127.0.0.1:8888/v1/models |
python3 -m json.tool

查看服务状态：

./status-deepseek-v4-flash-dspark.sh

看日志：

./logs-deepseek-v4-flash-dspark.sh

跑仓库自带 Smoke Test：

./smoke-deepseek-v4-flash-dspark.sh

这些也是我最后的启动和验证流程。

---

16. 再实际调用一次 OpenAI-compatible API

"/v1/models" 成功只能证明 API 活了，我一般还会真正生成一次。

例如：

curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash-0731",
    "messages": [
      {
        "role": "user",
        "content": "用一句话介绍 NVIDIA DGX Spark。"
      }
    ],
    "max_tokens": 256
  }'

如果要让局域网其他设备访问，因为这里配置的是：

VLLM_HOST=0.0.0.0
VLLM_PORT=8888

因此入口就是：

http://<HEAD_IP>:8888/v1

当前仓库同样默认使用 "8888"，并提醒如果绑定 "0.0.0.0"，应该自己通过防火墙或网络层控制 API 暴露范围；如果只在 Head 本机调用，则可以改成 "127.0.0.1"。

不要直接把一个没有鉴权的 "0.0.0.0:8888" 暴露到公网。

---

17. 我认为最值得记住的几个坑

第五点现在需要改成：

第五是第一次成功启动时先别急着调参，而且不要再照旧教程手工设置 "GPU_MEMORY_UTILIZATION=0.80"。我最初记录的这组 baseline 已经随着仓库更新发生变化。

当前默认的 1M text-only agent-serving profile 是：

MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=6
MAX_NUM_BATCHED_TOKENS=8192
LONG_PREFILL_TOKEN_THRESHOLD=1024
GPU_MEMORY_UTILIZATION_TEXT=0.835
MTP_NUM_TOKENS=5
DEFAULT_THINKING=max
ENABLE_VL_SIDECAR=0
VLLM_USE_BREAKABLE_CUDAGRAPH=0

"GPU_MEMORY_UTILIZATION" 现在由启动脚本根据 profile 自动生成。纯文本模式使用 "0.835"；实验性的 vision coexist 才使用 "0.80"。先把当前仓库这一组 baseline 跑通，再去调整 concurrency、MTP、context 和 memory utilization。

18. 关于性能：这里我暂时不编一个数字

原来 benchmark 条件里最后一项：

GPU_MEMORY_UTILIZATION

现在更准确地应该记录成：

Prompt tokens
Output tokens
Concurrency
TTFT / TTFC
Decode tok/s
Aggregate tok/s
DSpark acceptance rate
MAX_MODEL_LEN
MAX_NUM_SEQS
MAX_NUM_BATCHED_TOKENS
LONG_PREFILL_TOKEN_THRESHOLD
MTP_NUM_TOKENS
GPU_MEMORY_UTILIZATION_TEXT
实际生效的 GPU_MEMORY_UTILIZATION
VLLM_USE_BREAKABLE_CUDAGRAPH

因为现在 ".env.dspark" 里的 text / vision utilization 和启动时真正传给 vLLM 的 "GPU_MEMORY_UTILIZATION" 已经是两层配置。只记录一个 "GPU_MEMORY_UTILIZATION"，以后回头看 benchmark 很容易不知道当时到底运行的是哪个 profile。

---

18. 关于性能：这里我暂时不编一个数字

原始部署记录里我留下了服务启动、模型加载和 smoke test 的步骤，但没有留下足够完整、环境固定的 benchmark 结果。

所以这里我不打算拿一次随手测试的 tok/s 当“2× DGX Spark 的性能”。

当前仓库已经提供：

scripts/benchmark-0731.py
results/deepseek-v4-flash-0731-2x-dgx-spark.json

并且 README 明确说明性能结果会随着 prompt 长度、生成长度、batch、context、KV Cache 路径以及 speculative decoding 参数发生变化。

后面如果认真测试，我更倾向固定下面这些条件以后再记录：

Prompt tokens
Output tokens
Concurrency
TTFT / TTFC
Decode tok/s
Aggregate tok/s
DSpark acceptance rate
MAX_MODEL_LEN
MTP_NUM_TOKENS
GPU_MEMORY_UTILIZATION

这样得到的数据才比较值得横向比较。

---

结语

最终跑通以后，这套东西反而显得很简单：

2 × DGX Spark
        ↓
ConnectX-7 / RoCE
        ↓
SSH 免密
        ↓
相同 Docker Runtime
        ↓
两边完整的 DeepSeek-V4-Flash-0731 权重
        ↓
vLLM TP=2
        ↓
DSpark speculative decoding
        ↓
OpenAI-compatible API :8888

真正花时间的是把底层每一层都变得“无聊”——网络稳定、驱动正常、Docker 正常、两边镜像一致、两边模型一致、路径一致。

这次部署给我的最大经验也是这个：

多机大模型部署里，模型本身往往不是最难的；真正难的是让两台机器在网络、驱动、容器、文件和配置上都认为自己处在同一个世界里。

等这些都对齐以后：

./start-deepseek-v4-flash-dspark.sh
