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

部署过程中我前面配过一次 ".env.dspark"，后来发现有些路径和参数不够干净，于是最终重新从 example 生成了一次。

先备份：

cd ~/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark

[ -f .env.dspark ] && \
cp .env.dspark ".env.dspark.bak.$(date +%Y%m%d-%H%M%S)"

cp .env.dspark.example .env.dspark

我的最终关键参数是：

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

DSPARK_MODEL=/cache/huggingface/local/DeepSeek-V4-Flash-0731
DSPARK_ENCODING_FILE=/cache/huggingface/local/DeepSeek-V4-Flash-0731/encoding/encoding_dsv4.py
SERVED_MODEL_NAME=deepseek-v4-flash-0731

VLLM_HOST=0.0.0.0
VLLM_PORT=8888

DSPARK_VLLM_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1

MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=6
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.80
MTP_NUM_TOKENS=5

NCCL_DEBUG=INFO

这组参数就是我最终部署记录里的核心配置。

而且截至本文整理时，仓库最新 README 推荐的 agent-serving profile 已经和这组设置高度一致：

MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=6
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.80
MTP_NUM_TOKENS=5
DSPARK_VLLM_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1

Host 路径和 Container 路径不要搞混

这是我认为最容易出错的一个点。

Host 上模型实际存在于：

/home/dgx/.cache/huggingface/local/DeepSeek-V4-Flash-0731

但是容器里挂载的是：

/cache/huggingface

所以：

DSPARK_MODEL

应该写成容器里面看到的：

/cache/huggingface/local/DeepSeek-V4-Flash-0731

而不是 Host 的 "/home/dgx/..."。

同理：

DSPARK_ENCODING_FILE=/cache/huggingface/local/DeepSeek-V4-Flash-0731/encoding/encoding_dsv4.py

路径混用时，最常见的现象就是“宿主机明明有模型，但 vLLM 说找不到”。

关于 "NCCL_IB_GID_AUTO"

我自己的环境最终使用了：

NCCL_IB_GID_AUTO=1

但这里不建议把它理解为适合所有版本的永久配置。

当前仓库 README 明确提醒，RoCE 的 GID 配置取决于实际环境，"NCCL_IB_GID_INDEX" 并不一定永远是 0，需要匹配实际 RoCE GID。

所以如果遇到 NCCL 初始化卡死、RoCE 建链失败，接口 IP 明明互通却无法 TP=2，GID 是重点排查对象之一。

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

整套过程真正麻烦的地方，其实不是最后那一条 "start" 命令。

第一是 两台机器的底层环境必须先完全一致、完全健康。驱动、Docker、镜像、模型文件，只要其中一边有细微差异，TP=2 出问题以后排查会非常痛苦。

第二是 DGX Spark 的 ConnectX-7 接口命名比普通服务器更容易看错。尤其 "enP2p1s0f1np1" 里面的大写 "P" 是真的，不是 typo。最好永远先用：

ibdev2netdev
ip -br link

确认映射。NVIDIA 官方文档现在也专门花了很长一节解释 DGX Spark 的 QSFP → PCIe → Ethernet → RoCE 对应关系。

第三是 模型下载成功不等于模型完整。检查 "model.safetensors.index.json"、逐个确认 shard、确认 "encoding_dsv4.py" 都比看目录大小可靠得多。

第四是 Host path 和 Container path 一定要分清。我的模型在 Host 是：

/home/dgx/.cache/huggingface/...

但配置给 vLLM 的 "DSPARK_MODEL" 是：

/cache/huggingface/...

第五是 第一次成功启动时先别急着调参。"MAX_MODEL_LEN=1048576"、"MAX_NUM_SEQS=6"、"MAX_NUM_BATCHED_TOKENS=8192"、"GPU_MEMORY_UTILIZATION=0.80"、"MTP_NUM_TOKENS=5" 是当前仓库已经验证过的一组相对保守的 1M agent-serving 参数。先把这个 baseline 跑通，再调整 concurrency、MTP 和 memory utilization。

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
