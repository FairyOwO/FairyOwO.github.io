> 历史文章搬运

需求之初是想对年抛机, 月抛机进行统一的管理, 方便部署相关镜像, 类似于史莱姆的结构(拿到新的机器, 加入集群, 机器时间过期, 自动离线, 伸缩重启分配全由集群本身管理)

使用系统为 Debian

## 服务器搭建

### 搭建集群

主 server sh脚本
<details><summary>Details</summary>
<p>

```sh
deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware
deb-src https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware

deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware
deb-src https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware

deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-backports main contrib non-free non-free-firmware
deb-src https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-backports main contrib non-free non-free-firmware

# 以下安全更新软件源包含了官方源与镜像站配置，如有需要可自行修改注释切换
deb https://security.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware
deb-src https://security.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware" > /etc/apt/sources.list

apt update

curl -sfL https://rancher-mirror.rancher.cn/k3s/k3s-install.sh | \
  INSTALL_K3S_MIRROR=cn \
  sh -s - server \
  --cluster-init \
  --system-default-registry=registry.cn-hangzhou.aliyuncs.com

cat /var/lib/rancher/k3s/server/token

cat >> /etc/rancher/k3s/registries.yaml << EOF
mirrors:
  docker.io:
    endpoint:
      - "https://dockerproxy.net"
      - "https://registry.cn-hangzhou.aliyuncs.com/"
      - "https://mirror.ccs.tencentyun.com"
  k8s.gcr.io:
    endpoint:
      - "https://k8s.dockerproxy.net"
      - "https://registry.aliyuncs.com/google_containers"
  ghcr.io:
    endpoint:
      - "https://ghcr.dockerproxy.net"
      - "https://ghcr.m.daocloud.io/"
  gcr.io:
    endpoint:
      - "https://gcr.dockerproxy.net"
      - "https://gcr.m.daocloud.io/"
  quay.io:
    endpoint:
      - "https://quay.dockerproxy.net"
      - "https://quay.tencentcloudcr.com/"
  registry.k8s.io:
    endpoint:
      - "https://k8s.dockerproxy.net"
      - "https://registry.aliyuncs.com/v2/google_containers"
EOF
systemctl restart k3s
```

</p>
</details> 

副 server sh脚本

<details><summary>Details</summary>
<p>

```sh
deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware
deb-src https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware

deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware
deb-src https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware

deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-backports main contrib non-free non-free-firmware
deb-src https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-backports main contrib non-free non-free-firmware

# 以下安全更新软件源包含了官方源与镜像站配置，如有需要可自行修改注释切换
deb https://security.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware
deb-src https://security.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware" > /etc/apt/sources.list

apt update

curl -sfL https://rancher-mirror.rancher.cn/k3s/k3s-install.sh | \
  INSTALL_K3S_MIRROR=cn \
  sh -s - server \
  --cluster-init \
  --system-default-registry=registry.cn-hangzhou.aliyuncs.com

cat /var/lib/rancher/k3s/server/token

cat >> /etc/rancher/k3s/registries.yaml << EOF
mirrors:
  docker.io:
    endpoint:
      - "https://dockerproxy.net"
      - "https://registry.cn-hangzhou.aliyuncs.com/"
      - "https://mirror.ccs.tencentyun.com"
  k8s.gcr.io:
    endpoint:
      - "https://k8s.dockerproxy.net"
      - "https://registry.aliyuncs.com/google_containers"
  ghcr.io:
    endpoint:
      - "https://ghcr.dockerproxy.net"
      - "https://ghcr.m.daocloud.io/"
  gcr.io:
    endpoint:
      - "https://gcr.dockerproxy.net"
      - "https://gcr.m.daocloud.io/"
  quay.io:
    endpoint:
      - "https://quay.dockerproxy.net"
      - "https://quay.tencentcloudcr.com/"
  registry.k8s.io:
    endpoint:
      - "https://k8s.dockerproxy.net"
      - "https://registry.aliyuncs.com/v2/google_containers"
EOF
systemctl restart k3s
```

</p>
</details> 

client sh脚本

<details><summary>Details</summary>
<p>

```sh
deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware
deb-src https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware

deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware
deb-src https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware

deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-backports main contrib non-free non-free-firmware
deb-src https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-backports main contrib non-free non-free-firmware

# 以下安全更新软件源包含了官方源与镜像站配置，如有需要可自行修改注释切换
deb https://security.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware
deb-src https://security.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware" > /etc/apt/sources.list

apt update

curl -sfL https://rancher-mirror.rancher.cn/k3s/k3s-install.sh | \
  INSTALL_K3S_MIRROR=cn \
  K3S_URL=https://ip:6443 \
  K3S_TOKEN=your_token \
  sh -

mkdir -p /etc/rancher/k3s
cat >> /etc/rancher/k3s/registries.yaml << EOF
mirrors:
  docker.io:
    endpoint:
      - "https://dockerproxy.net"
      - "https://registry.cn-hangzhou.aliyuncs.com/"
      - "https://mirror.ccs.tencentyun.com"
  k8s.gcr.io:
    endpoint:
      - "https://k8s.dockerproxy.net"
      - "https://registry.aliyuncs.com/google_containers"
  ghcr.io:
    endpoint:
      - "https://ghcr.dockerproxy.net"
      - "https://ghcr.m.daocloud.io/"
  gcr.io:
    endpoint:
      - "https://gcr.dockerproxy.net"
      - "https://gcr.m.daocloud.io/"
  quay.io:
    endpoint:
      - "https://quay.dockerproxy.net"
      - "https://quay.tencentcloudcr.com/"
  registry.k8s.io:
    endpoint:
      - "https://k8s.dockerproxy.net"
      - "https://registry.aliyuncs.com/v2/google_containers"
EOF
systemctl restart k3s-agent
```

</p>
</details> 

> 注: k3s 搭建集群的方案需要保证主服务器不离线, 否则整个集群会离线, 考虑到k3s占用低, 机器一般是性能不高的类型, 我也有长期续费的服务器, 故使用这个方案

在主server服务器使用

```sh
kubectl get nodes -A
```
出现每台机子的信息, 代表集群内部网络通信没问题

在主server服务器使用
```sh
kubectl get pods --all-namespaces
```
在所有服务在 `RUNNING` 状态时, 为安装成功 (这些服务都是内部通信与均衡负载的镜像), 如果是卡在 `container creating`, 则安装失败, 原因是镜像没正确配置

### 安装helm (虽然不知道干什么用, 集群内也自带一个helm)

1. 手动安装
    1. 下载需要的版本 [下载地址](https://github.com/helm/helm/releases)
    2. 解压, 上传到服务器, chmod给执行权限
    3. 移动到环境变量的目录中
        ```sh
        mv helm /usr/local/bin/helm
        ```
2. 使用脚本安装
    ```sh
    https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
    ```

## 面板安装

为了简单, 面板选择的是 kubepi

[文档](https://github.com/1Panel-dev/KubePi/wiki/2%E3%80%81%E5%AE%89%E8%A3%85%E9%83%A8%E7%BD%B2#kubernetes)

这里选择的是非持久化部署, 在直接部署在刚刚建好的集群之中

> 持久化部署会有莫名其妙的分配问题, 应该是跟分配本地空间有关系, 我也不需要持久化集群信息(因为只有一个集群), 所以没什么关系

```sh
# 安装
sudo kubectl apply -f https://raw.githubusercontent.com/1Panel-dev/KubePi/master/docs/deploy/kubectl/kubepi.yaml
```

安装完成后, 根据安装教程, 获取访问地址

```sh
# 获取 NodeIp
export NODE_IP=$(kubectl get nodes -o jsonpath="{.items[0].status.addresses[0].address}")
# 获取 NodePort
export NODE_PORT=$(kubectl -n kube-system get services kubepi -o jsonpath="{.spec.ports[0].nodePort}")
# 获取 Address
echo http://$NODE_IP:$NODE_PORT
```

> 注: 内网组机子的时候这里会是内网地址, 需要使用端口转发转发到 `0.0.0.0` 之后才能外网访问
> ```sh
> kubectl port-forward --address 0.0.0.0 kubepi-d8477f9d8-drthz -n kube-system 2999:80
> ```
> 此命令不会中断, 会持续运行

登陆系统

```text
地址: http://$NODE_IP:$NODE_PORT
用户名: admin
密码: kubepi
```

登陆后记得修改密码

导入集群

在主服务器, 获取

```sh
cd /etc/rancher/k3s
cat k3s.yaml
```
在 kubepi 导入集群, 认证模式选择 kubeconfig文件, 把这个文件复制进去

在集群配置中, 配置一下网络, 使之可以直接通过外网端口访问

具体配置流程忘了, 此方法由同事指点

## 部署项目

待研究更新