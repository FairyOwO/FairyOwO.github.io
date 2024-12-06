> 可能并非最佳实现, 没有进行系统性学习, 欢迎交流

## 镜像

### 镜像管理

安装镜像管理平台 Harbor, 来为集群提供镜像源

在[上一篇](https://fairyowo.github.io/post/shi-yong-%20k3s%20-da-jian-%20k8s%20-ji-qun-%28-shi-yong-guo-nei-jing-xiang-%29.html), 介绍了集群的搭建, 在那个时候配置了集群的镜像源, 也就是说, 这个集群有简单的拉镜像能力, 其次, 安装了 helm, 可以根据 helm 仓库拉取应用

这里介绍拉取原始chart仓库的相关文件的方法安装

1. 获取chart仓库的文件
  前往 [harbor-helm](harbor-helm) 的 github 仓库
  前往 release, 下载源代码, 解压之后拿到其中的 `templates` 文件夹, `Chart.yaml` 文件, `values.yaml` 文件
2. 修改 `values.yaml` 文件
  详见 附录一
  > 我们的小集群既没有https, 又没有持久化存储, 所以这些都不需要写在配置文件中, 直接关掉这些功能即可
  > admin 的初始化密码在网页登陆后也能改
3. 创建 harbor 专属的 namespace
  ```sh
  kubectl create namespace harbor
  ```
4. 启动
  ```sh
  cd /path/to/harbor
  helm --namespace harbor install harbor .
  ```
  如果对配置项后悔, 使用 以下命令更新
  ```sh
  helm --namespace harbor upgrade harbor .
  ```

到 kebupi 中, 即可看到 harbor 的相关镜像在拉取了, 如果镜像配置正确的话, 过一段时间就会拉取成功
> 在 kubepi 中他会不断重试拉取, 实际上会缓慢拉取成功
> 我是2M小服务器拉了半个小时+

在拉取成功之后, 第一时间在刚刚 `values.yaml` 中配置的url中登录, 修改admin密码(如果没有修改默认密码)

### 向 Harbor 上传镜像

需要一台使用 docker 的机器, 向 Harbor 上传镜像 `docker pull`
> 这里我在安装好后才意识到, 我的集群使用的容器运行时 containerd, 好像没有能力对镜像进行修改, 只能对镜像进行部署, 所以令起了一台机子, 安装 docker, 配置镜像源(TODO)
> 另一种方案是重装集群, 使用 docker 作为 容器运行时, 但我没有选择这个方案

做好准备后 需要对 harbor 进行信任
> 没有 https 导致的, 有 https 可以跳过这一步

向 `/etc/docker/daemon.json` 写入
```text
"insecure-registries": ["harbor_ip:port"]
```

> 你需要自行处理他的json语法

后, 使用

```sh
systemctl daemon-reload
systemctl restart docker
```

重启 docker

在配置好后, 可以推一个简单的 helloworld镜像

```sh
docker login harbor_ip:port
# 这里填入你的账号与密码. 我这里是 admin 与相对于的密码

docker run hello-world:latest
# 修改tag
docker tag hello-world:latest harbor_ip:port/library/hello-world:latest
docker push harbor_ip:port/library/hello-world:latest
```

登录到 harbor 控制台, 即可看到刚刚推送上来的镜像

## 集群拉取镜像

> 如果你没有 https, 则需要以下额外一步, 如果有, 则保证集群可以连接到 harbor 即可

根据不同的集群搭建方法(这里是k3s), 将 harbor 添加进集群可以拉取的镜像源

与普通添加镜像一致, 首先需要到 `/etc/rancher/k3s` 目录下

向其中的 `registries.yaml` 添加内容

```yaml
mirrors:
  harbor_ip:port:
    endpoint:
     - http://harbor_ip:port

configs:
  harbor_ip:port:
    auth:
      username: 你的 harbor 账号
      password: 你的 harbor 密码
```

> 这里可能 `registries.yaml` 已经有内容了(配置过镜像源), 你需要根据 yaml 的语法自行处理他们的关系
> 每一台集群都要这么做

之后, 重启即可
```sh
systemctl restart k3s  # systemctl restart k3s-agent if agent
```

在编写好 kubectl 使用的 yaml 后, 即可从 harbor 拉取镜像
> 感谢 claude-sonnet 帮我写yaml

## 附录

### 一
这里给出常用的 harbor 的 values.yaml 的选项, 复制自 [(https://blog.starry-s.moe/posts/2023/harbor-helm-chart/)](https://blog.starry-s.moe/posts/2023/harbor-helm-chart/)

<details><summary>修改的选项</summary>
<p>

```yaml
expose:
# expose type, 可以设置为 ingress, clusterIP, nodePort, nodeBalancer，区分大小写
# 默认为 ingress（如果不想使用 80/443 标准端口，可以设置为 nodePort，端口为高位 3000X）
type: ingress
tls:
  # 是否启用 TLS (HTTPS)，建议启用
  enabled: true
  # TLS Certificate 的来源，可以为 auto, secret 或 none
  # 如果为 secret，需要在安装 Chart 之前先创建 TLS Secret
  # 1) auto: generate the tls certificate automatically
  # 2) secret: read the tls certificate from the specified secret.
  # The tls certificate can be generated manually or by cert manager
  # 3) none: configure no tls certificate for the ingress. If the default
  # tls certificate is configured in the ingress controller, choose this option
  certSource: secret
  secret:
    # The name of secret which contains keys named:
    # "tls.crt" - the certificate
    # "tls.key" - the private key
    secretName: "harbor-tls"
    # Only needed when the "expose.type" is "ingress".
    notarySecretName: "harbor-tls"
ingress:
  hosts:
    # Ingress Host，如果需要允许任意域名/IP 都能访问，将其设置为空字符串（不建议）
    # 这里填写的域名务必能解析到当前集群
    core: harbor.example.com
    notary: notary.example.com

# Harbor external URL
# 与 Ingress Host 相对应，如果启用了 TLS，那就是 https://<domain>
# 如果没启用 TLS，那就是 http://<domain>
# 如果 expose type 为 nodePort，则填写 http(s)://<IP_ADDRESS>:3000X (端口号不能丢)
externalURL: https://harbor.example.com

# 持久卷配置，默认为 true，如果是测试环境可以设置为 enabled: false (重新安装 Chart 时仓库里所有的数据都会丢失，不建议！)
# 如果需要启用持久卷，可以在安装 Chart 之前提前创建好 PVC，并配置 subPath
persistence:
enabled: true
resourcePolicy: "keep"
persistentVolumeClaim:
  registry:
    # 填写已经创建好的 PVC
    existingClaim: "harbor-pvc"
    storageClass: ""
    # 如果共用一个 PVC，需要设置子目录
    subPath: "registry"
    accessMode: ReadWriteOnce
    size: 5Gi
    annotations: {}
  jobservice:
    jobLog:
      existingClaim: "harbor-pvc"
      storageClass: ""
      subPath: "jobservice"
      accessMode: ReadWriteOnce
      size: 1Gi
      annotations: {}
  database:
    existingClaim: "harbor-pvc"
    storageClass: ""
    subPath: "database"
    accessMode: ReadWriteOnce
    size: 1Gi
    annotations: {}
  redis:
    existingClaim: "harbor-pvc"
    storageClass: ""
    subPath: "redis"
    accessMode: ReadWriteOnce
    size: 1Gi
    annotations: {}
  trivy:
    existingClaim: "harbor-pvc"
    storageClass: ""
    subPath: "trivy"
    accessMode: ReadWriteOnce
    size: 5Gi
    annotations: {}

# Admin 初始密码
harborAdminPassword: "Harbor12345"
```

</p>
</details> 

