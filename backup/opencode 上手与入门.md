## opencode 上手与入门

随着code agent的发展 效率与可用性已经大幅上涨
部分mcp服务器 (例如content7或者是定向检索代码的mcp服务器) 可以明显提升效果
这里记录一下 如何安装并配置opencode

这里选择的是 opencode + 反重力 cli to api 的方案

同类agent:
claude code(cc)
codex(cx)

> 如果选择更省心的方案推荐 claude code + 购买中转服务 + 自行配置一些mcp(尽管不是必须的, 但配置好后效果会大幅提升)
> 如果选择多 agent 方案 推荐 主模型使用 opus 细节使用 gpt5.x-codex 干活模型使用 glm-4.x [https://github.com/bfly123/claude_code_bridge](https://github.com/bfly123/claude_code_bridge) 注: 模型具有时效性

### 安装 你喜欢的包管理器

这里选择的是 [nodejs](https://nodejs.org/zh-cn/download/) 中的 npx

> 许多 mcp 服务器使用了npx来启动, 如果仅使用 opencode 的话 可以不进行安装

### 安装 opencode

详见文档, 这里仅作记录

[https://github.com/anomalyco/opencode](https://github.com/anomalyco/opencode)

```sh
# YOLO
curl -fsSL https://opencode.ai/install | bash

# Package managers
npm i -g opencode-ai@latest        # or bun/pnpm/yarn
scoop bucket add extras; scoop install extras/opencode  # Windows
choco install opencode             # Windows
brew install opencode              # macOS and Linux
paru -S opencode-bin               # Arch Linux
mise use -g opencode               # Any OS
nix run nixpkgs#opencode           # or github:anomalyco/opencode for latest dev branch
```

或者是前往 [release](https://github.com/anomalyco/opencode/releases) 直接下载安装包或者免安装的 zip

### 获取 api

> 有成品账号代理直连,中转等方案能拿到api可以忽略此章节

#### 反重力

这是一个google出的类似于cursor的代码编辑器 开通google ai pro会员, 在正确的地区, 即可使用

> 你可以用家庭组 大号需要拉五个小号进家庭组 注意地区与付款锁 这样额度可以乘以六倍

这里使用 [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI) 来将其转换成api提供给opencode使用

> 注意 当前 opencode 与 CLIProxyAPI 一起使用 会有兼容性问题 调用工具的时候会中断 需要手动继续
> [6.6.89](https://github.com/router-for-me/CLIProxyAPI/releases/tag/v6.6.89) 已修复

1. 下载 release 并解压
2. 修改配置文件 `config.example.yaml` 并修改成 `config.yaml`

    ```yaml
    proxy-url: socks5://user:pass@192.168.1.1:1080/  # 你需要改成你的, 这里proxy用于访问google 所以ip需要纯净
    allow-remote: false  # 如果你在服务器上搭建, 则需要允许
    secret-key: 123  # 一定要配置不然进不去
    ```

3. 启动, 并访问 [webui](http://localhost:8317/management.html)
4. 根据实际情况配置
5. OAuth 登录 反重力
    > 注意 这里需要 CLIProxyAPI 与 浏览器 一起经过代理

这个时候已经可以用了 response 地址是: [http://localhost:8317/v1/responses](http://localhost:8317/v1/responses)

> 不止有 response 的地址 还支持不同厂的地址, 例如 openai 地址

模型列表是:

```text
gemini-2.5-flash
gemini-3-pro-image-preview
gemini-2.5-computer-use-preview-10-2025
gemini-3-pro-preview
gemini-3-flash-preview
gemini-2.5-flash-lite
gemini-claude-sonnet-4-5
gemini-claude-opus-4-5-thinking
gemini-claude-sonnet-4-5-thinking
gpt-oss-120b-medium
```

> 注意 因为羊毛被薅太多 这里claude与gpt模型用量非常少

### opencode 配置代理与中转

opencode使用环境变量配置代理

```sh
export HTTPS_PROXY=https://proxy.example.com:8080
export HTTP_PROXY=http://proxy.example.com:8080
export NO_PROXY=localhost,127.0.0.1
# 注意 NO_PROXY 也要
```

opencode 需要手动写配置文件 `~/config/opencode/opencode.json`

参考:

```jsonc
{
  "plugin": [
    "oh-my-opencode",
    // "opencode-antigravity-auth@1.2.7"  // 自带的反重力插件 这里我们用其他方案代替, 所以不需要
  ],
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "Antigravity": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Antigravity",
      "options": {
        "baseURL": "http://localhost:8317/v1"  // 默认CLIProxyAPI地址
      },
      "models": {
        "gemini-3-pro-preview": {
          "name": "Gemini 3 Pro preview (Antigravity)",
          "thinking": true,
          "attachment": true,
          "limit": {
            "context": 1048576,
            "output": 65535
          },
          "modalities": {
            "input": [
              "text",
              "image",
              "pdf"
            ],
            "output": [
              "text"
            ]
          }
        },
        "gemini-3-flash-preview": {
          "name": "Gemini 3 Flash (Antigravity)",
          "attachment": true,
          "limit": {
            "context": 1048576,
            "output": 65536
          },
          "modalities": {
            "input": [
              "text",
              "image",
              "pdf"
            ],
            "output": [
              "text"
            ]
          }
        },
        "gemini-2.5-flash-lite": {
          "name": "Gemini 2.5 Flash Lite (Antigravity)",
          "attachment": true,
          "limit": {
            "context": 1048576,
            "output": 65536
          },
          "modalities": {
            "input": [
              "text",
              "image",
              "pdf"
            ],
            "output": [
              "text"
            ]
          }
        }
      }
    },
  }
}
```

> 你可以通过ai来自动生成配置文件

如果这边是中转的 api, 那么仅需要修改 opencode.json 中 修改 provider 中的 options.baseURL
新版本的 opencode 可能不会创建这个配置文件, 需要手动配置
配置好后去 opencode 中配置 api 即可

### 安装oh my opencode

#### 使用opencode agent 安装

既然都安装了 opencode 不妨来试试
直接将 [oh my opencode](https://github.com/code-yeongyu/oh-my-opencode) 项目地址发送给 opencode 来让他帮你安装吧

[https://github.com/code-yeongyu/oh-my-opencode](https://github.com/code-yeongyu/oh-my-opencode)

> 注意 这里opencode 会自动帮你安装 bunx 来安装这个项目, 如果不想要的话 需要强调使用npx安装
> oh my opencode 也可以用来登录反重力账号, 插件名称是 opencode-antigravity-auth, 将反重力支持模型到opencode中运行 但如果之前使用 CLIProxyAPI 来使用反重力, 则无需安装这个插件

#### 手动

详见文档, 这里仅作记录

```sh
bunx oh-my-opencode install
# or use npx if bunx doesn't work
npx oh-my-opencode install
```

#### 配置文件

之前使用我们自己的 opencode 配置文件 这里当然需要自行配置 oh-my-opencode 的配置文件, 当然你也可以让ai来做这件事情 详见文档

这里摘抄一下每个agent是干什么的

```text
Sisyphus (anthropic/claude-opus-4-5)：默认 Agent。 OpenCode 专属的强力 AI 编排器。指挥专业子 Agent 搞定复杂任务。主打后台任务委派和 Todo 驱动。用 Claude Opus 4.5 加上扩展思考（32k token 预算），智商拉满。
oracle (openai/gpt-5.2)：架构师、代码审查员、战略家。GPT-5.2 的逻辑推理和深度分析能力不是盖的。致敬 AmpCode。
librarian (anthropic/claude-sonnet-4-5 或 google/gemini-3-flash)：多仓库分析、查文档、找示例。配置 Antigravity 认证时使用 Gemini 3 Flash，否则使用 Claude Sonnet 4.5 深入理解代码库，GitHub 调研，给出的答案都有据可查。致敬 AmpCode。
explore (opencode/grok-code、google/gemini-3-flash 或 anthropic/claude-haiku-4-5)：极速代码库扫描、模式匹配。配置 Antigravity 认证时使用 Gemini 3 Flash，Claude max20 可用时使用 Haiku，否则用 Grok。致敬 Claude Code。
frontend-ui-ux-engineer (google/gemini-3-pro-preview)：设计师出身的程序员。UI 做得那是真漂亮。Gemini 写这种创意美观的代码是一绝。
document-writer (google/gemini-3-pro-preview)：技术写作专家。Gemini 文笔好，写出来的东西读着顺畅。
multimodal-looker (google/gemini-3-flash)：视觉内容专家。PDF、图片、图表，看一眼就知道里头有啥。
```

```jsonc
{
  "$schema": "https://raw.githubusercontent.com/code-yeongyu/oh-my-opencode/master/assets/oh-my-opencode.schema.json",
  "google_auth": false,
  "agents": {
    "Sisyphus": {
      "model": "Antigravity/gemini-claude-opus-4-5-thinking"
    },
    "librarian": {
      "model": "Antigravity/gemini-3-flash"
    },
    "explore": {
      "model": "Antigravity/gemini-3-flash"
    },
    "oracle": {
      "model": "Antigravity/gemini-3-pro-preview"  // 后期更换成 gpt 5.2
    },
    "frontend-ui-ux-engineer": {
      "model": "Antigravity/gemini-3-pro-preview"
    },
    "document-writer": {
      "model": "Antigravity/gemini-3-flash"
    },
    "multimodal-looker": {
      "model": "Antigravity/gemini-3-flash"
    }
  }
}
```

### ref

https://linux.do/t/topic/1404993
https://opencode.ai/docs
https://help.router-for.me/cn/
https://github.com/code-yeongyu/oh-my-opencode
