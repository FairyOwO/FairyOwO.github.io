## 记录一些学习 rust 中的非语法问题

> 常来说 每个语言都有专门的工作流，用这个语言编写某个项目/功能时，需要遵循一定的规范，这里做一点记录。

### 换源

自动
使用 `chsrc` 进行换源，详见上一篇工具

手动
在 `$HOME/.cargo/config.toml` 中
```toml
[source.crates-io]
replace-with = 'mirror'

[source.mirror]
registry = "sparse+https://mirrors.bfsu.edu.cn/crates.io-index/"
```

### 三方库

1. [crates.io](https://crates.io/)
    官方支持的 rust 注册三方库
2. [lib.rs](https://lib.rs/)    
    > 我没用过 更好的分类不同库与更好的搜索
3. [docs.rs](https://docs.rs/)
    官方文档托管
3. github / google / 等 直接搜索
4. [Awesome Rust](https://github.com/rust-unofficial/awesome-rust)
   社区维护的优秀 rust 三方库

对于一些固定的功能来说 三方库的选择几乎是固定的，例如：
解析 JSON：序列化与反序列 serde_json
随机数：rand
异步：tokio
错误处理：thiserror anyhow
日志：log tracing
等

> 我对三方库的积累还不够多 欢迎补充
> 查找三方库与其他语言类似 使用人数 维护状态 社区与依赖（对于 rust 来说 需要额外看unsafe 代码块多不多） 编译速度 等


对于一个项目来说 首先需要确定“大件”，也就是对核心功能做技术选型

例如 web 服务器 axum, actix-web, rocket
数据库 sqlx, diesel

### 一些常用命令

```
cargo test  # 放在 #[cfg(test)] 宏 或者在 tests/ 中的测试代码
cargo build --release
cargo run
cargo doc  # 在代码中进行文档注释(///) 后 能为函数与结构体自动编写文档
cargo clippy  # rust 自带的 linter 能检查很多引用与实现问题
cargo check  # 快速检查代码问题 不 build 就可以看到代码能不能成功 build
cargo fmt  # 代码格式化工具
cargo add  # 添加三方库
```

### Cargo.toml

[package]: 你的项目的元数据（名称、版本、作者等）。
[dependencies]: 你所有的第三方库（称为 "crates"）都在这里声明。
[dev-dependencies]: cargo test 或者 cargo bench 才会被编译，例如我在测试中加入了额外的错误处理anyhow，但代码中没有使用

```toml
[dependencies]
# 我需要 serde 的 1.0 版本，只需要他的  derive 功能
serde = { version = "1.0", features = ["derive"] }

[dev-dependencies]
anyhow = "1.0"
```

cargo 自动生成 Cargo.lock 文件 会锁定项目使用的版本

[workspace] 管理多个 crates
> 还没接触到

### 思想

rust 的所有权与生命周期思想与其他 C like 的不太一样，有的在 C like 中很简单的代码在 rust 中非常复杂，例如链表

C/C++：你持有原始指针，你告诉编译器：“相信我，我知道我在做什么。”

Java/Python：你持有引用，GC（垃圾回收器）在背后帮你处理生命周期。

Rust：你必须在编译期就向编译器证明你的所有权和借用是 100% 安全的。编译器不“相信”你，它只“验证”你。

Result 与 Option 另一个重大的思想转变是错误处理。Rust 没有 try...catch 异常，也没有 null。

可能会失败的操作，返回 Result<T, E> (成功或错误)。

可能没有值的操作，返回 Option<T> (有值或无值)。 这种方式强迫你在编译时就必须显式处理所有可能的失败路径，极大地提高了代码的健壮性。

### 编译器

rust 编译器是我见过的错误信息最完善的编译器，他会将问题具体指出，并给出一个可能的修复方案（有时候不是很准）
在ai时代，这无疑非常方便于 ai coding agent，尽管在 benchmark 中 各类 ai 对于 rust 支持并非 python 或者 java 这么完善，但有编译器详细的报错信息，在一轮代码输出错误的情况下，很快就能自行修复

