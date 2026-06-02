## 从 0 配置 vscode 编写 51 单片机程序

### 1. 安装 vscode 和 插件

1. 安装vscode
2. 安装插件
    embedded IDE  主要的插件 用于管理构建
    C/C++  用于自动填充
    C/C++ Extension pack 用于补充上述的插件

### 2. 安装 kail5 usb转串口驱动

官网下载 kail5 并使用你能使用的方法 激活 kail5
> 51单片机安装 C51 版本的 kail5
> 如果你的 kail5 激活不成功 提示找不到构建库/构建器 需要在安装根目录下的配置文件中额外指定位置 (我也不知道为什么会这样 猜测是: 没有创建桌面快件方式 桌面快捷方式里面的启动参数有指定安装位置)

安装单片机厂商提供的串口驱动

### 3. 配置插件

在安装好 embedded IDE 后 会引导你去配置插件的相关设置
其中有一个编译器选项 要选择第一个 输入 kail5 的 uv4.exe 路径

### 4. 安装 Python 及其 三方包

1. 安装python 注意脚本需要加入环境变量
2. 配置镜像源
3. 安装stcgal包

在命令可以使用 stcgal 后 为安装成功

### 5. 配置编译选项

随便创建一个项目 并使用vscode打开这个项目

在安装好 C/C++ 后 按下 ctrl + shift + p 唤出搜索框 然后搜索 C/C++ 选择 C/C++: Edit Configurations (JSON)
在 includePath 中 添加:

1. 当前代码路径 (.c/.h文件写在哪就放哪)
2. kail5 的头文件路径 在 kail5 安装路径/C51/INC/Atmel

> 这个是让 C/C++ 插件提供补全功能

点击vscode工具栏侧 embedded IDE 按钮 会出现 EIED 配置
项目资源放入所有 .c .h 文件
构建配置选择C51编译器
烧录配置选择stcgal
C/C++ 属性配置:
    包含路径: 当前代码路径 (.c/.h文件写在哪就放哪)
    > 添加完之后应该是一个 . 代表当前路径 如果你的代码放在这里
    包含路径: kail5 的头文件路径 在 kail5 安装路径/C51/INC/Atmel

### 6. 尝试编译

创建一个 main.c 文件

```c
#include <REG52.H>;

void main() {
    while (1) {
        P2=0xFE;  // 1111 1110
    }
}
```

点击编译 没问题的话 连接单片机 点击烧录 重启单片机的电源即可烧录成功

参考视频: [https://www.bilibili.com/video/BV1ZDR9BJEwJ/](https://www.bilibili.com/video/BV1ZDR9BJEwJ/)