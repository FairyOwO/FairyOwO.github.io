## 概述

逆向新手题 `jocker.exe` 的windows可执行文件。该程序要求用户输入一个 Flag，并通过一系列验证来判断输入的正确性。通过对 main 函数伪代码的初步分析，发现程序包含自修改代码（Self-Modifying Code, SMC）以及多阶段的 Flag 验证逻辑。

> 以下不能显示的图片即程序，下载后需要修改后缀，建议使用虚拟机等工具来进行分析以防程序被植入病毒。

![Image](https://github.com/user-attachments/assets/032b632a-f08c-4a5c-b066-ca211fa89956)

## 环境工具

- windows11
- IDA Pro 9.1
- python（自带于IDA中）

## 分析流程

### 反编译与初步分析

在 ctf 的逆向过程中首先需要找到入口，也就是 `main` 函数。
使用 IDA Pro 加载 jocker.exe 后，首先定位到 `main` 函数。其就在 IDA 左侧 function 窗口，segment 在 .text 中。其伪代码如下所示：

```c
int __cdecl main(int argc, const char **argv, const char **envp)
{
  char Str[50];      // [esp+12h] [ebp-96h] BYREF
  char Destination[80]; // [esp+44h] [ebp-64h] BYREF
  DWORD flOldProtect; // [esp+94h] [ebp-14h] BYREF
  size_t v7;         // [esp+98h] [ebp-10h]
  int i;             // [esp+9Ch] [ebp-Ch]

  __main();
  puts("please input you flag:");
  if ( !VirtualProtect(encrypt, 0xC8u, 4u, &flOldProtect) )
    exit(1);
  scanf("%40s", Str);
  v7 = strlen(Str);
  if ( v7 != 24 )
  {
    puts("Wrong!");
    exit(0);
  }
  strcpy(Destination, Str);
  wrong(Str);
  omg(Str);
  for ( i = 0; i <= 186; ++i )
    *((_BYTE *)encrypt + i) ^= 0x41u;
  if ( encrypt(Destination) )
    finally(Destination);
  return 0;
}
```

> 在跳转到 main 汇编函数段的时候，按下空格会进入图视角，按下 f5 会进入伪代码视角

关键信息：
1. 程序通过 `scanf("%40s", Str)`  获取用户输入，并检查其长度。此处 `v7 != 24` 的判断表明 Flag 的总长度应为 24 字节。
2. 存在自修改代码
    - `VirtualProtect(encrypt, 0xC8u, 4u, &flOldProtect)`: 这行代码修改了 `encrypt` 函数所在内存区域的保护属性，使其变为可写 (PAGE_READWRITE)。
    - `for ( i = 0; i <= 186; ++i ) *((_BYTE *)encrypt + i) ^= 0x41u;`: 这是一个循环，对 `encrypt` 函数起始地址后的 187 字节（从 `encrypt` 到 `encrypt + 186`）进行 `0x41` 的异或操作。这表明 `encrypt` 函数在程序加载时是加密状态，运行时才被解密。
3. `encrypt(Destination)` 和 `finally(Destination)` 是两个关键的验证函数。`Destination` 变量保存了用户输入的 Flag 副本。
4. `wrong(Str)` 和 `omg(Str)` 在 `strcpy(Destination, Str)` 之后被调用，可能用于对 `Str` 进行混淆或无关操作。由于 `encrypt` 和 `finally` 使用的是 `Destination`，因此可以暂时忽略 `wrong` 和 `omg` 对 Flag 核心逻辑的影响。

### 解密自修改代码（修复 `encrypt` 函数）

由于 `encrypt` 函数在静态文件中是加密状态，直接在 IDA Pro 查看其伪代码将是无效的。需要先执行其自修改逻辑。
这里有两种方法，可以通过 IDA Pro 动态调试， dbg 等动态调试工具在解密完之后打断点，来获取其解密的代码。
也可以审阅其代码来获得其自修改逻辑，根据他的其自修改逻辑来解密获得源代码。
这里自修改逻辑较为简单（直接在 `main` 函数中即可得知，且为简单的异或操作），故选择第二种方法

操作流程
1. 定位 encrypt 函数地址: 在 main 函数的伪代码中双击 encrypt，可以跳转到其汇编视图，并获取其起始地址。本例中，该地址为 0x401500。
2. 编写 IDA PYTHON 并运行：
    ```python
    import idc

    start_addr = 0x401500  # encrypt 函数的起始地址
    length = 187           # 循环长度，0到186，共187个字节
    key = 0x41             # 异或的 Key
    
    print(f"开始修复地址 {hex(start_addr)} 处的代码...")
    
    for i in range(length):
        current_addr = start_addr + i
        original_byte = idc.get_wide_byte(current_addr)
        new_byte = original_byte ^ key
        idc.patch_byte(current_addr, new_byte)
    
    print("修复完成！请刷新反汇编视图。")
    ```
    > 其位于左上角 file 下拉框中 Script Command 下 选择 python 语言
3. 刷新重建 IDA 的函数识别功能
    1. 跳转到 0x401500 (G 快捷键，输入地址)。
    2. 选中该区域，按 U 键取消定义。
    3. 将光标置于 0x401500 处，按 C 键将其转换为代码。
    4. 按 P 键创建函数，让 IDA Pro 重新分析该区域的代码逻辑。
    5. 按 F5 键查看 encrypt 函数的伪代码。
   
### 解密 flag 第一部分

操作流程
1. 分析修复好的 `encrtpt` 函数
    ```c
    int __cdecl encrypt(char *a1)
    {
      _DWORD v2[19]; // [esp+1Ch] [ebp-6Ch] BYREF
      int v3;        // [esp+68h] [ebp-20h]
      int i;         // [esp+6Ch] [ebp-1Ch]
    
      v3 = 1;
      qmemcpy(v2, &unk_403040, sizeof(v2)); // 将数据从 unk_403040 拷贝到 v2
      for ( i = 0; i <= 18; ++i )           // 循环 19 次 (0-18)
      {
        if ( (char)(a1[i] ^ Buffer[i]) != v2[i] ) // 核心比较逻辑
        {
          puts("wrong ~");
          v3 = 0;
          exit(0);
        }
      }
      puts("come here");
      return v3;
    }
    ```
    1. 循环 `i <= 18` 表明 `encrypt` 函数验证了 Flag 的前 19 个字节。结合 `main` 函数中 Flag 总长度为 24 字节的判断，可知还有 5 个字节未被 `encrypt` 验证。
    2. `(char)(a1[i] ^ Buffer[i]) != v2[i]
        - `a1[i]` 是用户输入的 Flag 字符。
        - `Buffer[i]` 是一个硬编码的字符串。在汇编代码中，`Buffer` 被IDA自动识别为 `"hahahaha_do_you_find_me?"`。
        - `v2[i]` 是从 `unk_403040` 拷贝过来的 4 字节整数数组。
    3. 解密公式：根据 `(a1[i] ^ Buffer[i]) == v2[i]`，可以推导出 `a1[i] = v2[i] ^ Buffer[i]`。
2. 提取密文数据
    `v2` 数组的密文数据位于 `unk_403040`。在 IDA Pro 中按 `G` 键跳转到 `0x403040` 处。可以看到一系列 4 字节的数值（例如 `0E 00 00 00` 代表十进制的 14）。
3. 解密密文数据，编写 IDA PYTHON 并运行
    ```python
    import idc

    cipher_addr = 0x403040 # 密文数据起始地址
    length = 19            # 验证的 Flag 长度
    key_string = "hahahaha_do_you_find_me?" # Buffer 字符串

    print("-" * 20)
    print("开始解密第一部分 Flag...")

    flag_part1 = ""
    for i in range(length):
        cipher_val = idc.get_wide_dword(cipher_addr + i * 4) # 读取 4 字节的密文值
        key_val = ord(key_string[i])                         # 获取 Key 字符串对应字符的 ASCII 值
        flag_char = chr(cipher_val ^ key_val)                # 异或还原
        flag_part1 += flag_char

    print("解密结果(Flag第一部分): " + flag_part1)
    print("-" * 20)
    ```
   
最终获得flag的第一部分：`flag{d07abccf8a410c`。

### 获取 flag 第二部分

程序在 `encrypt(Destination)` 成功后调用 `finally(Destination)`。根据程序运行时的提示信息 "I hide the last part, you will not succeed!!!" 和 "最后五位隐藏在finally函数中"，可知 Flag 的剩余 5 字节在此函数中验证。

`finally` 函数的伪代码如下：

```c
int __cdecl finally(char *a1)
{
  __time32_t v1; // eax
  char v3[7];    // [esp+13h] [ebp-15h] BYREF
  __int16 v4;    // [esp+1Ah] [ebp-Eh]
  int v5;        // [esp+1Ch] [ebp-Ch]

  strcpy(v3, "%tp&:"); // 关键信息：v3 被初始化为 "%tp&:"
  v1 = time(0);
  srand(v1);
  v5 = rand() % 100; // 生成一个随机数
  v3[6] = 0;         // 填充终止符，实际 v3 长度为 5
  v4 = 0;
  // 核心比较逻辑，但被随机数 v5 混淆
  if ( (v3[(unsigned __int8)v3[5]] != a1[(unsigned __int8)v3[5]]) == v5 )
    return puts("Really??? Did you find it?OMG!!!");
  else
    return puts("I hide the last part, you will not succeed!!!");
}
```

关键信息：
1. 函数中使用了 `time(0)`、`srand(v1)` 和 `rand() % 100` 生成随机数 `v5`，并将其用于判断条件。
2. 最重要的线索在于 `strcpy(v3, "%tp&:")`。这个字符串的长度正好是 5 个字节（不包含终止符），与我们缺失的 Flag 长度吻合。
3. ctf 的 flag 格式遵循 `flag{...}` 格式，因此最后一位字符必然是 `}`。

考虑到整个挑战中使用了异或操作，推测 `finally` 函数也使用异或来隐藏 Flag 的最后一部分。
已知 `v3` 字符串为 `"%tp&:"`。
我们知道 Flag 的最后一位是 `}` (ASCII 0x7D)。
`v3` 字符串的最后一位是 `:` (ASCII 0x3A)。
假设存在一个 Key，使得 `v3[4] ^ Key = Flag[23]`。
即 `0x3A ^ Key = 0x7D`。
通过计算 `Key = 0x3A ^ 0x7D = 0x47`。

使用计算出的 Key (`0x47`) 对 `"%tp&:"` 的所有字符进行异或解密。

编写 IDA PYTHON 并运行
```python
cipher_last = "%tp&:" # finally 函数中隐藏的字符串
key = 0x47            # 推测出的异或 Key (通过 ':' ^ '}' 计算)

flag_last_part = ""

print("-" * 20)
print("正在解密最后一部分 Flag...")

for char in cipher_last:
    decoded_char = chr(ord(char) ^ key) # 异或解密
    flag_last_part += decoded_char

print("解密结果: " + flag_last_part)

flag_first_part = "flag{d07abccf8a410c" # 第一部分 Flag
final_flag = flag_first_part + flag_last_part
print("\n完整 Flag: " + final_flag)
print("-" * 20)
```
运行上述脚本，得到 Flag 的最后 5 字节：`b37a}`。
将两部分 Flag 拼接，得到最终的 Flag：`flag{d07abccf8a410cb37a}`。

