[Advent of Code](https://adventofcode.com/)

TODO, 每天更新一题一周前的题目

使用 python 编写, 没有整理代码, 所以非常乱(变量乱取名, 没有注释, 逻辑奇怪, 并非最佳实现)
可以使用 gpt 相关工具辅助查看

> 如果没有特意说明, 变量 a 统一存放所有原始字符串

## 第九题

[https://adventofcode.com/2024/day/9](https://adventofcode.com/2024/day/9)

### 1题

一个磁盘中的文件表示方法, 每两位数字各表示, x块文件, x块空的区域
例如
`12345`
表示 一块文件 两块空文件 三块文件 四块空文件 五块文件
每个文件从前往后的id为他们在磁盘中的顺序, 例如上述可以表示为
`0..111....22222`

希望从后往前的将文件塞入从前往后中空的地方

例如
```text
0..111....22222
02.111....2222.
022111....222..
0221112...22...
02211122..2....
022111222......
```

最后输出每个块位置*id号之和

<details><summary>Details</summary>
<p>

```python
# 构建
t = []
flag = True
file_id = 0
for i in list(a):
    if flag:
        for _ in range(int(i)):
            t.append(file_id)
        flag = False
        file_id += 1
    else:
        for _ in range(int(i)):
            t.append('.')
        flag = True

i = 0
j = len(t) - 1

while i < j:
    if t[i] == '.':
        if t[j] != '.':
            t[i], t[j] = t[j], t[i]
        else:
            i -= 1
        j -= 1
    i += 1

ans = 0
for i, j in enumerate(t):
    if j != '.':
        ans += i * j
print(ans)

```

</p>
</details> 

### 2题

在一题的基础上, 修改从后往前放入从前往后空区域的方法, 从单个块移动转换为整个文件移动, 例如
```text
00...111...2...333.44.5555.6666.777.888899
0099.111...2...333.44.5555.6666.777.8888..
0099.1117772...333.44.5555.6666.....8888..
0099.111777244.333....5555.6666.....8888..
00992111777.44.333....5555.6666.....8888..
```

最后输出每个块位置*id号之和

<details><summary>Details</summary>
<p>

```python
# 构建
t = []
flag = True
file_id = 0
for i in list(a):
    if flag:
        for _ in range(int(i)):
            t.append(file_id)
        flag = False
        file_id += 1
    else:
        for _ in range(int(i)):
            t.append('.')
        flag = True


def get_file_size(file_id):
    return t.count(file_id)

for i in range(file_id - 1, -1, -1):
    file_size = get_file_size(i)
    file_index = t.index(i)
    flag = False
    size = 0
    for id, j in enumerate(t):
        if j == '.':
            flag = True
        else:
            flag = False
            size = 0
        
        if flag:
            size += 1
            if size == file_size:
                # change位置
                for k in range(id, id - file_size, -1):
                    t[k], t[file_index] = t[file_index], t[k]
                    file_index += 1

                break
        if id >= file_index:
            break

ans = 0
for i, j in enumerate(t):
    if j != '.':
        ans += i * j
print(ans)

```

</p>
</details> 

## 第十题

[https://adventofcode.com/2024/day/10](https://adventofcode.com/2024/day/10)

### 1题

给定一幅由数字组成的地图, 从 0 开始, 一步一步上下左右移动到 9
求一副图中, 每一个 0 能到达的 9 有多少个, 输出他们的和

<details><summary>Details</summary>
<p>

```python
aaa = []
for i in a.splitlines():
    aaa.append(list(map(int, list(i))))

# print(aaa)

# 获取所有 0 跟 9 的位置
pos_0 = []
pos_9 = []
for i in range(len(aaa)):
    for j in range(len(aaa[i])):
        if aaa[i][j] == 0:
            pos_0.append((i, j))
        elif aaa[i][j] == 9:
            pos_9.append((i, j))

def get_allow_pos(pos):
    for i in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
        next_step = (pos[0] + i[0], pos[1] + i[1])
        if len(aaa) > next_step[0] >= 0 and len(aaa[0]) > next_step[1] >= 0:
            if aaa[next_step[0]][next_step[1]] == aaa[pos[0]][pos[1]] + 1:
                yield (pos[0] + i[0], pos[1] + i[1])


def dfs(pos, pos_9):
    if pos == pos_9:
        return 1
    aa = 0
    for p in get_allow_pos(pos):
        aa = dfs(p, pos_9)
        if aa > 0:
            return aa
    return aa

ans = 0
for i in pos_0:
    for j in pos_9:
        ans += dfs(i, j)

print(ans)

``` 

</p>
</details> 

### 2题

求 0 到每一个 9 有多少种不同的走法 (注意与第一题的区别, 第一题只要求到达, 第二题需要找到所有路线)

<details><summary>Details</summary>
<p>

```python
aaa = []
for i in a.splitlines():
    aaa.append(list(map(int, list(i))))

# print(aaa)

# 获取所有 0 跟 9 的位置
pos_0 = []
pos_9 = []
for i in range(len(aaa)):
    for j in range(len(aaa[i])):
        if aaa[i][j] == 0:
            pos_0.append((i, j))
        elif aaa[i][j] == 9:
            pos_9.append((i, j))

def get_allow_pos(pos):
    for i in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
        next_step = (pos[0] + i[0], pos[1] + i[1])
        if len(aaa) > next_step[0] >= 0 and len(aaa[0]) > next_step[1] >= 0:
            if aaa[next_step[0]][next_step[1]] == aaa[pos[0]][pos[1]] + 1:
                yield (pos[0] + i[0], pos[1] + i[1])


def dfs(pos, pos_9):
    if pos == pos_9:
        return 1
    aa = 0
    for p in get_allow_pos(pos):
        aa = dfs(p, pos_9)
        if aa > 0:
            return aa
    return aa

ans = 0
for i in pos_0:
    for j in pos_9:
        ans += dfs(i, j)

print(ans)

```

</p>
</details> 

> 在一题的基础上, dfs固定返回1 变成 dfs 返回上一次dfs + 1

## 第十一题

[https://adventofcode.com/2024/day/11](https://adventofcode.com/2024/day/11)

## 1题

给定一串数字, 根据以下规则变换

1. 如果数字是0, 则数字是1
2. 如果数字是偶数位, 则数字变成两个数字, 左半跟右半, 例如 1000 变成 10 跟 00, 不保留前导0, 00变成0
3. 如果没有碰到前两条规则, 则数字=数字*2024

顺序都会被保留 (但是这题没有用到, 而且会误导第二题)
模拟上述规则25次

<details><summary>Details</summary>
<p>

```python
aa = a.split(' ')

for _ in range(25):
    i = 0
    while i < len(aa):
        if aa[i] == '0':
            aa[i] = '1'
        elif len(aa[i]) % 2 == 0:
            left = aa[i][:len(aa[i]) // 2]
            left = str(int(left))
            right = aa[i][len(aa[i]) // 2:]
            right = str(int(right))
            aa.insert(i+1, right)
            aa[i] = left
            i += 1
        else:
            aa[i] = str(int(aa[i]) * 2024)
        
        i += 1

print(len(aa))
```

</p>
</details> 

> 模拟即可, 25次蛮少的可以直接出来

### 2题

在1题的基础上, 模拟75次

<details><summary>Details</summary>
<p>

```python
from collections import defaultdict
from tqdm import tqdm
aa = list(map(int, a.split(' ')))

def get_length(num):
    i = 0
    while num > 0:
        num //= 10
        i += 1
    return i

t = defaultdict(int)

for i in aa:
    t[i] += 1

for _ in tqdm(range(75)):
    tt = defaultdict(int)
    for i, j in t.items():
        length = get_length(i)
        if i == 0:
            tt[1] += j
        elif length % 2 == 0:
            tt[i // 10 ** (length // 2)] += j
            tt[i % 10 ** (length // 2)] += j
        else:
            tt[i * 2024] += j
        
        t = tt

print(sum(t.values()))
```

</p>
</details> 

> `tqdm` 是为了监控速度, 非必要引入
> 与第一题不同, 这题指数爆炸, 75次会超时, 因为答案不要求顺序, 所以可以用缓存