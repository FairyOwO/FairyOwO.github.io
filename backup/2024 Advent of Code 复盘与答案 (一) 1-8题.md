[Advent of Code](https://adventofcode.com/)

使用 python 编写, 没有整理代码, 所以非常乱(变量乱取名, 没有注释, 逻辑奇怪, 并非最佳实现)
可以使用 gpt 相关工具辅助查看

> 如果没有特意说明, 变量 a 统一存放所有原始字符串

## 第一题

[https://adventofcode.com/2024/day/1](https://adventofcode.com/2024/day/1)

### 1题

请将数字配对并测量它们之间的距离。将左侧列表中的最小数字与右侧列表中的最小数字配对，然后将左侧第二小的数字与右侧第二小的数字配对，依此类推。

左右两边先排序再相减, 取绝对值相加

<details><summary>Details</summary>
<p>

```python

list1, list2 = [], []

for i in a.split("\n"):
    m, n = i.split("   ")
    list1.append(int(m))
    list2.append(int(n))

ans = 0
for i in zip(sorted(list1), sorted(list2)):
    ans += abs(i[0] - i[1])
print(ans)

```

</p>
</details> 

经典的签到1题

### 2题

左侧列表中的每个数字乘以该数字在右侧列表中出现的次数后将其相加来计算总相似度得分

就按他说的做即可, 使用字典(map)管理出现次数

<details><summary>Details</summary>
<p>

```python
list1, list2 = [], []

for i in a.split("\n"):
    m, n = i.split("   ")
    list1.append(int(m))
    list2.append(int(n))

t = {}
for i in list2:
    if i in t:
        t[i] += 1
    else:
        t[i] = 1

ans = 0
for i in list1:
    if i in t:
        ans += i * t[i]

print(ans)
```

</p>
</details> 

经典的签到2题

## 第二题

[https://adventofcode.com/2024/day/2](https://adventofcode.com/2024/day/2)

### 1题

检测严格单调序列, 相邻两项之间差距在 [1, 3]

<details><summary>Details</summary>
<p>

```python
t = []
for i in a.split('\n'):
    t.append(list(map(int, i.split())))

ans = 0
for i in t:
    temp = 1 if i[1] - i[0] > 0 else -1
    for j in range(1, len(i)):

        if 0 < (i[j] - i[j - 1]) * temp < 4:
            continue
        else:
            ans += 1
            break


print(1000-ans)
```

</p>
</details> 

检测错误的, 总数减去错误的即可

### 2题

在一题的基础上, 错误的一行允许剔除一项, 检测剔除一项后能变成正确的一行

<details><summary>Details</summary>
<p>

```python
t = []
for i in a.split('\n'):
    t.append(list(map(int, i.split())))

ans = 0
for i in t:
    temp = 1 if i[1] - i[0] > 0 else -1
    for j in range(1, len(i)):
        if 0 < (i[j] - i[j - 1]) * temp < 4:
            continue
        else:
            # ans += 1
            for k in range(len(i)):
                copy = deepcopy(i)
                del copy[k]
                temp = 1 if copy[1] - copy[0] > 0 else -1
                for t in range(1, len(copy)):
                    if 0 < (copy[t] - copy[t - 1]) * temp < 4:
                        continue
                    else:
                        break
                else:
                    ans += 1
                    break
            break
    else:
        ans += 1

print(ans)
```

</p>
</details> 

没想到这题的捷径(想了一个捷径但是不对), 纯暴力解决, 每一项都遍历剔除的情况

> 需要注意的是, 不同语言的 拷贝 是否为深拷贝 (复制一份新值)

## 第三题

[https://adventofcode.com/2024/day/3](https://adventofcode.com/2024/day/3)

### 1题

正则提取 `mul\((\d{1,3}),(\d{1,3})\)`, 然后把两个捕获组捕获出来的东西乘起来

<details><summary>Details</summary>
<p>

```python
temp = re.findall(r'mul\((\d{1,3}),(\d{1,3})\)', a)
print(temp)
ans = 0
for i in temp:
    ans += int(i[0]) * int(i[1])

print(ans)
```

</p>
</details> 

### 2题

在一题的基础上加入开关, 在"开"后捕获, 在"关"后不捕获

<details><summary>Details</summary>
<p>

```python
allsearch = r'mul\((\d{1,3}),(\d{1,3})\)|(do\(\))|(don\'t\(\))'

aaa = re.findall(allsearch, a)
print(aaa)
ans = 0
flag = True
for i in aaa:
    if i[2]:
        flag = True
    if i[3]:
        flag = False
    if i[0] == '':
        continue
    if flag:
        ans += int(i[0]) * int(i[1])

print(ans)
```

</p>
</details>

## 第四题

[https://adventofcode.com/2024/day/4](https://adventofcode.com/2024/day/4)

### 1题

检测正反的 横着竖着斜着的字符串

<details><summary>Details</summary>
<p>

```python

t = a.split('\n')
aaa = []
for i in t:
    aaa.append(list(i))

print(aaa)
ans = 0

i = 0
while i < len(aaa):
    j = 0
    while j < len(aaa[i]):
        # 检测正的
        if aaa[i][j] == 'X':
            # 检测横着的
            if j + 3 < len(aaa[i]):
                if aaa[i][j+1] == 'M' and aaa[i][j+2] == 'A' and aaa[i][j+3] == 'S':
                    ans += 1
            # 检测竖着的
            if i + 3 < len(aaa):
                if aaa[i+1][j] == 'M' and aaa[i+2][j] == 'A' and aaa[i+3][j] == 'S':
                    ans += 1
            # 检测对角
            if j + 3 < len(aaa[i]) and i + 3 < len(aaa):
                if aaa[i+1][j+1] == 'M' and aaa[i+2][j+2] == 'A' and aaa[i+3][j+3] == 'S':
                    ans += 1
            if j - 3 >= 0 and i + 3 < len(aaa):
                if aaa[i+1][j-1] == 'M' and aaa[i+2][j-2] == 'A' and aaa[i+3][j-3] == 'S':
                    ans += 1
        # 检测反的
        if aaa[i][j] == 'S':
            # 检测横着的
            if j + 3 < len(aaa[i]):
                if aaa[i][j+1] == 'A' and aaa[i][j+2] == 'M' and aaa[i][j+3] == 'X':
                    ans += 1
            # 检测竖着的
            if i + 3 < len(aaa):
                if aaa[i+1][j] == 'A' and aaa[i+2][j] == 'M' and aaa[i+3][j] == 'X':
                    ans += 1
            # 检测对角
            if j + 3 < len(aaa[i]) and i + 3 < len(aaa):
                if aaa[i+1][j+1] == 'A' and aaa[i+2][j+2] == 'M' and aaa[i+3][j+3] == 'X':
                    ans += 1
            if j - 3 >= 0 and i + 3 < len(aaa):
                if aaa[i+1][j-1] == 'A' and aaa[i+2][j-2] == 'M' and aaa[i+3][j-3] == 'X':
                    ans += 1
        j += 1
    i += 1

print(ans)

```

</p>
</details> 

### 2题

检测特殊的形状, 形状有多种可能(正反)

<details><summary>Details</summary>
<p>

```python
t = a.split('\n')
aaa = []
for i in t:
    aaa.append(list(i))

print(aaa)
ans = 0

i = 0
while i + 2 < len(aaa):
    j = 0
    while j + 2 < len(aaa[i]):
        if aaa[i][j] == 'M':
            if aaa[i+1][j+1] == 'A' and aaa[i+2][j+2] == 'S':
                if aaa[i][j+2] =='S' and aaa[i+2][j] == 'M':
                    ans += 1
                elif aaa[i][j+2] == 'M' and aaa[i+2][j] == 'S':
                    ans += 1
        if aaa[i][j] == 'S':
            if aaa[i+1][j+1] == 'A' and aaa[i+2][j+2] == 'M':
                if aaa[i][j+2] =='M' and aaa[i+2][j] == 'S':
                    ans += 1
                elif aaa[i][j+2] == 'S' and aaa[i+2][j] == 'M':
                    ans += 1

        j += 1
    i += 1

print(ans)
```

</p>
</details> 

## 第五题

[https://adventofcode.com/2024/day/5](https://adventofcode.com/2024/day/5)

### 1题

分为两部分, 第一部分指定数字的顺序, 第二部分是待检测的数组
答案返回所有符合条件数组的最中间那个之和

<details><summary>Details</summary>
<p>

```python

ahead, after = a.split('\n\n')

ahead_procress = {}
for i in ahead.split('\n'):
    t = list(map(int, i.split('|')))
    if t[0] not in ahead_procress:
        ahead_procress[t[0]] = [t[1]]
    else:
        ahead_procress[t[0]].append(t[1])

print(ahead_procress)

after_procress = []
for i in after.split('\n'):
    after_procress.append(list(map(int, i.split(','))))

ans = 0

for i in after_procress:
    aaa = []
    for j in i[::-1]:
        if j not in aaa:
            if j in ahead_procress:
                aaa.extend(ahead_procress[j])
        else:
            break
    else:
        ans += i[len(i)//2]

print(ans)

```

</p>
</details>

### 2题

修改数组中的数字顺序, 使之满足第一部分给定的顺序排列
答案返回 不满足第一题条件的 修改好的, 所有最中间的那个数之和

<details><summary>Details</summary>
<p>

```python

ahead, after = a.split('\n\n')

ahead_procress = {}
ahead_procress2 = {}
for i in ahead.split('\n'):
    t = list(map(int, i.split('|')))
    if t[0] not in ahead_procress:
        ahead_procress[t[0]] = [t[1]]
    else:
        ahead_procress[t[0]].append(t[1])

    if t[1] not in ahead_procress2:
        ahead_procress2[t[1]] = [t[0]]
    else:
        ahead_procress2[t[1]].append(t[0])

print(ahead_procress)

after_procress = []
for i in after.split('\n'):
    after_procress.append(list(map(int, i.split(','))))

ans = 0

for i in after_procress:
    aaa = []
    bbb = []
    flag = True
    for j in i[::-1]:
        bbb.append(j)
        if j not in aaa:
            if j in ahead_procress:
                aaa.extend(ahead_procress[j])
        else:
            bbb.pop()
            flag = False
            for k in bbb:
                if k in ahead_procress2[j]:
                    bbb.insert(bbb.index(k), j)
                    break
            
    if flag is False:
        ans += bbb[len(bbb)//2]

print(ans)

```

</p>
</details> 

## 第六题

[https://adventofcode.com/2024/day/6](https://adventofcode.com/2024/day/6)

### 1题

一个图, 遇到障碍物往右转, 否则往前走, 访问多少个不同位置

<details><summary>Details</summary>
<p>

```python

aaa = [list(i) for i in a.split('\n')]


begin_pos = (0, 0)
for i in range(len(aaa)):
    for j in range(len(aaa[i])):
        if aaa[i][j] == '^':
            begin_pos = (i, j)
            aaa[i][j] = 'X'

print(begin_pos)
t = (-1, 0)

def turn(t):
    if t == (1, 0):
        return (0, -1)
    elif t == (0, 1):
        return (1, 0)
    elif t == (-1, 0):
        return (0, 1)
    elif t == (0, -1):
        return (-1, 0)


while 1:
    if aaa[begin_pos[0] + t[0]][begin_pos[1] + t[1]] == '#':
        t = turn(t)
    else:
        aaa[begin_pos[0] + t[0]][begin_pos[1] + t[1]] = 'X'
        begin_pos = (begin_pos[0] + t[0], begin_pos[1] + t[1])

    if begin_pos[0] in [0, len(aaa) - 1] or begin_pos[1] in [0, len(aaa[0]) - 1]:
        break

ans = 0
for i in range(len(aaa)):
    for j in range(len(aaa[0])):
        if aaa[i][j] == 'X':
            ans += 1

print(ans)
```

</p>
</details> 

### 2题

在1的基础上, 添加一个障碍物, 使得其循环

<details><summary>Details</summary>
<p>

```python
aaa = [list(i) for i in a.split('\n')]


begin_pos = (0, 0)
for i in range(len(aaa)):
    for j in range(len(aaa[i])):
        if aaa[i][j] == '^':
            begin_pos = (i, j)
            aaa[i][j] = 'X'

bak_begin_pos = deepcopy(begin_pos)
print(begin_pos)
t = (-1, 0)

def turn(t):
    if t == (1, 0):
        return (0, -1)
    elif t == (0, 1):
        return (1, 0)
    elif t == (-1, 0):
        return (0, 1)
    elif t == (0, -1):
        return (-1, 0)


while 1:
    if aaa[begin_pos[0] + t[0]][begin_pos[1] + t[1]] == '#':
        t = turn(t)
        continue
    else:
        aaa[begin_pos[0] + t[0]][begin_pos[1] + t[1]] = 'X'
        begin_pos = (begin_pos[0] + t[0], begin_pos[1] + t[1])

    if begin_pos[0] in [0, len(aaa) - 1] or begin_pos[1] in [0, len(aaa[0]) - 1]:
        break

ttt = [0, 0, 0, 0]
ttt = [deepcopy(ttt) for _ in range(len(aaa[0]))]
ttt = [deepcopy(ttt) for _ in range(len(aaa))]

ans = 0
aaa[bak_begin_pos[0]][bak_begin_pos[1]] = '.'
for i in tqdm(range(len(aaa))):
    for j in tqdm(range(len(aaa[i]))):
        if aaa[i][j] == 'X':
            begin_pos = deepcopy(bak_begin_pos)
            temp = deepcopy(aaa)
            temp[i][j] = '#'
            ttt_b = deepcopy(ttt)
            t = (-1, 0)
            flag = 0
            ttt_b[begin_pos[0]][begin_pos[1]][flag] = 1
            while 1:
                if temp[begin_pos[0] + t[0]][begin_pos[1] + t[1]] == '#':
                    t = turn(t)
                    flag = (flag + 1) % 4
                    continue
                else:
                    begin_pos = (begin_pos[0] + t[0], begin_pos[1] + t[1])
                    if ttt_b[begin_pos[0]][begin_pos[1]][flag] == 0:
                        ttt_b[begin_pos[0]][begin_pos[1]][flag] = 1
                    else:
                        ans += 1
                        break
                if begin_pos[0] in [0, len(ttt_b) - 1] or begin_pos[1] in [0, len(ttt_b[0]) - 1]:
                    break
print(ans)

```

</p>
</details> 

> 需要一个状态数组, 记录当前块下 运动方向, 如果运动方向相同的出现了两次, 则认为是循环

## 第七题

[https://adventofcode.com/2024/day/7](https://adventofcode.com/2024/day/7)

### 1题

给定运算符 相加 与 相乘(运算顺序固定从左往右, 无优先级), 给几个数字, 问他能不能变成另一个数字

<details><summary>Details</summary>
<p>

```python
def cal(numbers, ops):
    result = numbers[0]
    for i in range(len(ops)):
        if ops[i] == '+':
            result += numbers[i + 1]
        else:
            result *= numbers[i + 1]
    return result


ans = 0
ops = ['+', '*']
for line in a.splitlines():
    aa, bb = line.split(': ')
    b_int = [int(i) for i in bb.split()]
    aa = int(aa)
    for i in itertools.product(ops, repeat=len(b_int)-1):
        if aa == cal(b_int, i):
            ans += aa
            break
    
print(ans)
```

</p>
</details> 

> 任意个相同数组的笛卡尔积, 使用 `itertools.product` 生成

### 2题

在一题的基础上多了个类似于 合并 str (str1+str2) 的运算符

<details><summary>Details</summary>
<p>

```python

def cal(numbers, ops):
    result = numbers[0]
    for i in range(len(ops)):
        if ops[i] == '+':
            result += numbers[i + 1]
        elif ops[i] == '*':
            result *= numbers[i + 1]
        else:
            result = int(str(result) + str(numbers[i+1]))
    return result


ans = 0
ops = ['+', '*', '||']
for line in a.splitlines():
    aa, bb = line.split(': ')
    b_int = [int(i) for i in bb.split()]
    aa = int(aa)
    for i in itertools.product(ops, repeat=len(b_int)-1):
        if aa == cal(b_int, i):
            ans += aa
            break
    
print(ans)

```

</p>
</details> 

> `ops` 多一个 合并运算符, `cal` 多一种计算方法, 本质与第一题相同

## 第八题

[https://adventofcode.com/2024/day/8](https://adventofcode.com/2024/day/8)

## 1题

二维数组中, 字母/数字 连线, 向两边延长线延长 连线间的距离, 要求1. 在二维数组中2. 可以与其他字母数字重合 3. 与另一组 字母/数字 的延长处重合算一次

<details><summary>Details</summary>
<p>

```python

aaa = []
for i in a.splitlines():
    aaa.append(list(i))

print(aaa)

all_char = list(set(a))
all_char.remove('\n')
all_char.remove('.')

print(all_char)

def get_all_char_pos(char):
    char_pos = []
    for i in range(len(aaa)):
        for j in range(len(aaa[i])):
            if aaa[i][j] == char:
                char_pos.append((i, j))
    return char_pos

def is_valid(pos):
    if 0 <= pos[0] < len(aaa) and 0 <= pos[1] < len(aaa[0]):
        return True
    return False


empty_aaa = np.zeros((len(aaa), len(aaa[0])))

for i in all_char:
    char_pos = get_all_char_pos(i)
    for j in range(len(char_pos)):
        for k in range(j + 1, len(char_pos)):
            pos = (char_pos[j][0] - char_pos[k][0], char_pos[j][1] - char_pos[k][1])
            negative_pos = (-pos[0], -pos[1])
            if is_valid((char_pos[j][0] + pos[0], char_pos[j][1] + pos[1])):
                empty_aaa[char_pos[j][0] + pos[0]][char_pos[j][1] + pos[1]] = 1
            if is_valid((char_pos[k][0] + negative_pos[0], char_pos[k][1] + negative_pos[1])):
                empty_aaa[char_pos[k][0] + negative_pos[0]][char_pos[k][1] + negative_pos[1]] = 1

ans = 0
for i in range(len(empty_aaa)):
    for j in range(len(empty_aaa[i])):
        if empty_aaa[i][j] == 1:
            ans += 1
                
print(ans)

```

</p>
</details>

### 2题

在一题的基础上, 延长连线间的距离的k倍

<details><summary>Details</summary>
<p>

```python

aaa = []
for i in a.splitlines():
    aaa.append(list(i))

print(aaa)

all_char = list(set(a))
all_char.remove('\n')
all_char.remove('.')

print(all_char)

def get_all_char_pos(char):
    char_pos = []
    for i in range(len(aaa)):
        for j in range(len(aaa[i])):
            if aaa[i][j] == char:
                char_pos.append((i, j))
    return char_pos

def is_valid(pos):
    if 0 <= pos[0] < len(aaa) and 0 <= pos[1] < len(aaa[0]):
        return True
    return False

def get_k(pos):
    return min(len(aaa) // abs(pos[0]), len(aaa[0]) // abs(pos[1])) + 1


empty_aaa = np.zeros((len(aaa), len(aaa[0])))

for i in all_char:
    char_pos = get_all_char_pos(i)
    if len(char_pos) > 1:
        for j in char_pos:
            empty_aaa[j[0]][j[1]] = 1
    for j in range(len(char_pos)):
        for k in range(j + 1, len(char_pos)):
            pos = (char_pos[j][0] - char_pos[k][0], char_pos[j][1] - char_pos[k][1])
            k = get_k(pos)
            negative_pos = (-pos[0], -pos[1])
            for m in range(1, k + 1):
                new_pos = (char_pos[j][0] + pos[0] * m, char_pos[j][1] + pos[1] * m)
                if is_valid(new_pos):
                    empty_aaa[new_pos[0]][new_pos[1]] = 1
                new_negative_pos = (char_pos[j][0] + negative_pos[0] * m, char_pos[j][1] + negative_pos[1] * m)
                if is_valid(new_negative_pos):
                    empty_aaa[new_negative_pos[0]][new_negative_pos[1]] = 1


ans = 0
for i in range(len(empty_aaa)):
    for j in range(len(empty_aaa[i])):
        if empty_aaa[i][j] != 0:
            ans += 1
                
print(ans)

```

</p>
</details> 