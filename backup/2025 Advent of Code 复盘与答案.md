[Advent of Code](https://adventofcode.com/)

使用 python 编写, 没有整理代码, 所以非常乱(变量乱取名, 没有注释, 逻辑奇怪, 并非最佳实现)
可以使用 gpt 相关工具辅助查看

> 如果没有特意说明, 变量 aaa 统一存放所有原始字符串

> 今年的 ai 翻译工具好了很多 翻译的更容易懂了 许多

> 今年只有12题

## 第一题

[https://adventofcode.com/2025/day/1](https://adventofcode.com/2025/day/1)

### 1题

钟表 一共100个刻度(0-99) 往左转或者往右转 n 个度数
刚好转到 0 的次数是多少
模拟即可

<details><summary>Details</summary>
<p>

```python
counts = 0
now = 50
for line in aaa.splitlines():
    direction = line[0]
    steps = int(line[1:])
    if direction == 'L':
        now -= steps
    elif direction == 'R':
        now += steps
    
    now = (now + 100) % 100
    if now == 0:
        counts += 1

print(counts)
```

</p>
</details> 

### 2题

在1题的基础上 变成旋转过程中 经过0的次数 (转到0也算)

继续模拟 注意有的会转超过一圈

<details><summary>Details</summary>
<p>

```python

counts = 0
now = 50
for line in aaa.splitlines():
    is_zero_now = (now == 0)
    direction = line[0]
    steps = int(line[1:])

    cricles = steps // 100
    counts += cricles
    steps = steps % 100

    if direction == 'L':
        now -= steps
    elif direction == 'R':
        now += steps
    
    if not is_zero_now:
        if now <= 0:
            counts += 1
    
    if now >= 100:
        counts += 1

    now = (now + 100) % 100
    # if now == 0:
    #     counts += 1

print(counts)
```

</p>
</details> 

## 第二题

[https://adventofcode.com/2025/day/2](https://adventofcode.com/2025/day/2)

### 1题

给定范围 找到仅由重复两次的数序数字组成的任何 ID

<details><summary>Details</summary>
<p>

```python
input_list = aaa.split(",")

ans = 0
for range_pair in input_list:
    start, end = map(int, range_pair.split("-"))
    for number in range(start, end + 1):
        str_number = str(number)
        if len(str_number) % 2 != 0:
            # if len(set(str_number)) == 1:
            #     ans += number
            continue
        else:
            mid = len(str_number) // 2
            first_half = str_number[:mid]
            second_half = str_number[mid:]
            if first_half == second_half:
                ans += number
                continue
print(ans)
```

</p>
</details> 

### 2题

在1题的基础上 重复两次变成重复若干次

<details><summary>Details</summary>
<p>

```python
input_list = aaa.split(",")

def get_factors(n: int) -> list:
    factors = []
    for i in range(2, int(sqrt(n)) + 1):
        if n % i == 0:
            factors.append([i, n // i])
            
    factors.append([1, n])
    return factors


ans = 0
for range_pair in input_list:
    start, end = map(int, range_pair.split("-"))
    for number in range(start, end + 1):
        str_number = str(number)
        if len(str_number) < 2:
            continue
            
        factors = get_factors(len(str_number))
        for f1, f2 in factors:
            if str_number[:f1] * f2 == str_number:
                # print(number)
                ans += number
                break
            if 1 not in [f1, f2]:
                if str_number[:f2] * f1 == str_number:
                    # print(number)
                    ans += number
                    break

print(ans)
```

</p>
</details> 

## 第三题

[https://adventofcode.com/2025/day/3](https://adventofcode.com/2025/day/3)

### 1题

给定一串数字 这些数字在里面取 相对位置固定的两个数字 来组成最大的数

<details><summary>Details</summary>
<p>

```python
ans = 0
for line in aaa.split("\n"):
    max_left_num = int(line[0])
    max_left_index = 0
    for i, c in enumerate(line[1:-1]):
        if int(c) > max_left_num:
            max_left_num = int(c)
            max_left_index = i + 1
    max_right_num = int(line[-1])
    for j in range(len(line) - 2, max_left_index, -1):
        if int(line[j]) > max_right_num:
            max_right_num = int(line[j])
    
    ans += max_left_num * 10 + max_right_num

print(ans)
```

</p>
</details> 

### 2题

在1题的基础上 取12个数字 组成最大的数

<details><summary>Details</summary>
<p>

```python
ans = 0
n = 12
for line in aaa.split("\n"):
    dp = [[-1] * (n + 1) for _ in range(len(line) + 1)]
    dp[0][0] = 0
    for i in range(len(line)):
        for j in range(n + 1):
            if dp[i][j] == -1:
                continue
            # 不选
            dp[i + 1][j] = max(dp[i + 1][j], dp[i][j])
            # 选
            if j < n:
                dp[i + 1][j + 1] = max(dp[i + 1][j + 1], dp[i][j] * 10 + int(line[i]))
    ans += dp[len(line)][n]

print(ans)
```

</p>
</details> 

## 第四题

[https://adventofcode.com/2025/day/4](https://adventofcode.com/2025/day/4)

### 1题

给定一张图 标记所有东西的位置 找出 相邻八个位置中 满足一半或更多的位置是空的 有东西的数量

> 有点难描述 可以点进原文查看原题

<details><summary>Details</summary>
<p>

```python
a = [list(line) for line in aaa.splitlines()]

for row in range(len(a)):
    for col in range(len(a[0])):
        # if a[row][col] == '.':
        pairs = []
        # 统计周围八个少于4
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                if 0 <= row + i < len(a) and 0 <= col + j < len(a[0]):
                    if a[row + i][col + j] == '@' or a[row + i][col + j] == 'x':
                        pairs.append((row + i, col + j))
        # 少于4个@ 则将@变成x
        if len(pairs) < 4 and a[row][col] == '@':
            # for i, j in pairs:
            #     a[i][j] = 'x'
            a[row][col] = 'x'

ans = 0
for row in range(len(a)):
    for col in range(len(a[0])):
        if a[row][col] == 'x':
            ans += 1

print(ans)
```

</p>
</details> 

> 特别解法
> 可以将空的位置转换成0 有东西的位置转换成1 这样获得了一个矩阵
> 对他进行卷积 取值小于等于4的点即可 padding要是1

### 2题

在1题的基础上 找出来的位置都可以去掉变成空的 然后继续找

<details><summary>Details</summary>
<p>

```python
a = [list(line) for line in aaa.splitlines()]
ans = 0
num_of_at = sum(row.count('@') for row in a)
temp_num_of_at = num_of_at
while True:
    for row in range(len(a)):
        for col in range(len(a[0])):
            pairs = []
            # 统计周围八个少于4
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i == 0 and j == 0:
                        continue
                    if 0 <= row + i < len(a) and 0 <= col + j < len(a[0]):
                        if a[row + i][col + j] == '@' or a[row + i][col + j] == 'x':
                            pairs.append((row + i, col + j))
            # 少于4个@ 则将@变成x
            if len(pairs) < 4 and a[row][col] == '@':
                a[row][col] = '.'
    new_num_of_at = sum(row.count('@') for row in a)
    if new_num_of_at == temp_num_of_at:
        break
    temp_num_of_at = new_num_of_at

ans = num_of_at - new_num_of_at 

print(ans)
```

</p>
</details> 

> 相似的 也可以使用 1题的卷积思路

## 第五题

[https://adventofcode.com/2025/day/5](https://adventofcode.com/2025/day/5)

### 1题

给定一个范围 给定一系列数字 找到所有在范围内的数

<details><summary>Details</summary>
<p>

```python
fresh_ranges_text, vegtables_text = aaa.split("\n\n")

fresh_ranges = []
for line in fresh_ranges_text.strip().splitlines():
    start, end = map(int, line.split("-"))
    fresh_ranges.append((start, end))

ans = 0
for line in vegtables_text.strip().splitlines():
    iid = int(line)
    is_fresh = any(start <= iid <= end for start, end in fresh_ranges)
    ans += is_fresh

print(ans)
```

</p>
</details> 

### 2题

在1题的基础上 一系列数字不需要使用
找到范围内有多少数字
需要合并区间

<details><summary>Details</summary>
<p>

```python
fresh_ranges_text, vegtables_text = aaa.split("\n\n")

fresh_ranges = []
for line in fresh_ranges_text.strip().splitlines():
    start, end = map(int, line.split("-"))
    fresh_ranges.append((start, end))

# merge
fresh_ranges.sort()
merged_fresh = []
for start, end in fresh_ranges:
    if not merged_fresh or merged_fresh[-1][1] < start:
        merged_fresh.append((start, end))
    else:
        merged_fresh[-1] = (merged_fresh[-1][0], max(merged_fresh[-1][1], end))

ans = 0
for i, j in merged_fresh:
    ans += j - i + 1
print(ans)
```

</p>
</details> 

## 第六题

[https://adventofcode.com/2025/day/6](https://adventofcode.com/2025/day/6)

### 1题

竖着的加法乘法计算器

<details><summary>Details</summary>
<p>

```python
a = aaa.splitlines()
b = [list(map(int, line.split())) for line in a if line and not line.startswith('*')]
c = [line.split() for line in a[-1]]

# 去除split后的空字符串
b = [[num for num in line if num] for line in b]
c = [op[0] for op in c if op]


result = 0
for i in range(len(b[0])):
    match c[i]:
        case '+':
            result += b[0][i] + b[1][i] + b[2][i] + b[3][i]
        case '*':
            result += b[0][i] * b[1][i] * b[2][i] * b[3][i]

print(result)
```

</p>
</details> 

### 2题

竖着的 按位的 加减乘除计算器

<details><summary>Details</summary>
<p>

```python
ops = aaa.splitlines()[-1]
ops = [op for op in ops.split(" ") if op]
total_lines = aaa.splitlines()
lines = total_lines[:-1]

width = len(lines[0])
height = len(lines)

ans = 0
temp = []
number = 0

for pos_x in range(width - 1, -1, -1):
    for pos_y in range(height + 1):
        char = total_lines[pos_y][pos_x]
        if pos_y == height:
            temp.append(number)
            number = 0
        if char == " " and pos_y != height:
            continue
        elif char in '+*':
            if char == '+':
                ans += sum(temp)
            if char == '*':
                prod = 1
                for t in temp:
                    if t != 0:
                        prod *= t
                ans += prod
            temp = []
        else:
            if char == " ":
                continue
            number *= 10
            number += int(char)

print(ans)
```

</p>
</details> 

## 第七题

[https://adventofcode.com/2025/day/7](https://adventofcode.com/2025/day/7)

### 1题

图 发射一道激光往下 遇到 ^分成左右两股激光继续向下
会分裂多少次

<details><summary>Details</summary>
<p>

```python
a = [list(line) for line in aaa.splitlines()]
width = len(a[0])
height = len(a)

pos_s_x, pos_s_y = aaa.index('S'), 0

deque = [(pos_s_x, pos_s_y)]
ans = 0
visited = set()
for i in deque:
    x, y = i
    if y + 1 >= height:
        continue
    if a[y + 1][x] == '^':
        if y + 1 < height:
            if (x + 1, y + 1) not in visited:
                deque.append((x + 1, y + 1))
                visited.add((x + 1, y + 1))
            if (x - 1, y + 1) not in visited:
                deque.append((x - 1, y + 1))
                visited.add((x - 1, y + 1))
            ans += 1
    elif a[y + 1][x] == '.':
        if y + 1 < height:
            if (x, y + 1) not in visited:
                deque.append((x, y + 1))
                visited.add((x, y + 1))
    

print(len(visited))
print(ans)
```

</p>
</details> 

### 2题

在1题的基础上 有多少条不同路径的光线 (到达同一个目的地的不同路径光线 也需要计算)

<details><summary>Details</summary>
<p>

```python
a = [list(line) for line in aaa.splitlines()]
width = len(a[0])
height = len(a)

pos_s_x, pos_s_y = aaa.index('S'), 0

cache = {}

def dfs(x, y):
    if y + 1 >= height or x < 0 or x >= width:
        return 0
    if (x, y) in cache:
        return cache[(x, y)]
    
    if a[y + 1][x] == '^':
        result = dfs(x + 1, y + 1) + dfs(x - 1, y + 1) + 1
    elif a[y + 1][x] == '.':
        result = dfs(x, y + 1)
    else:
        result = 0
    
    cache[(x, y)] = result
    return result
    
print(dfs(pos_s_x, pos_s_y) + 1)
```

</p>
</details> 

## 第八题

[https://adventofcode.com/2025/day/8](https://adventofcode.com/2025/day/8)

### 1题

给定若干个x,y,z坐标的点 连接最近的1000个点 问 有多少簇

<details><summary>Details</summary>
<p>

```python
a = aaa.split('\n')
a = [list(map(int, i.split(','))) for i in a]

def distance(x1, y1, z1, x2, y2, z2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5

distances = []
for i in range(len(a)):
    for j in range(i + 1, len(a)):
        dist = distance(a[i][0], a[i][1], a[i][2], a[j][0], a[j][1], a[j][2])
        distances.append((dist, i, j))

distances.sort(key=lambda x: x[0])

needed_distances = distances[:1000]

# 邻接表
graph = {i: [] for i in range(len(a))}
for dist, i, j in needed_distances:
    graph[i].append((j, dist))
    graph[j].append((i, dist))

# dfs 遍历所有群组 获取所有群组节点数量
visited = [False] * len(a)
def dfs(node):
    stack = [node]
    size = 0
    while stack:
        current = stack.pop()
        if not visited[current]:
            visited[current] = True
            size += 1
            for neighbor, _ in graph[current]:
                if not visited[neighbor]:
                    stack.append(neighbor)
    return size
group_sizes = []
for i in range(len(a)):
    if not visited[i]:
        group_size = dfs(i)
        group_sizes.append(group_size)
group_sizes.sort()
print(group_sizes)
```

</p>
</details> 

### 2题

最小生成树
一点一点连接最近的两个点 最后连接的两个点是哪两个点

<details><summary>Details</summary>
<p>

```python
a = aaa.split('\n')
a = [list(map(int, i.split(','))) for i in a]

def distance(x1, y1, z1, x2, y2, z2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5

distances = []
for i in range(len(a)):
    for j in range(i + 1, len(a)):
        dist = distance(a[i][0], a[i][1], a[i][2], a[j][0], a[j][1], a[j][2])
        distances.append((dist, i, j))

distances.sort(key=lambda x: x[0])

# 构建最小生成树
parent = [i for i in range(len(a))]
num_components = len(a)
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    rootX = find(x)
    rootY = find(y)
    if rootX != rootY:
        parent[rootY] = rootX
        return True
    return False

mst_edges = []
for dist, i, j in distances:
    if union(i, j):
        mst_edges.append((dist, i, j))
        num_components -= 1
        if num_components == 1:
            print("i, j, dist:", i, j, dist)
            break

# 获取 i, j 对应的点坐标
point_i = a[i]
point_j = a[j]

print("Point i:", point_i)
print("Point j:", point_j)
```

</p>
</details> 

## 第九题

[https://adventofcode.com/2025/day/9](https://adventofcode.com/2025/day/9)

### 1题

给定若干个x y 坐标点 找到最大的 以这两个点作为对角的 矩形

<details><summary>Details</summary>
<p>

```python
a = [(int(x), int(y)) for x, y in (line.split(",") for line in aaa.split("\n"))]

def cal_space(x1, y1, x2, y2):
    return (abs(x1 - x2) + 1) * (abs(y1 - y2) + 1)

ans = 0
for i in range(len(a) - 1):
    for j in range(i + 1, len(a)):
        x1, y1 = a[i]
        x2, y2 = a[j]
        space = cal_space(x1, y1, x2, y2)
        if space > ans:
            ans = space

print(ans)
```

</p>
</details> 

### 2题

在1题的基础上 上一个点 的x或者y与下一个点的x或者y相同 由此绕出了一个形状 需要找到最大的 在这个形状中的 以这两个点作为对角的矩形

> 检查这个矩形在不在这个形状里

<details><summary>Details</summary>
<p>

```python
a = [(int(x), int(y)) for x, y in (line.split(",") for line in aaa.split("\n"))]

def cal_space(x1, y1, x2, y2):
    return (abs(x1 - x2) + 1) * (abs(y1 - y2) + 1)

def check(edges, x1, y1, x2, y2) -> bool:
    min_x = min(x1, x2)
    max_x = max(x1, x2)
    min_y = min(y1, y2)
    max_y = max(y1, y2)
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    def ray_cast(midpoint, edges):
        count = 0
        mx, my = midpoint
        for (x3, y3), (x4, y4) in edges:
            if x3 > mx and x4 > mx:
                if min(y3, y4) <= my < max(y3, y4):
                    count += 1
        return count % 2 == 1
    
    if not ray_cast((mid_x, mid_y), edges):
        return False

    for (x3, y3), (x4, y4) in edges:
        if x3 == x4:
            if min_x < x3 < max_x:
                edge_min_y = min(y3, y4)
                edge_max_y = max(y3, y4)
                if not (edge_max_y <= min_y or edge_min_y >= max_y):
                    return False
                
        elif y3 == y4:
            if min_y < y3 < max_y:
                edge_min_x = min(x3, x4)
                edge_max_x = max(x3, x4)
                if not (edge_max_x <= min_x or edge_min_x >= max_x):
                    return False
    return True


edges = []
for i in range(0, len(a) - 1):
    edges.append((a[i], a[i + 1]))

edges.append((a[-1], a[0]))

ans = 0
for i in range(0, len(a) - 1):
    for j in range(i + 1, len(a)):
        x1, y1 = a[i]
        x2, y2 = a[j]
        space = cal_space(x1, y1, x2, y2)
        if space <= ans:
            continue
        if check(edges, x1, y1, x2, y2):
            ans = space

print(ans)
```

</p>
</details> 

## 第十题

[https://adventofcode.com/2025/day/10](https://adventofcode.com/2025/day/10)

### 1题

给定 灯最终需要到达的状态
给定 每个按钮 按下按钮后 各个灯的变化
给定 计数器 (第一题 用不到)

求 将完全关闭的灯泡 切换到 最终需要到达的状态需要按多少次

<details><summary>Details</summary>
<p>

```python
import re

a = aaa.splitlines()
t = []
for i in a:
    lights = re.findall(r'\[(.*?)\]', i)[0]
    steps = re.findall(r'\((.*?)\)', i)
    joltages = re.findall(r'\{(.*?)\}', i)[0]
    steps = [list(map(int, s.split(','))) if s else [] for s in steps]
    joltages = list(map(int, joltages.split(',')))
    lights_len = len(lights)
    t.append((lights, lights_len, steps, joltages))

ans = 0
for lights, lights_len, steps, joltages in t:
    # 最少操作数让所有灯满足lights中的要求 不需要考虑电压
    dp = {0: 0}  # 状态压缩dp
    for step in steps:
        next_dp = {}
        for state, cnt in dp.items():
            # 不按
            if state not in next_dp or next_dp[state] > cnt:
                next_dp[state] = cnt
            # 按
            next_state = state
            for s in step:
                next_state ^= (1 << s)
            if next_state not in next_dp or next_dp[next_state] > cnt + 1:
                next_dp[next_state] = cnt + 1
        dp = next_dp
    final_state = 0
    for i in range(lights_len):
        if lights[i] == '#':
            final_state |= (1 << i)
    ans += dp[final_state]

print(ans)
```

</p>
</details> 

### 2题

题意转换为
给定 灯最终需要到达的状态 (第二题用不到)
给定 每个按钮 按下按钮后 各个灯电压加一
给定 电压计数器

达到给定电压(超过也行) 需要按多少次
> 这是一题 带有约束的 正整数线性方程求解 直接调用库求解即可

<details><summary>Details</summary>
<p>

```python
import re

a = aaa.splitlines()
t = []
for i in a:
    lights = re.findall(r'\[(.*?)\]', i)[0]
    steps = re.findall(r'\((.*?)\)', i)
    joltages = re.findall(r'\{(.*?)\}', i)[0]
    steps = [list(map(int, s.split(','))) if s else [] for s in steps]
    joltages = list(map(int, joltages.split(',')))
    lights_len = len(lights)
    t.append((lights, lights_len, steps, joltages))

import z3

def solve(lights, lights_len, steps, joltages):
    # 设 steps 列表中的第 i 个元素为xi 约束: xi大于等于0 且整数
    # sum(Aji * xi) = bj
    # Aji = 1 表示第i个按钮可以控制到第j个灯, 也就是可以增加第j个灯的电压
    # xi 表示第i个按钮按下的次数
    # bj 表示第j个灯的目标电压
    # 求解 minimize sum(xi)
    s = z3.Solver()
    x = [z3.Int(f'x{i}') for i in range(len(steps))]
    for i in range(len(steps)):
        s.add(x[i] >= 0)
    for j in range(lights_len):
        coeffs = []
        for i in range(len(steps)):
            if j in steps[i]:
                coeffs.append(x[i])
        s.add(z3.Sum(coeffs) == joltages[j])
    obj = z3.Sum(x)
    h = z3.Optimize()
    h.add(s.assertions())
    h.minimize(obj)
    if h.check() == z3.sat:
        m = h.model()
        res = [m.evaluate(x[i]).as_long() for i in range(len(steps))]
        return res
    else:
        return [-1] * len(steps)
    
results = []
for lights, lights_len, steps, joltages in t:
    res = solve(lights, lights_len, steps, joltages)
    results.append(res)
for res in results:
    print(' '.join(map(str, res)))

ans = sum(sum(r) for r in results)
print(ans)
```

</p>
</details> 

## 第十一题

[https://adventofcode.com/2025/day/11](https://adventofcode.com/2025/day/11)

### 1题

给定 若干个映射 f(a) -> b
b不一定只有一个 找到从 特定字符开始 到结束字符的 每条路径

<details><summary>Details</summary>
<p>

```python
a = {}
for i in aaa.splitlines():
    p, q = i.split(": ")
    a[p] = q.split(" ")


def dfs(t):
    if t == 'out':
        return 0
    nt = a[t]
    b = 0
    for i in nt:
        b += dfs(i)
    return b + len(nt) - 1

print(dfs('you') + 1)
```

</p>
</details> 

### 2题

 在1题的基础上 特定字符换了一下 且需要同时经过 某两个字符(顺序无关)

<details><summary>Details</summary>
<p>

```python
import functools

a = {}
for i in aaa.splitlines():
    p, q = i.split(": ")
    a[p] = q.split(" ")


@functools.lru_cache(None)
def dfs(t, visited_dac, visited_fft):
    if t == 'dac':
        visited_dac = True
    if t == 'fft':
        visited_fft = True
    if t == 'out':
        if visited_dac and visited_fft:
            return 1, visited_dac, visited_fft
        else:
            return 0, visited_dac, visited_fft
    nt = a[t]
    b = 0
    for i in nt:
        b += dfs(i, visited_dac, visited_fft)[0]
    return b, visited_dac, visited_fft

print(dfs('svr', False, False))
```

</p>
</details> 

> 路线一下就不是同一个数量级的

## 第十二题

[https://adventofcode.com/2025/day/12](https://adventofcode.com/2025/day/12)

### 只有一题

建议看原文

给定特定形状 给定一个二维箱子 大小有限 可以旋转 以及见缝插针

问这些东西能不能放进去

> 这是一题整蛊题 正常来说算法实现非常困难

<details><summary>Details</summary>
<p>

```python
all_data = aaa.split('\n\n')
t = all_data[:-1]
gifts = []
for i in t:
    gift = []
    for line in i.split('\n'):
        if ':' in line:
            continue
        gift.append(line)
    gifts.append(gift)

print(gifts)
gift_area = []
for i in gifts:
    area_num = 0
    for line in i:
        area_num += line.count("#")
    gift_area.append(area_num)

print(gift_area)

ans = 0
a = all_data[-1]
for line in a.splitlines():
    region, gift_nums_str = line.split(': ')
    wide_, long_ = region.split('x')
    gift_nums = gift_nums_str.split(" ")
    all_area = int(wide_) * int(long_)
    needed_area = 0
    for i in range(len(gift_nums)):
        needed_area += int(gift_nums[i]) * gift_area[i]
    if needed_area <= all_area:
        ans += 1

print(ans)
```

</p>
</details> 

> 仅需检测面积即可