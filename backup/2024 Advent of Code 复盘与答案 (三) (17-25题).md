[Advent of Code](https://adventofcode.com/)

使用 python 编写, 没有整理代码, 所以非常乱(变量乱取名, 没有注释, 逻辑奇怪, 并非最佳实现)
可以使用 gpt 相关工具辅助查看

> 如果没有特意说明, 变量 a 统一存放所有原始字符串

## 第十七题

[https://adventofcode.com/2024/day/17](https://adventofcode.com/2024/day/17)

### 1题

尝试模拟计算机, 寄存器+特定操作码

> 题非常麻烦, 建议看原文

<details><summary>Details</summary>
<p>

```python

def main():
    input_data = a.splitlines()
    
    # Parse initial register values
    registers = {
        'A': int(input_data[0].split(': ')[1]),
        'B': int(input_data[1].split(': ')[1]),
        'C': int(input_data[2].split(': ')[1]),
    }
    
    # Parse program
    program = list(map(int, input_data[4].split(': ')[1].split(',')))
    
    output = []
    ip = 0
    program_length = len(program)
    
    def get_combo_value(operand):
        if operand == 4:
            return registers['A']
        elif operand == 5:
            return registers['B']
        elif operand == 6:
            return registers['C']
        else:
            return operand  # 0-3
    
    while ip < program_length:
        if ip + 1 >= program_length:
            break  # Invalid instruction, halt
        opcode = program[ip]
        operand = program[ip + 1]
        
        if opcode == 0:  # adv
            denominator = 2 ** get_combo_value(operand)
            registers['A'] = registers['A'] // denominator
            ip += 2
        elif opcode == 1:  # bxl
            registers['B'] ^= operand
            ip += 2
        elif opcode == 2:  # bst
            registers['B'] = get_combo_value(operand) % 8
            ip += 2
        elif opcode == 3:  # jnz
            if registers['A'] != 0:
                ip = operand
            else:
                ip += 2
        elif opcode == 4:  # bxc
            registers['B'] ^= registers['C']
            ip += 2
        elif opcode == 5:  # out
            output_value = get_combo_value(operand) % 8
            output.append(str(output_value))
            ip += 2
        elif opcode == 6:  # bdv
            denominator = 2 ** get_combo_value(operand)
            registers['B'] = registers['A'] // denominator
            ip += 2
        elif opcode == 7:  # cdv
            denominator = 2 ** get_combo_value(operand)
            registers['C'] = registers['A'] // denominator
            ip += 2
        else:
            ip += 2  # Invalid opcode, skip
        
    print(','.join(output))

if __name__ == "__main__":
    main()
``` 

</p>
</details> 

### 2题

在给定操作码下, 推断最小的寄存器中的满足条件的值(寄存器中的值等于输出的值)

<details><summary>Details</summary>
<p>

```python
program = list(map(int, a.splitlines()[4].split(': ')[1].split(',')))
n = len(program)
program = program[::-1]

choices = next_choices = [0]
for oo in program:
    choices = next_choices
    next_choices = []
    while choices:
        a = choices.pop()
        for i in range(8):
            aa = (a << 3) | i
            bb = i ^ 1
            cc = aa >> bb
            bb = (bb ^ cc ^ 4) % 8
            if bb == oo:
                next_choices.append(aa)
    # print(next_choices)
    
print(min(next_choices))

# z = (A % 8) ^ 1
# out = z ^ (A >> z) ^ 4
``` 

</p>
</details> 

## 第十八题

[https://adventofcode.com/2024/day/18](https://adventofcode.com/2024/day/18)

### 1题

走二维迷宫

<details><summary>Details</summary>
<p>

```python

import heapq
from collections import defaultdict

# print(all_map)

start = (0, 0)
end = (70, 70)




flag = False
temp = a.splitlines()
temp = temp[:1024]
# while not flag:
all_map = [['.' for _ in range(71)] for _ in range(71)]
for i in temp:
    x, y = map(int, i.split(','))
    all_map[x][y] = '#'


def get_allow_pos(pos):
    for i in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
        next_step = (pos[0] + i[0], pos[1] + i[1])
        if len(all_map) > next_step[0] >= 0 and len(all_map[0]) > next_step[1] >= 0:
            if all_map[next_step[0]][next_step[1]] == '.':
                yield (pos[0] + i[0], pos[1] + i[1])

node = []
heapq.heappush(node, (0, start))
visited = defaultdict(int)
visited[start] = 0

while node:
    step, (x, y) = heapq.heappop(node)
    if (x, y) == end:
        flag = True
        print(step)
        break
    if visited[(x, y)] < step:
        continue
    for next in get_allow_pos((x, y)):
        if next not in visited or step + 1 < visited[(next[0], next[1])]:
            visited[(next[0], next[1])] = step + 1
            heapq.heappush(node, (step + 1, next))
``` 

</p>
</details> 

### 2题

让迷宫无解的最新的坐标是多少

<details><summary>Details</summary>
<p>

```python

import heapq
from collections import defaultdict

# print(all_map)

start = (0, 0)
end = (70, 70)

flag = False
temp = a.splitlines()
temp.append('')
while not flag:
    all_map = [['.' for _ in range(71)] for _ in range(71)]
    print(temp.pop())
    for i in temp:
        x, y = map(int, i.split(','))
        all_map[x][y] = '#'

    def get_allow_pos(pos):
        for i in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            next_step = (pos[0] + i[0], pos[1] + i[1])
            if len(all_map) > next_step[0] >= 0 and len(all_map[0]) > next_step[1] >= 0:
                if all_map[next_step[0]][next_step[1]] == '.':
                    yield (pos[0] + i[0], pos[1] + i[1])

    node = []
    heapq.heappush(node, (0, start))
    visited = defaultdict(int)
    visited[start] = 0

    while node:
        step, (x, y) = heapq.heappop(node)
        if (x, y) == end:
            flag = True
            print(step)
            break
        if visited[(x, y)] < step:
            continue
        for next in get_allow_pos((x, y)):
            if next not in visited or step + 1 < visited[(next[0], next[1])]:
                visited[(next[0], next[1])] = step + 1
                heapq.heappush(node, (step + 1, next))
``` 

</p>
</details> 

## 第十九题

[https://adventofcode.com/2024/day/19](https://adventofcode.com/2024/day/19)

### 1题

给定几个图案片段, 与几种图案(包括不可完成图案), 找出有多少种设计方案

<details><summary>Details</summary>
<p>

```python

aa = a.splitlines()
has = aa[0]
has = has.split(', ')
all_possible = aa[2:]


def dp(s):
    n = len(s)
    dp_state = [0] * (n + 1)
    dp_state[0] = 1
    for i in range(n):
        if dp_state[i]:
            for j in has:
                l = len(j)
                if i + l <= n and s[i:i+l] == j:
                    dp_state[i+l] = 1
    return dp_state[n]

print(dp('abaa'))

ans = 0
for i in all_possible:
    ans += dp(i)

print(ans)

``` 

</p>
</details> 

### 2题

再将每个设计的不同方法的数量加起来

<details><summary>Details</summary>
<p>

```python
aa = a.splitlines()
has = aa[0]
has = has.split(', ')
all_possible = aa[2:]

def dp(s):
    n = len(s)
    dp_state = [0] * (n + 1)
    dp_state[0] = 1
    for i in range(n):
        if dp_state[i]:
            for j in has:
                l = len(j)
                if i + l <= n and s[i:i+l] == j:
                    dp_state[i+l] += dp_state[i]
    return dp_state[n]


ans = 0
for i in all_possible:
    ans += dp(i)

print(ans)

``` 

</p>
</details> 

## 第二十题

[https://adventofcode.com/2024/day/20](https://adventofcode.com/2024/day/20)

### 1题

走迷宫, 但允许穿墙 走一步1ps, 允许穿墙一次 最多2 皮秒内禁止碰撞一次
求有多少种能省100ps的穿墙方式

<details><summary>Details</summary>
<p>

```python

aa = [list(i) for i in a.splitlines()]
walls = [(-1, -1)]
# 开始结束
for i in range(len(aa)):
    for j in range(len(aa[i])):
        if aa[i][j] == 'S':
            start = (i, j)
            aa[i][j] = '.'
        elif aa[i][j] == 'E':
            end = (i, j)
            aa[i][j] = '.'
        elif aa[i][j] == '#':
            walls.append((i, j))

new_walls = []
for i in range(len(walls)):
    if i == 0:
        continue
    if walls[i][0] == 0 or walls[i][0] == len(aa) - 1 or walls[i][1] == 0 or walls[i][1] == len(aa[0]) - 1:
        continue
    # 如果wall的上下或者左右无墙, 则加入new_walls, 注意边界
    if walls[i][0] - 1 >= 0 and aa[walls[i][0] - 1][walls[i][1]] != '#' and walls[i][0] + 1 < len(aa) and aa[walls[i][0] + 1][walls[i][1]] != '#':
        new_walls.append((walls[i][0], walls[i][1]))
    if walls[i][1] - 1 >= 0 and aa[walls[i][0]][walls[i][1] - 1] != '#' and walls[i][1] + 1 < len(aa[0]) and aa[walls[i][0]][walls[i][1] + 1] != '#':
        new_walls.append((walls[i][0], walls[i][1]))

# print(new_walls)
walls = new_walls

print(start, end)

from copy import deepcopy
import heapq
from collections import defaultdict

ans = []
aaa = 0
flag = True
for i in walls:
    all_map = deepcopy(aa)
    if not flag:
        all_map[i[0]][i[1]] = '.'

    def get_allow_pos(pos):
        for i in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            next_step = (pos[0] + i[0], pos[1] + i[1])
            if len(all_map) > next_step[0] >= 0 and len(all_map[0]) > next_step[1] >= 0:
                if all_map[next_step[0]][next_step[1]] == '.':
                    yield (pos[0] + i[0], pos[1] + i[1])


    node = []
    heapq.heappush(node, (0, start))
    visited = defaultdict(int)
    visited[start] = 0
    while node:
        step, (x, y) = heapq.heappop(node)
        if (x, y) == end:
            if flag:
                aaa = step
                flag = False
            else:
                if aaa - step >= 100:
                    ans.append(aaa - step)
            break
        if visited[(x, y)] < step:
            continue
        for next in get_allow_pos((x, y)):
            if next not in visited or step + 1 < visited[(next[0], next[1])]:
                visited[(next[0], next[1])] = step + 1
                heapq.heappush(node, (step + 1, next))
print(len(ans))

``` 

</p>
</details> 

### 2题

穿墙允许20ps, 作弊不需要使用全部 20 皮秒；作弊可以持续任意时间，最长可达 20 皮秒, 但还是只能用一次, 任何未使用的作弊时间都会丢失；它不能被保存以供以后再次作弊。求节约100ps的方案有多少种

<details><summary>Details</summary>
<p>

```python
from collections import deque

def read_input():
    grid = [list(i) for i in a.splitlines()]
    return grid

def find_positions(grid, target):
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == target:
                return (r, c)
    return None

def bfs(grid, start):
    rows, cols = len(grid), len(grid[0])
    dist = [[float('inf')] * cols for _ in range(rows)]
    queue = deque()
    start_r, start_c = start
    dist[start_r][start_c] = 0
    queue.append((start_r, start_c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] != '#' and dist[nr][nc] == float('inf'):
                    dist[nr][nc] = dist[r][c] + 1
                    queue.append((nr, nc))
    return dist

def main():
    grid = read_input()
    S = find_positions(grid, 'S')
    E = find_positions(grid, 'E')
    if not S or not E:
        print(0)
        return
    dist_S = bfs(grid, S)
    dist_E = bfs(grid, E)
    D = dist_S[E[0]][E[1]]
    if D == float('inf'):
        print(0)
        return
    rows, cols = len(grid), len(grid[0])
    cheat_count = 0
    for a_r in range(rows):
        for a_c in range(cols):
            if dist_S[a_r][a_c] == float('inf'):
                continue
            for b_r in range(rows):
                for b_c in range(cols):
                    if dist_E[b_r][b_c] == float('inf'):
                        continue
                    t_a_b = abs(a_r - b_r) + abs(a_c - b_c)
                    if t_a_b <= 20 and dist_S[a_r][a_c] + dist_E[b_r][b_c] + t_a_b <= D - 100:
                        cheat_count += 1
    print(cheat_count)

if __name__ == "__main__":
    main()
``` 

</p>
</details> 

## 第二十一题

[https://adventofcode.com/2024/day/21](https://adventofcode.com/2024/day/21)

### 1题

操纵 控制方向盘 去 操纵 控制方向盘 去 操纵 控制方向盘 去操纵 数字键盘

<details><summary>Details</summary>
<p>

```python

aa = a.splitlines()

def is_available(x, y, type=0):
    if type == 0:  # 数字键盘
        if x < 0 or y < 0 or x > 2 or y > 3:
            return False
        
        if x == 0 and y == 3:
            return False
        return True
    elif type == 1:
        if x < 0 or y < 0 or x > 2 or y > 1:  # 移动
            return False
        if x == 0 and y == 0:
            return False
    else:
        raise ValueError("Invalid type")


def get_keyboard():
    return [
        [7, 8, 9],
        [4, 5, 6],
        [1, 2, 3],
        ['*', 0, 'A']
    ]

def get_keyboard_map():
    return {
        '1': (2, 0),
        '2': (2, 1),
        '3': (2, 2),
        '4': (1, 0),
        '5': (1, 1),
        '6': (1, 2),
        '7': (0, 0),
        '8': (0, 1),
        '9': (0, 2),
        '*': (3, 0),
        '0': (3, 1),
        'A': (3, 2)
    }

def get_posboard():
    return [
        ['*', '^', 'A'],
        ['<', 'v', '>'],
    ]

def get_posboard_map():
    return {
        '^': (0, 1),
        'v': (1, 1),
        '<': (1, 0),
        '>': (1, 2),
        'A': (0, 2),
        '*': (0, 0),
    }


def get_path(way: str, type=0):
    ans = ''
    if type == 0:  # 数字键盘
        now_pos = (3, 2)
        keyboard_map = get_keyboard_map()
        for i in range(len(way)):
            new_pos = keyboard_map[way[i]]
            if now_pos[0] == 3 and new_pos[1] == 0:
                for step in range(abs(now_pos[0] - new_pos[0])):
                    ans += '^' if now_pos[0] > new_pos[0] else 'v'
                for step in range(abs(now_pos[1] - new_pos[1])):
                    ans += '<' if now_pos[1] > new_pos[1] else '>'
            else:
                for step in range(abs(now_pos[1] - new_pos[1])):
                    ans += '<' if now_pos[1] > new_pos[1] else '>'
                for step in range(abs(now_pos[0] - new_pos[0])):
                    ans += '^' if now_pos[0] > new_pos[0] else 'v'

            ans += 'A'
            now_pos = new_pos

    if type == 1:  # 方向键盘
        now_pos = (0, 2)
        posboard_map = get_posboard_map()
        for i in range(len(way)):
            new_pos = posboard_map[way[i]]
            if now_pos[0] == 0 and new_pos[1] == 0:
                for step in range(abs(now_pos[0] - new_pos[0])):
                    ans += '^' if now_pos[0] > new_pos[0] else 'v'
                for step in range(abs(now_pos[1] - new_pos[1])):
                    ans += '<' if now_pos[1] > new_pos[1] else '>'
            else:
                for step in range(abs(now_pos[1] - new_pos[1])):
                    ans += '<' if now_pos[1] > new_pos[1] else '>'
                for step in range(abs(now_pos[0] - new_pos[0])):
                    ans += '^' if now_pos[0] > new_pos[0] else 'v'    
            ans += 'A'
            now_pos = new_pos

    return ans

t = 0
for i in aa:
    first = get_path(i)
    second = get_path(first, type=1)
    third = get_path(second, type=1)
    t += len(third) * int(i[:-1])
    print('#########')
    print(first)
    print(second)
    print(third)
    print(len(third), i[:-1])

print(t)

``` 

</p>
</details> 

### 2题

上述一题中的 控制方向盘 变为25个

> 未解决

## 第二十二题

[https://adventofcode.com/2024/day/22](https://adventofcode.com/2024/day/22)

### 1题

按规律计算

<details><summary>Details</summary>
<p>

```python
def mix(num, secret):
    return num ^ secret

def prune(num):
    return num % 16777216

def cal(num):
    secret = num * 64
    num = prune(mix(num, secret))
    secret = num // 32
    num = prune(mix(num, secret))
    secret = num * 2048
    num = prune(mix(num, secret))
    return num


aa = [int(i) for i in a.split('\n')]
ans = 0
for i in aa:
    t = i
    for j in range(2000):
        t = cal(t)
    ans += t
print(ans)
``` 

</p>
</details> 

### 2题

更新计算方式, 根据序列判断方式来计算最大收益

<details><summary>Details</summary>
<p>

```python

def mix(num, secret):
    return num ^ secret

def prune(num):
    return num % 16777216

def cal(num):
    secret = num * 64
    num = prune(mix(num, secret))
    secret = num // 32
    num = prune(mix(num, secret))
    secret = num * 2048
    num = prune(mix(num, secret))
    return num

def generate_secrets(initial_secret, num=2000):
    secrets = [initial_secret]
    for _ in range(num):
        secret = cal(secrets[-1])
        secrets.append(secret)
    return secrets

def generate_prices_and_changes(secrets):
    prices = [s % 10 for s in secrets]
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    return prices, changes

def change_seq_to_index(seq):
    a, b, c, d = seq
    a = a + 9
    b = b + 9
    c = c + 9
    d = d + 9
    index = a * 19**3 + b * 19**2 + c * 19 + d
    return index

def main():
    buyers = [int(i) for i in a.splitlines()]
    total_bananas = [0] * (19**4)
    
    for buyer_initial in buyers:
        secrets = generate_secrets(buyer_initial)
        prices, changes = generate_prices_and_changes(secrets)
        found = {}
        for j in range(len(changes)-3):
            seq = (changes[j], changes[j+1], changes[j+2], changes[j+3])
            if seq not in found:
                if j+4 < len(prices):
                    found[seq] = prices[j+4]
                else:
                    found[seq] = 0  # 防止越界
        for seq, price in found.items():
            index = change_seq_to_index(seq)
            total_bananas[index] += price
    max_bananas = max(total_bananas)
    print("最大香蕉数:", max_bananas)

if __name__ == "__main__":
    main()
``` 

</p>
</details> 

## 第二十三题

[https://adventofcode.com/2024/day/23](https://adventofcode.com/2024/day/23)

### 1题

图的创建, 找到三台互联 至少有一台以t开头的节点

<details><summary>Details</summary>
<p>

```python

from itertools import combinations

# 读取输入数据
lines = a.splitlines()

# 构建邻接表
connections = {}
for line in lines:
    a, b = line.strip().split('-')
    if a not in connections:
        connections[a] = set()
    if b not in connections:
        connections[b] = set()
    connections[a].add(b)
    connections[b].add(a)

# 获取所有电脑名称
nodes = list(connections.keys())

# 生成所有三个电脑的组合
triplets = combinations(nodes, 3)

# 检查是否完全连接
def is_complete(triplet):
    a, b, c = triplet
    return b in connections[a] and c in connections[a] and c in connections[b]

# 统计符合条件的三元组数量
count = 0
for triplet in triplets:
    if is_complete(triplet):
        if any(name.startswith('t') for name in triplet):
            count += 1

print(f"符合条件的三元组共有 {count} 个。")
``` 

</p>
</details> 

### 2题

最大的一组相互连接的计算机, 按字母排序输出

<details><summary>Details</summary>
<p>

```python
# 读取输入数据
lines = a.splitlines()

# 构建邻接表
connections = {}
for line in lines:
    a, b = line.strip().split('-')
    if a not in connections:
        connections[a] = set()
    if b not in connections:
        connections[b] = set()
    connections[a].add(b)
    connections[b].add(a)

# Bron-Kerbosch算法找所有团
def bron_kerbosch(R, P, X):
    if not P and not X:
        cliques.append(R)
        return
    u = next(iter(P.union(X))) if P.union(X) else None
    if u:
        for v in P - connections[u]:
            bron_kerbosch(R + [v], P.intersection(connections[v]), X.intersection(connections[v]))
            P.remove(v)
            X.add(v)
    else:
        cliques.append(R)

cliques = []
bron_kerbosch([], set(connections.keys()), set())

# 找到最大的团
max_clique = max(cliques, key=lambda x: len(x))

# 生成密码
password = ','.join(sorted(max_clique))

print(password)
``` 

</p>
</details> 

## 第二十四题

[https://adventofcode.com/2024/day/25](https://adventofcode.com/2024/day/25)

### 1题

模拟逻辑门输入输出

<details><summary>Details</summary>
<p>

```python
from collections import deque


def main():
    input_data = a
    parts = input_data.split('\n\n')
    if len(parts) < 2:
        initial_values_section = parts[0]
        gate_definitions_section = ''
    else:
        initial_values_section, gate_definitions_section = parts

    # 解析初始值
    wire_values = {}
    for line in initial_values_section.splitlines():
        if not line.strip():
            continue
        wire, value = line.split(': ')
        wire_values[wire] = int(value)

    # 解析逻辑门定义
    gate_definitions = {}
    for line in gate_definitions_section.splitlines():
        if not line.strip():
            continue
        left, right = line.split(' -> ')
        for op in [' AND ', ' OR ', ' XOR ']:
            if op in left:
                inputs = left.split(op)
                operator = op.strip()
                break
        else:
            continue  # 忽略不支持的操作符
        gate_definitions[right] = (operator, inputs[0], inputs[1])


    # 找到所有输入都有值的逻辑门，加入队列
    queue = deque()
    for output_wire, (op, in1, in2) in gate_definitions.items():
        if in1 in wire_values and in2 in wire_values:
            queue.append(output_wire)

    already_computed = set()
    # 处理队列中的逻辑门
    while queue:
        output_wire = queue.popleft()
        op, in1, in2 = gate_definitions[output_wire]
        value1 = wire_values[in1]
        value2 = wire_values[in2]
        if op == 'AND':
            result = value1 & value2
        elif op == 'OR':
            result = value1 | value2
        elif op == 'XOR':
            result = value1 ^ value2
        else:
            continue  # 无效的操作符
        wire_values[output_wire] = result
        already_computed.add(output_wire)
        # 检查是否有新的逻辑门可以计算
        for gw, (gate_op, gw_in1, gw_in2) in gate_definitions.items():
            if gw not in wire_values and gw_in1 in wire_values and gw_in2 in wire_values:
                if gw not in already_computed and gw not in queue:
                    queue.append(gw)

    # 收集所有以z开头的电线的值
    z_wires = [wire for wire in wire_values if wire.startswith('z')]
    # 按编号排序，z00, z01, ..., z12
    z_wires_sorted = sorted(z_wires, key=lambda x: int(x[1:]))
    # 组合成二进制字符串，从z12到z00
    binary_str = ''.join(str(wire_values[w]) for w in reversed(z_wires_sorted))
    # 转换成十进制
    decimal_output = int(binary_str, 2)
    print(decimal_output)

if __name__ == "__main__":
    main()
``` 

</p>
</details> 

### 2题

> 未完成

## 第二十五题

[https://adventofcode.com/2024/day/25](https://adventofcode.com/2024/day/25)

> 未完成