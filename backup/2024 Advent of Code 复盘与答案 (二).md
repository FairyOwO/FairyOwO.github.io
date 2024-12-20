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

## 第十二题

[https://adventofcode.com/2024/day/12](https://adventofcode.com/2024/day/12)

### 1题

划分区域

```text
AAAA
BBCD
BBCC
EEEC
```

划分成

```text
+-+-+-+-+
|A A A A|
+-+-+-+-+     +-+
              |D|
+-+-+   +-+   +-+
|B B|   |C|
+   +   + +-+
|B B|   |C C|
+-+-+   +-+ +
          |C|
+-+-+-+   +-+
|E E E|
+-+-+-+
```

分为五个区域, 
计算每个区域的周长*面积之和

<details><summary>Details</summary>
<p>

```python

aa = []
for i in a.splitlines():
    aa.append(list(i))

def get_perimeter(arr):
    perimeter = 0
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j] == 1:
                perimeter += 4
                if i > 0 and arr[i - 1][j] == 1:
                    perimeter -= 1
                if j > 0 and arr[i][j - 1] == 1:
                    perimeter -= 1
                if i < len(arr) - 1 and arr[i + 1][j] == 1:
                    perimeter -= 1
                if j < len(arr[0]) - 1 and arr[i][j + 1] == 1:
                    perimeter -= 1
    
    return perimeter

def get_area(arr):
    area = 0
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j] == 1:
                area += 1
    return area

def get_allow_pos(pos):
    allow_pos = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    char = aa[pos[0]][pos[1]]
    for i in allow_pos:
        next_pos = (pos[0] + i[0], pos[1] + i[1])
        if len(aa) > next_pos[0] >= 0 and len(aa[0]) > next_pos[1] >= 0:
            if aa[next_pos[0]][next_pos[1]] == char:
                yield next_pos


ans = 0

already_visited = np.zeros((len(aa), len(aa[0])))
for i in range(len(aa)):
    for j in range(len(aa[i])):
        if already_visited[i][j] == 0:
            # print(aa[i][j])
            array_ = np.zeros((len(aa), len(aa[0])))
            already_visited[i][j] = 1
            array_[i][j] = 1
            def dfs(pos):
                for next_pos in get_allow_pos(pos):
                    if already_visited[next_pos[0]][next_pos[1]] == 0:
                        already_visited[next_pos[0]][next_pos[1]] = 1
                        array_[next_pos[0]][next_pos[1]] = 1
                        dfs(next_pos)
            
            dfs((i, j))
            area = get_area(array_)
            perimeter = get_perimeter(array_)
            ans += area * perimeter
        
print(ans)

```

</p>
</details> 

> 加一个正方形边长+4, 如果旁边每有一个正方形就-1的边长

### 2题

1题的边长变成边的数量

<details><summary>Details</summary>
<p>

```python

aa = []
for i in a.splitlines():
    aa.append(list(i))



def get_area(arr):
    area = 0
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j] == 1:
                area += 1
    return area

def get_perimeter(region):
        
    max_x = len(region) - 1
    max_y = len(region[0]) - 1
    min_x = min_y = 0
    
    def state(x, y):
        if x < 0 or x > max_x or y < 0 or y > max_y:
            return False
        return region[x][y]
    
    perimeter = 0
    
    # 垂直方向扫描
    for i in range(max_x + 1):
        st = state(i, -1)
        for j in range(max_y + 2):
            if st != state(i, j):
                if st != state(i-1, j-1) or st == state(i-1, j):
                    perimeter += 1
                if st != state(i+1, j-1) or st == state(i+1, j):
                    perimeter += 1
                st = not st
    
    # 水平方向扫描
    for j in range(max_y + 1):
        st = state(-1, j)
        for i in range(max_x + 2):
            if st != state(i, j):
                if st != state(i-1, j-1) or st == state(i, j-1):
                    perimeter += 1
                if st != state(i-1, j+1) or st == state(i, j+1):
                    perimeter += 1
                st = not st
    
    return perimeter // 2

def get_allow_pos(pos):
    allow_pos = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    char = aa[pos[0]][pos[1]]
    for i in allow_pos:
        next_pos = (pos[0] + i[0], pos[1] + i[1])
        if len(aa) > next_pos[0] >= 0 and len(aa[0]) > next_pos[1] >= 0:
            if aa[next_pos[0]][next_pos[1]] == char:
                yield next_pos

ans = 0

already_visited = np.zeros((len(aa), len(aa[0])))
for i in range(len(aa)):
    for j in range(len(aa[i])):
        if already_visited[i][j] == 0:
            # print(aa[i][j])
            array_ = np.zeros((len(aa), len(aa[0])))
            already_visited[i][j] = 1
            array_[i][j] = 1
            def dfs(pos):
                for next_pos in get_allow_pos(pos):
                    if already_visited[next_pos[0]][next_pos[1]] == 0:
                        already_visited[next_pos[0]][next_pos[1]] = 1
                        array_[next_pos[0]][next_pos[1]] = 1
                        
                        dfs(next_pos)
            
            dfs((i, j))
            area = get_area(array_)
            perimeter = get_perimeter(array_)
            # print(perimeter)
            ans += area * perimeter
        
print(ans)

```

</p>
</details> 

> 一条边有两个角, 找到所有角之后整除2即可

## 第十三题

[https://adventofcode.com/2024/day/13](https://adventofcode.com/2024/day/13)

### 1题

按下按钮A或B, 移动老虎爪, 让老虎爪移动到指定的位置, 按下A需要三块钱, B需要一块钱
有可能无解(无法移动到指定位置)

<details><summary>Details</summary>
<p>

```python

aa = a.split('\n\n')
from sympy import *

ans = 0
for i in aa:
    inputs = i.split('\n')
    # 提取输入
    ax, ay = inputs[0].split(': ')[1].split(', ')
    bx, by = inputs[1].split(': ')[1].split(', ')
    t1, t2 = inputs[2].split(': ')[1].split(', ')
    ax, ay = int(ax[2:]), int(ay[2:])
    bx, by = int(bx[2:]), int(by[2:])
    t1, t2 = int(t1[2:]), int(t2[2:])
    m = Symbol('m')
    n = Symbol('n')
    temp = solve([m * ax + n * bx - t1, m * ay + n * by - t2], [m, n])
    # print(temp)
    if temp[m].is_Integer and temp[n].is_Integer:
        ans += int(temp[m]) * 3 + int(temp[n])

print(ans)

```

</p>
</details> 
> 二元一次方程组的整数解

### 2题

在一题的基础上, X轴和Y轴上都高出10000000000000(大数)

<details><summary>Details</summary>
<p>

```python

aa = a.split('\n\n')
from sympy import *

ans = 0
for i in aa:
    inputs = i.split('\n')
    # 提取输入
    ax, ay = inputs[0].split(': ')[1].split(', ')
    bx, by = inputs[1].split(': ')[1].split(', ')
    t1, t2 = inputs[2].split(': ')[1].split(', ')
    ax, ay = int(ax[2:]), int(ay[2:])
    bx, by = int(bx[2:]), int(by[2:])
    t1, t2 = int(t1[2:]) + 10000000000000, int(t2[2:]) + 10000000000000
    m = Symbol('m')
    n = Symbol('n')
    temp = solve([m * ax + n * bx - t1, m * ay + n * by - t2], [m, n])
    # print(temp)
    if temp[m].is_Integer and temp[n].is_Integer:
        ans += int(temp[m]) * 3 + int(temp[n])

print(ans)
```

</p>
</details> 

> python无限精度整数, 不需要额外处理

## 第十四题

[https://adventofcode.com/2024/day/14](https://adventofcode.com/2024/day/14)

## 1题

给定点坐标, 点的移动速度, 大小固定的图

求 四个象限内点的数量 之积
象限是去掉最中间一列与一行使得图分成四个区域
点到边界会传送到另一侧 (mod)

<details><summary>Details</summary>
<p>

```python

def main():
    lines = a.splitlines()

    robots = []
    for line in lines:
        p_part, v_part = line.strip().split()
        p_x, p_y = map(int, p_part[2:].split(','))
        v_x, v_y = map(int, v_part[2:].split(','))
        robots.append({'p': (p_x, p_y), 'v': (v_x, v_y)})

    positions = {}
    for robot in robots:
        x = (robot['p'][0] + robot['v'][0] * 100) % 101
        y = (robot['p'][1] + robot['v'][1] * 100) % 103
        positions[(x, y)] = positions.get((x, y), 0) + 1

    q1 = q2 = q3 = q4 = 0
    for (x, y), count in positions.items():
        if x < 50 and y < 51:
            q1 += count
        elif x > 50 and y < 51:
            q2 += count
        elif x > 50 and y > 51:
            q3 += count
        elif x < 50 and y > 51:
            q4 += count

    safety_factor = q1 * q2 * q3 * q4
    print(safety_factor)

if __name__ == "__main__":
    main()

```

</p>
</details> 

### 2题

直接贴原题, 因为原题是阅读理解

During the bathroom break, someone notices that these robots seem awfully similar to ones built and used at the North Pole. If they're the same type of robots, they should have a hard-coded Easter egg: very rarely, most of the robots should arrange themselves into a picture of a Christmas tree.
> 在上厕所的时候，有人注意到这些机器人看起来与在北极建造和使用的机器人非常相似。如果它们是同一类型的机器人，它们应该有一个硬编码的复活节彩蛋：极少数情况下，大多数机器人应该将自己排列成圣诞树的图片。

What is the fewest number of seconds that must elapse for the robots to display the Easter egg?
> 机器人展示复活节彩蛋所需的最少秒数是多少？

<details><summary>Details</summary>
<p>

```python
def find_min_unique_t(input_lines):
    robots = []
    for line in input_lines:
        p_part, v_part = line.split(' v=')
        x, y = map(int, p_part[2:].split(','))
        dx, dy = map(int, v_part.split(','))
        robots.append({'x': x, 'y': y, 'dx': dx, 'dy': dy})
    
    for t in range(10403):
        position_map = {}
        for robot in robots:
            x = (robot['x'] + robot['dx'] * t) % 101
            y = (robot['y'] + robot['dy'] * t) % 103
            position = (x, y)
            if position in position_map:
                break
            else:
                position_map[position] = True
        else:
            return t
    return -1

input_lines = a.splitlines()

min_t = find_min_unique_t(input_lines)
print("最小的秒数是:", min_t)
```

</p>
</details> 

> 这题等价于, 每个机器人不互相重合的时间(题目特意设计过)
> 如果有其他解法(理解)可以在评论区交流

## 第十五题

[https://adventofcode.com/2024/day/15](https://adventofcode.com/2024/day/15)

### 1题

推箱子

<details><summary>Details</summary>
<p>

aa, bb = a.split('\n\n')

all_map = [list(i) for i in aa.splitlines()]

step = ''.join(bb.splitlines())

def is_edge(x, y):
    return x < 0 or y < 0 or x >= len(all_map) or y >= len(all_map[0]) or all_map[x][y] == '#'

def get_robot_pos():
    for i in range(len(all_map)):
        for j in range(len(all_map[0])):
            if all_map[i][j] == '@':
                return i, j

directions = {'>': (0,1), 'v': (1,0), '<': (0,-1), '^': (-1,0)}

def move(pos, move_to):
    temp = []
    x, y = pos
    dx, dy = move_to
    for i in range(1, max(len(all_map), len(all_map[0]))):

        temp.append(all_map[x][y])

        x += dx
        y += dy
        if not is_edge(x, y):
            if all_map[x][y] == '.':
                # 当前位置开始向前一步
                for _ in range(i):
                    all_map[x][y] = temp.pop()
                    x -= dx
                    y -= dy
                all_map[pos[0]][pos[1]] = '.'
                return (pos[0] + dx, pos[1] + dy)
        else:
            return pos
    return pos

def putty_print(all_map):
    for i in all_map:
        for j in i:
            print(j, end='')
        print()



now_pos = get_robot_pos()
for i in step:
    now_pos = move(now_pos, directions[i])
putty_print(all_map)



ans = 0
for i in range(len(all_map)):
    for j in range(len(all_map[i])):
        if all_map[i][j] == 'O':
            ans += i * 100 + j 

print(ans)

</p>
</details> 

## 2题

推更宽(宽一倍), 但是高度不变的箱子

<details><summary>Details</summary>
<p>


aa, bb = a.split('\n\n')

all_map = [list(i) for i in aa.splitlines()]

new_map = []
for i in all_map:
    temp_map = []
    for j in i:
        if j == '#':
            temp_map.append('#')
            temp_map.append('#')
        if j == 'O':
            temp_map.append('[')
            temp_map.append(']')
        if j == '.':
            temp_map.append('.')
            temp_map.append('.')
        if j == '@':
            temp_map.append('@')
            temp_map.append('.')
    new_map.append(temp_map)

all_map = new_map
moves = ''.join(bb.splitlines())
def is_edge(x, y):
    return x < 0 or y < 0 or x >= len(all_map) or y >= len(all_map[0]) or all_map[x][y] == '#'

def get_robot_pos():
    for i in range(len(all_map)):
        for j in range(len(all_map[0])):
            if all_map[i][j] == '@':
                return i, j

directions = {'>': (0,1), 'v': (1,0), '<': (0,-1), '^': (-1,0)}

all_boxes = []

def move(pos, move_to):
    temp = []
    x, y = pos
    dx, dy = move_to
    if dx == 0:
        for i in range(1, max(len(all_map), len(all_map[0]))):

            temp.append(all_map[x][y])

            x += dx
            y += dy
            if not is_edge(x, y):
                if all_map[x][y] == '.':
                    # 当前位置开始向前一步
                    for _ in range(i):
                        all_map[x][y] = temp.pop()
                        x -= dx
                        y -= dy
                    all_map[pos[0]][pos[1]] = '.'
                    return (pos[0] + dx, pos[1] + dy)
            else:
                return pos
        return pos
    else:
        box_pos = []
        next_pos = [(x, y)]
        can_move = True

        while next_pos:
            x, y = next_pos.pop(0)
            x, y = x + dx, y + dy
            if all_map[x][y] == '[':
                box_pos.append(((x, y), (x, y+1)))
                next_pos.append((x, y+1))
                next_pos.append((x, y))
            elif all_map[x][y] == ']':
                box_pos.append(((x, y), (x, y-1)))
                next_pos.append((x, y-1))
                next_pos.append((x, y))
            elif all_map[x][y] == '#':
                can_move = False
                break
            # 去重
            tt = []
            [tt.append(i) for i in next_pos if i not in tt]
            next_pos = tt
        tt = []
        [tt.append(i) for i in box_pos if (i not in tt) and ((i[1], i[0]) not in tt)]
        box_pos = tt
        if not can_move:
            return pos
        else:
            for box in box_pos[::-1]:
                for box_pos in box:
                    all_map[box_pos[0] + dx][box_pos[1] + dy] = all_map[box_pos[0]][box_pos[1]]
                    all_map[box_pos[0]][box_pos[1]] = '.'

            # 更改@的位置
            all_map[pos[0] + dx][pos[1] + dy] = '@'
            all_map[pos[0]][pos[1]] = '.'
            return (pos[0] + dx, pos[1] + dy)




def putty_print(all_map):
    for i in all_map:
        for j in i:
            print(j, end='')
        print()



now_pos = get_robot_pos()
for i in moves:
    now_pos = move(now_pos, directions[i])
putty_print(all_map)



ans = 0
for i in range(len(all_map)):
    for j in range(len(all_map[i])):
        if all_map[i][j] == '[':
            ans += i * 100 + j 

print(ans)

</p>
</details> 

> 横着推可以复用一题代码, 竖着推需要检测所有能推的箱子(bfs)
