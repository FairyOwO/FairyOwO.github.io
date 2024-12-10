[Advent of Code](https://adventofcode.com/)

TODO, 每天更新一题一周前的题目

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