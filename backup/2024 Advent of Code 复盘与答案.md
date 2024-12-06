[Advent of Code](https://adventofcode.com/)

TODO, 每天更新一题一周前的题目

使用 python 编写, 没有整理代码, 所以非常乱(变量乱取名, 没有注释, 逻辑奇怪)
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