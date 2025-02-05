## 相关原理

详见[这里](https://paldb.cc/cn/Breeding_Farm#BreedingFarm)

简单来说, 子帕鲁是两个父帕鲁繁殖力与1的和整除2, 特殊配方需要按照专有配方才可以繁殖

## 准备数据

你需要从[帕鲁编年史](https://paldb.cc)中, 自行整理 `a.tsv` (所有帕鲁的CombiRank 繁殖力 IndexOrder 索引), `b.tsv` (特殊繁殖配方)
> 此索引与帕鲁id不同
<details><summary>两个表的格式示例</summary>
<p>

## a.tsv

[帕鲁编年史 Breed Combi](https://paldb.cc/cn/Breeding_Farm)

```tsv
name	CombiRank	IndexOrder
皮皮鸡	1500	66
壶小象	1490	17
喵丝特	1480	7
```

> 注意中间是 `\t` 制表符

## b.tsv

[帕鲁编年史 Breed Unique](https://paldb.cc/cn/Breeding_Farm#BreedUnique)
[帕鲁编年史 Breed Self](https://paldb.cc/cn/Breeding_Farm#BreedSelf)


```tsv
parent	孵化完成
佩克龙  伏特喵	派克龙
炎魔羊  噬魂兽	暗魔羊
喵丝特  企丸丸	冰丝特
```
> 注意中间是 `\t` 制表符, parent中间是两个空格

</p>
</details> 

> 注意语言

## 代码

<details><summary>未整理代码</summary>
<p>

```python
from collections import deque
import csv

def load_data():
    data = {}
    with open(r'a.tsv', 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            data[row[0]] = {"fertility": int(row[1]), "index_order": int(row[2])}

    data2 = {}
    exclusive = set()

    with open(r'b.tsv', 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            parents = tuple(sorted(row[0].split('  ')))
            child = row[1]
            data2[parents] = child
            exclusive.add(child)

    return data, data2, exclusive

pal_data, special_breeding, exclusive_pals = load_data()

def breed(parent1, parent2):
    key = tuple(sorted([parent1, parent2]))
    if key in special_breeding:
        return special_breeding[key]
    
    # 塔主处理
    p1_fertility = pal_data[parent1]["fertility"]
    p2_fertility = pal_data[parent2]["fertility"]
    if p1_fertility == 9999 or p2_fertility == 9999:
        return "皮皮鸡"
    
    value = (p1_fertility + p2_fertility + 1) // 2
    
    valid_pals = [
        (name, info) for name, info in pal_data.items() 
        if name not in exclusive_pals
    ]
    
    closest_pal = min(
        valid_pals,
        key=lambda x: (abs(x[1]["fertility"] - value), x[1]["index_order"])
    )
    return closest_pal[0]

def find_breeding_path(start, end):
    if start not in pal_data or end not in pal_data:
        return []
    
    target_fertility = pal_data[end]["fertility"]
    visited = {}
    queue = deque([(start, [start])])
    results = []
    min_steps = None

    while queue:
        level_size = len(queue)
        found = False
        
        for _ in range(level_size):
            current_pal, path = queue.popleft()
            
            if min_steps and len(path) > min_steps:
                continue
                
            if current_pal == end:
                if not min_steps:
                    min_steps = len(path)
                if len(path) == min_steps:
                    results.append(path)
                found = True
                continue
                
            # 获取当前帕鲁繁殖力
            current_fertility = pal_data[current_pal]["fertility"]
            
            for mate in pal_data:
                if mate == current_pal:
                    continue
                
                key = tuple(sorted([current_pal, mate]))
                
                # 特殊繁殖直接处理
                if key in special_breeding:
                    child = special_breeding[key]
                    new_path = path + [mate, child]
                    if child not in visited or len(new_path) < visited.get(child, float('inf')):
                        visited[child] = len(new_path)
                        queue.append((child, new_path))
                    continue
                
                mate_fertility = pal_data[mate]["fertility"]
                
                # 塔主特殊处理
                if current_fertility == 9999 or mate_fertility == 9999:
                    child = "皮皮鸡"
                else:
                    if current_fertility < target_fertility and mate_fertility < current_fertility:
                        continue  # 配偶生育力更低，子代只会更小
                    if current_fertility > target_fertility and mate_fertility > current_fertility:
                        continue  # 配偶生育力更高，子代只会更大
                    
                    child = breed(current_pal, mate)
                
                new_path = path + [mate, child]
                
                # 如果子代生育力偏离目标方向
                child_fertility = pal_data[child]["fertility"]
                if (target_fertility > current_fertility and child_fertility < current_fertility) or \
                   (target_fertility < current_fertility and child_fertility > current_fertility):
                    continue
                
                if child not in visited or len(new_path) < visited[child]:
                    visited[child] = len(new_path)
                    queue.append((child, new_path))
                elif len(new_path) == visited[child]:
                    queue.append((child, new_path))
        
        if found:
            break

    final_results = [p for p in results if len(p) == min_steps]
    return final_results if final_results else []

start_pal = "混沌骑士"
end_pal = "八云犬"
paths = find_breeding_path(start_pal, end_pal)

if paths:
    print(f"找到{len(paths)}条最短繁殖路径（步数：{(len(paths[0])-1)//2}）:")
    for i, path in enumerate(paths, 1):
        print(f"\n路径{i}:")
        for j in range(0, len(path)-1, 2):
            print(f"{path[j]} + {path[j+1]} → {path[j+2]}")
else:
    print("无可行繁殖路径")

```

</p>
</details> 

> 不知道为什么, 相对于帕鲁编年史的实现, 我的实现慢一点, 估计是他预计算好了