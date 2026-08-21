"""
304辺解収集 v2: 自己同型変換 + remove-and-repair

Q7の自己同型群（ビット置換×ビット反転、位数645,120）を使って
構造的に多様な出発点を生成し、広範な解空間を探索する。
"""

import json
import random
import time
import os
import hashlib
from collections import defaultdict, Counter

n = 7
num_vertices = 1 << n

all_edges = []
for u in range(num_vertices):
    for bit in range(n):
        v = u ^ (1 << bit)
        if u < v:
            all_edges.append((u, v))
all_edges_set = set(all_edges)

edge_to_squares = defaultdict(list)
all_squares = []
for k1 in range(n):
    for k2 in range(k1 + 1, n):
        free_bits = [i for i in range(n) if i not in (k1, k2)]
        for m in range(1 << (n - 2)):
            base = 0
            for j in range(n - 2):
                if m & (1 << j): base |= (1 << free_bits[j])
            v1=base; v2=base^(1<<k1); v3=base^(1<<k1)^(1<<k2); v4=base^(1<<k2)
            sq = tuple(sorted([
                (min(v1,v2),max(v1,v2)), (min(v2,v3),max(v2,v3)),
                (min(v3,v4),max(v3,v4)), (min(v4,v1),max(v4,v1))
            ]))
            all_squares.append(sq)
            for e in sq: edge_to_squares[e].append(sq)

def count_violations(edge_set):
    return sum(1 for sq in all_squares if all(e in edge_set for e in sq))

def delta_add(e, edge_set):
    return sum(1 for sq in edge_to_squares[e]
               if all(x in edge_set for x in sq if x != e))

def greedy_fill(edge_set, target=304):
    current = set(edge_set)
    missing = list(all_edges_set - current)
    random.shuffle(missing)
    for e in missing:
        if len(current) >= target:
            break
        if delta_add(e, current) == 0:
            current.add(e)
    return frozenset(current)

def apply_automorphism(edge_set, perm, flips):
    """Q7の自己同型: ビット置換 perm + ビット反転 flips"""
    result = set()
    for u, v in edge_set:
        def transform(x):
            nx = 0
            for new_bit, old_bit in enumerate(perm):
                if x & (1 << old_bit):
                    nx |= (1 << new_bit)
            return nx ^ flips
        tu, tv = transform(u), transform(v)
        result.add((min(tu, tv), max(tu, tv)))
    return frozenset(result)

def random_automorphism(edge_set):
    """ランダムな自己同型を適用"""
    perm = list(range(n))
    random.shuffle(perm)
    flips = random.randint(0, (1 << n) - 1)
    return apply_automorphism(edge_set, perm, flips)

def edge_set_hash_raw(edge_set):
    """正規化なしのハッシュ（速度優先）"""
    s = ",".join(f"{u},{v}" for u, v in sorted(edge_set))
    return hashlib.md5(s.encode()).hexdigest()[:12]

def canonicalize(edge_set):
    el = list(edge_set)
    dc = [0] * n
    for u, v in el:
        diff = u ^ v
        for b in range(n):
            if diff == 1 << b: dc[b] += 1; break
    perm = sorted(range(n), key=lambda b: -dc[b])
    def remap(v):
        nv = 0
        for nb, ob in enumerate(perm):
            if v & (1 << ob): nv |= (1 << nb)
        return nv
    el2 = frozenset((min(remap(u),remap(v)),max(remap(u),remap(v))) for u,v in el)
    best = el2
    for b in range(n):
        mask = 1 << b
        flipped = frozenset((min(u^mask,v^mask),max(u^mask,v^mask)) for u,v in el2)
        if flipped < best: best = flipped
    return best

def edge_set_hash(edge_set):
    canon = canonicalize(edge_set)
    s = ",".join(f"{u},{v}" for u,v in sorted(canon))
    return hashlib.md5(s.encode()).hexdigest()[:12]

# ==============================
# コレクション読み込み
# ==============================
COLLECTION_FILE = 'solutions_304.jsonl'
found_hashes = set()
all_solutions = []

if os.path.exists(COLLECTION_FILE):
    with open(COLLECTION_FILE) as f:
        for line in f:
            d = json.loads(line)
            es = frozenset(tuple(e) for e in d['edges'])
            found_hashes.add(d['hash'])
            all_solutions.append(es)
    print(f"既存コレクション: {len(all_solutions)}件")

def save_solution(edge_set, h, trial, elapsed, method=''):
    with open(COLLECTION_FILE, 'a') as f:
        rec = {'hash': h, 'trial': trial, 'elapsed': round(elapsed),
               'method': method, 'edges': [list(e) for e in sorted(edge_set)]}
        f.write(json.dumps(rec) + '\n')
        f.flush(); os.fsync(f.fileno())
    all_solutions.append(edge_set)
    found_hashes.add(h)

with open('selected_edges_best.json') as f:
    data = json.load(f)
seed_set = frozenset(tuple(e) for e in data['edges'])
assert data['value'] == 304
print(f"初期解: {data['value']}辺")

h0 = edge_set_hash(seed_set)
if h0 not in found_hashes:
    save_solution(seed_set, h0, 0, 0, 'seed')
    print(f"  初期解追加: {h0}")

# ==============================
# 分析
# ==============================
def analyze():
    if len(all_solutions) < 2: return
    print(f"\n{'='*60}")
    print(f"=== 構造分析 ({len(all_solutions)}件) ===")
    edge_counter = Counter(e for es in all_solutions for e in es)
    total = len(all_solutions)
    always = [e for e,c in edge_counter.items() if c == total]
    never  = [e for e in all_edges if edge_counter[e] == 0]
    often  = [e for e,c in edge_counter.items() if c >= total*0.8 and c < total]
    rare   = [e for e,c in edge_counter.items() if 0 < c <= total*0.2]
    print(f"全解共通辺(必須辺): {len(always)}本")
    print(f"出現なし辺:         {len(never)}本")
    print(f"80%以上出現:        {len(often)}本")
    print(f"20%以下出現:        {len(rare)}本")
    sols = random.sample(list(all_solutions), min(30, len(all_solutions)))
    dists = [1-len(sols[i]&sols[j])/len(sols[i]|sols[j])
             for i in range(len(sols)) for j in range(i+1,len(sols))]
    if dists:
        print(f"ジャッカード距離: 平均{sum(dists)/len(dists):.4f} 最大{max(dists):.4f}")
    # 必須辺からの貪欲充填
    if always:
        always_set = set(always)
        best = 0
        for _ in range(200):
            cur = set(always)
            cands = list(all_edges_set - cur)
            random.shuffle(cands)
            for e in cands:
                if delta_add(e, cur) == 0: cur.add(e)
            best = max(best, len(cur))
        print(f"必須辺({len(always)}本)から貪欲充填最大: {best}辺")
        # 出現なし辺の競合確認
        never_no_conflict = [(114,115)] # 注目辺
        for e in never:
            c = sum(1 for sq in edge_to_squares[e]
                    if sum(1 for x in sq if x in always_set and x != e) == 3)
            if c == 0:
                never_no_conflict.append(e)
        unique_nc = list(set(never_no_conflict))
        print(f"必須辺競合なし出現なし辺: {sorted(unique_nc)}")
    print(f"{'='*60}\n")

# ==============================
# メインループ
# ==============================
start_time = time.time()
trial = 0
RUNTIME = 36 * 3600
ANALYZE_INTERVAL = 50
new_found = 0
auto_count = 0

print(f"\n開始: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"手法: 自己同型変換 + remove-and-repair")
print("-"*60)

while time.time() - start_time < RUNTIME:
    trial += 1
    method = ''

    # 出発点の選択戦略（3種類）
    r = random.random()
    if r < 0.4:
        # 既存解にランダム自己同型を適用（多様性確保）
        base_sol = random.choice(all_solutions) if all_solutions else seed_set
        base = set(random_automorphism(base_sol))
        method = 'auto'
        auto_count += 1
    elif r < 0.7:
        # 既存解から直接 remove-and-repair
        base = set(random.choice(all_solutions) if all_solutions else seed_set)
        method = 'repair'
    else:
        # seed に自己同型を適用
        base = set(random_automorphism(seed_set))
        method = 'auto_seed'
        auto_count += 1

    # k本抜く
    k = random.choices([1,2,3,4,5,8,12,20],
                       weights=[3,8,12,12,12,15,15,10])[0]
    k = min(k, len(base))
    removed = random.sample(list(base), k)
    base -= set(removed)

    # 貪欲充填
    result = greedy_fill(base, target=304)

    if len(result) < 304: continue
    if count_violations(result) > 0: continue

    # 305以上
    if len(result) >= 305:
        elapsed = time.time() - start_time
        print(f"\n  🎉🎉🎉 {len(result)}辺解発見!!! (trial{trial}, {elapsed:.0f}s, {method})")
        with open(f'selected_edges_{len(result)}.txt', 'w') as f:
            for u,v in sorted(result): f.write(f"{u},{v}\n")
            f.flush(); os.fsync(f.fileno())
        with open('selected_edges_best.json', 'w') as f:
            json.dump({'value':len(result),'n':n,
                       'edges':[list(e) for e in sorted(result)]},f)
            f.flush(); os.fsync(f.fileno())
        continue

    # 新しい304辺解か
    h = edge_set_hash(result)
    if h not in found_hashes:
        elapsed = time.time() - start_time
        save_solution(result, h, trial, elapsed, method)
        new_found += 1
        print(f"  ★ trial{trial}: 新解! (合計{len(all_solutions)}件, k={k}, {method}, {elapsed:.0f}s)",
              flush=True)
        if len(all_solutions) % ANALYZE_INTERVAL == 0:
            analyze()

    if trial % 2000 == 0:
        elapsed = time.time() - start_time
        rate = trial / elapsed
        print(f"  [trial{trial}] 収集:{len(all_solutions)}件 "
              f"自己同型:{auto_count}回 速度:{rate:.0f}/s 経過:{elapsed:.0f}s", flush=True)

analyze()
print(f"\n=== 最終結果 ===")
print(f"総試行数: {trial:,}")
print(f"収集: {len(all_solutions)}件 (新規{new_found}件)")
print(f"自己同型適用: {auto_count}回")
print(f"総時間: {time.time()-start_time:.0f}秒")
