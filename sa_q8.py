"""
sa_q8.py  ―  ex(Q8, C4) 下界更新

Q8: 256頂点, 1024辺, C4数=1792
Brass公式下限 (n=8, 0.9√n版):
  f(8) >= 1/2 * (8 + 0.9*√8) * 2^7 ≈ 675辺

目標: 675辺超を確認し、さらに上を狙う

戦略:
  Phase1: Penalty SA (C4違反を一時許容して辺数を最大化)
  Phase2: Swap-SA (辺数固定・違反ゼロ化)
  
  Q7で効いたパラメータを Q8 にスケールアップ。
  辺数1024は448の2.3倍 → 温度・ステップ数を調整。
"""

import json, random, time, os, math
from collections import defaultdict, Counter

# ==============================
# Q8 グラフ構造
# ==============================
n = 8
num_vertices = 1 << n   # 256

all_edges = []
for u in range(num_vertices):
    for bit in range(n):
        v = u ^ (1 << bit)
        if u < v:
            all_edges.append((u, v))
N_EDGES = len(all_edges)  # 1024

edge_to_squares = defaultdict(list)
all_squares = []
for k1 in range(n):
    for k2 in range(k1 + 1, n):
        free_bits = [i for i in range(n) if i not in (k1, k2)]
        for m in range(1 << (n - 2)):
            base = 0
            for j in range(n - 2):
                if m & (1 << j):
                    base |= (1 << free_bits[j])
            v1 = base; v2 = base ^ (1 << k1)
            v3 = base ^ (1 << k1) ^ (1 << k2); v4 = base ^ (1 << k2)
            sq = tuple(sorted([
                (min(v1,v2),max(v1,v2)), (min(v2,v3),max(v2,v3)),
                (min(v3,v4),max(v3,v4)), (min(v4,v1),max(v4,v1))
            ]))
            all_squares.append(sq)
            for e in sq:
                edge_to_squares[e].append(sq)

print(f"Q{n}: {num_vertices}頂点, {N_EDGES}辺, C4数={len(all_squares)}")

# Brass公式下限
import math as _math
brass_lb = 0.5 * (n + 0.9 * _math.sqrt(n)) * (2 ** (n - 1))
print(f"Brass公式下限: {brass_lb:.1f}辺")
print(f"目標: {int(brass_lb) + 1}辺以上")

# ==============================
# 基本関数
# ==============================
def delta_v(e, edge_set):
    return sum(1 for sq in edge_to_squares[e]
               if all(x in edge_set for x in sq if x != e))

def count_violations(edge_set):
    return sum(1 for sq in all_squares if all(e in edge_set for e in sq))

def save_result(edge_set, value):
    fname = f'q8_edges_{value}.txt'
    with open(fname, 'w') as f:
        for u, v in sorted(edge_set):
            f.write(f"{u},{v}\n")
        f.flush(); os.fsync(f.fileno())
    with open('q8_best.json', 'w') as f:
        json.dump({'value': value, 'n': n,
                   'edges': [list(e) for e in sorted(edge_set)]}, f)
        f.flush(); os.fsync(f.fileno())
    print(f"  💾 保存: {fname}")

# ==============================
# 初期解: 既存ファイルがあれば読み込む
# ==============================
RUNTIME = 36 * 3600
start_time = time.time()

best_valid_count = 0
best_valid_sol = None

if os.path.exists('q8_best.json'):
    with open('q8_best.json') as f:
        data = json.load(f)
    best_valid_sol = frozenset(tuple(e) for e in data['edges'])
    best_valid_count = data['value']
    print(f"既存最良解: {best_valid_count}辺")
else:
    # 初回: 貪欲構築で初期解を作る
    print("初期解を貪欲構築中...")
    shuffled = list(all_edges)
    random.shuffle(shuffled)
    greedy = set()
    for e in shuffled:
        if delta_v(e, greedy) == 0:
            greedy.add(e)
    best_valid_count = len(greedy)
    best_valid_sol = frozenset(greedy)
    print(f"貪欲初期解: {best_valid_count}辺")
    save_result(best_valid_sol, best_valid_count)

stats = {
    'trial': 0,
    'phase2_attempts': 0,
    'max_edges_seen': best_valid_count,
    'min_viol_at_target': float('inf'),
}

TARGET = int(brass_lb) + 1  # まずここを超える
print(f"\n開始: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Q8 C4フリー下界探索 | 目標: {TARGET}辺以上")
print("=" * 60)

# ==============================
# メインループ
# ==============================
while time.time() - start_time < RUNTIME:
    stats['trial'] += 1
    elapsed = time.time() - start_time

    # ハイパーパラメータ (Q8スケール)
    lam      = random.uniform(0.35, 0.85)
    T_start  = random.uniform(0.30, 3.00)
    T_end    = random.uniform(0.002, 0.025)
    # Q8は辺数2.3倍 → ステップ数も比例して増やす
    steps_p1 = random.randint(2_000_000, 12_000_000)
    max_viol = random.randint(8, 35)

    # ==============================
    # Phase 1: Penalty SA
    # ==============================
    current = set(best_valid_sol)
    n_e = len(current); n_v = 0
    energy = float(-n_e)
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / steps_p1)

    p1_best_e = n_e; p1_best_v = n_v
    p1_best_sol = frozenset(current)

    for step in range(steps_p1):
        e = all_edges[random.randrange(N_EDGES)]
        dv = delta_v(e, current)

        if e in current:
            new_ne, new_nv = n_e - 1, n_v - dv
        else:
            new_ne, new_nv = n_e + 1, n_v + dv

        if new_nv > max_viol:
            T *= cooling; continue

        new_energy = -new_ne + lam * new_nv
        de = new_energy - energy

        if de < 0 or random.random() < math.exp(max(-30.0, -de / T)):
            if e in current: current.remove(e)
            else: current.add(e)
            n_e, n_v, energy = new_ne, new_nv, new_energy

            # 有効解で最良更新
            if n_v == 0 and n_e > best_valid_count:
                best_valid_count = n_e
                best_valid_sol = frozenset(current)
                print(f"\n  🎉 {n_e}辺達成! (λ={lam:.2f} T0={T_start:.2f} "
                      f"step={step} {elapsed:.0f}s)")
                save_result(best_valid_sol, n_e)
                TARGET = max(TARGET, n_e + 1)

            # Phase1ベスト更新
            if n_e > p1_best_e or (n_e == p1_best_e and n_v < p1_best_v):
                p1_best_e = n_e; p1_best_v = n_v
                p1_best_sol = frozenset(current)

            if n_e > stats['max_edges_seen']:
                stats['max_edges_seen'] = n_e

            # 目標辺数での最小違反追跡
            if n_e >= TARGET and n_v < stats['min_viol_at_target']:
                stats['min_viol_at_target'] = n_v
                print(f"  📊 {TARGET}辺での最小違反更新: {n_v} "
                      f"(λ={lam:.2f} step={step})", flush=True)

        T *= cooling

    # 進捗表示
    if stats['trial'] % 20 == 0:
        print(f"  [trial{stats['trial']}] "
              f"P1:{p1_best_e}辺/{p1_best_v}違反 "
              f"max:{stats['max_edges_seen']}辺 "
              f"目標{TARGET}辺最小違反:{stats['min_viol_at_target']} "
              f"経過:{elapsed:.0f}s", flush=True)

    # Phase2 に渡す条件: 目標辺数以上
    if p1_best_e < TARGET:
        continue

    # ==============================
    # Phase 2: 目標辺数固定 Swap-SA
    # ==============================
    stats['phase2_attempts'] += 1

    # 目標辺数に整理
    repair_sol = set(p1_best_sol)
    while len(repair_sol) > TARGET:
        worst = max(repair_sol, key=lambda e: delta_v(e, repair_sol))
        repair_sol.remove(worst)
    if len(repair_sol) < TARGET:
        for e in random.sample(all_edges, N_EDGES):
            if len(repair_sol) >= TARGET: break
            if e not in repair_sol and delta_v(e, repair_sol) == 0:
                repair_sol.add(e)

    if len(repair_sol) < TARGET:
        continue

    repair_v = count_violations(repair_sol)
    if repair_v == 0:
        if len(repair_sol) > best_valid_count:
            best_valid_count = len(repair_sol)
            best_valid_sol = frozenset(repair_sol)
            save_result(best_valid_sol, best_valid_count)
        continue

    # Swap-SA: 違反最小化
    edges_in  = list(repair_sol)
    edges_out = [e for e in all_edges if e not in repair_sol]
    n_out = len(edges_out)
    assert len(edges_in) == TARGET

    # Q8: 低違反なら低温、高違反なら高温
    T2     = max(0.08, 0.12 * repair_v)
    T2_end = 0.0002
    steps_p2 = max(3_000_000, 1_000_000 * min(repair_v, 15))
    cool2  = (T2_end / T2) ** (1.0 / steps_p2)

    min_v_p2   = repair_v
    best_in_p2 = frozenset(repair_sol)

    for step2 in range(steps_p2):
        idx_rm  = random.randrange(TARGET)
        idx_add = random.randrange(n_out)
        e_rm    = edges_in[idx_rm]
        e_add   = edges_out[idx_add]

        dv_rm = delta_v(e_rm, repair_sol)
        repair_sol.discard(e_rm)
        dv_add = delta_v(e_add, repair_sol)
        repair_sol.add(e_rm)

        new_v = repair_v - dv_rm + dv_add
        d = new_v - repair_v

        if d < 0 or (T2 > 1e-12 and random.random() < math.exp(max(-30.0, -d / T2))):
            repair_sol.discard(e_rm)
            repair_sol.add(e_add)
            edges_in[idx_rm]   = e_add
            edges_out[idx_add] = e_rm
            repair_v = new_v

            if repair_v < min_v_p2:
                min_v_p2   = repair_v
                best_in_p2 = frozenset(repair_sol)

                if repair_v == 0:
                    elapsed2 = time.time() - start_time
                    print(f"\n  🎉🎉🎉 {TARGET}辺・違反ゼロ達成!! ({elapsed2:.0f}s)")
                    best_valid_count = TARGET
                    best_valid_sol   = best_in_p2
                    save_result(best_valid_sol, TARGET)
                    TARGET += 1
                    stats['min_viol_at_target'] = float('inf')
                    break

        T2 *= cool2

    print(f"  ▶ P2完了: {TARGET-1}辺 P2最小違反={min_v_p2} "
          f"(trial={stats['trial']})", flush=True)

# ==============================
# 最終結果
# ==============================
print(f"\n=== 最終結果 ===")
print(f"ex(Q8, C4) >= {best_valid_count}辺")
print(f"Brass公式下限: {brass_lb:.1f}辺")
print(f"更新量: +{best_valid_count - int(brass_lb)}辺")
print(f"総試行数: {stats['trial']}")
print(f"Phase2回数: {stats['phase2_attempts']}")
print(f"総時間: {time.time()-start_time:.0f}秒")
