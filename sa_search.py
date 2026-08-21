"""
改良版シミュレーテッドアニーリング for ex(Q7, C4)

主な改良点:
1. missing辺をsetで管理 → get_missing() の O(n) コストを撤廃
2. キック操作を追加 → 局所最適を脱出するための大きな摂動
3. 2-opt近傍 (remove2 + add3) → 1-swap より広い近傍
4. 温度スケジュールを高温スタートに変更 → 広域探索優先
5. 違反ペナルティを統一 → ロジックの一貫性
"""

import json
import random
import math
import time
import os
from collections import defaultdict

n = 7
num_vertices = 1 << n

all_edges = []
for u in range(num_vertices):
    for bit in range(n):
        v = u ^ (1 << bit)
        if u < v:
            all_edges.append((u, v))
all_edges_set = set(all_edges)
all_edges_list = all_edges  # リストも保持

# 各辺が属するC4面を事前計算
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
            v1 = base; v2 = base^(1<<k1); v3 = base^(1<<k1)^(1<<k2); v4 = base^(1<<k2)
            sq = tuple(sorted([
                (min(v1,v2),max(v1,v2)), (min(v2,v3),max(v2,v3)),
                (min(v3,v4),max(v3,v4)), (min(v4,v1),max(v4,v1))
            ]))
            all_squares.append(sq)
            for e in sq:
                edge_to_squares[e].append(sq)

all_squares_set = set(all_squares)

def safe_exp(x):
    if x >= 700: return float("inf")
    if x <= -700: return 0.0
    return math.exp(x)

def count_violations(edge_set):
    return sum(1 for sq in all_squares if all(e in edge_set for e in sq))

def delta_add(e, edge_set):
    return sum(1 for sq in edge_to_squares[e]
               if all(x in edge_set for x in sq if x != e))

def delta_remove(e, edge_set):
    return sum(1 for sq in edge_to_squares[e]
               if all(x in edge_set for x in sq if x != e))

# ==============================
# 初期解ロード
# ==============================
# NOTE (added retrospectively): this load is unconditional -- no
# try/except -- so running this script with no selected_edges_best.json
# present in the working directory raises FileNotFoundError
# immediately. There is no random/greedy fallback: this script
# searches for improvements starting from an existing solution, not a
# from-scratch discovery mechanism. See the accompanying manuscript's
# Computational Method section for what this means for provenance.
best_file = 'selected_edges_best.json'
with open(best_file) as f:
    data = json.load(f)
base_set = set(tuple(e) for e in data['edges'])
best_count = data['value']
best_set = base_set.copy()
print(f"初期解: {best_count}辺")

def save_best(edge_set, value):
    with open(best_file, 'w') as f:
        json.dump({'value': value, 'n': n,
                   'edges': [list(e) for e in sorted(edge_set)]}, f)
    with open(f'selected_edges_{value}.txt', 'w') as f:
        for u, v in sorted(edge_set):
            f.write(f"{u},{v}\n")

start_time = time.time()
total_iter = 0
run = 0

# ==============================
# SAメインループ
# ==============================
while time.time() - start_time < 3600:  # 1時間
    run += 1

    # 温度スケジュール: 最初は広域探索、後半は精密探索
    if run <= 5:
        T_start, T_end = 2.0, 0.01    # 高温: 大きく崩して探索
        max_iter = 800_000
    elif run <= 15:
        T_start, T_end = 0.5, 0.002   # 中温: バランス
        max_iter = 600_000
    else:
        T_start, T_end = 0.15, 0.0005 # 低温: 精密探索
        max_iter = 500_000

    # ── 改良①: missing辺をsetで管理 ──────────────────────────────
    current_set = best_set.copy()
    current_missing = all_edges_set - current_set
    current_v = count_violations(current_set)
    local_best = len(current_set)

    # missing/currentをリスト化（ランダムサンプリング用）
    # ※毎回作ると遅いので、一定イテレーションごとに再構築
    REBUILD_INTERVAL = 5000
    cur_list = list(current_set)
    mis_list = list(current_missing)

    for it in range(max_iter):
        total_iter += 1
        T = T_start * (T_end / T_start) ** (it / max_iter)

        # リストを定期再構築（追加/削除で徐々にずれるため）
        if it % REBUILD_INTERVAL == 0:
            cur_list = list(current_set)
            mis_list = list(current_missing)

        r = random.random()

        # ── 改良②: キック操作 (run初期に高確率で実行) ───────────────
        # 局所最適から脱出するため、k辺削除 + k辺追加の大きな摂動
        if r < 0.05 and it < max_iter // 4:
            k = random.randint(3, 6)
            if len(cur_list) >= k and len(mis_list) >= k:
                rem_edges = random.sample(cur_list, k)
                temp_set = current_set - set(rem_edges)
                temp_mis = current_missing | set(rem_edges)
                # 違反が増えない辺をk本追加
                added = []
                temp_mis_list = list(temp_mis)
                random.shuffle(temp_mis_list)
                for e in temp_mis_list:
                    if delta_add(e, temp_set) == 0:
                        temp_set.add(e)
                        added.append(e)
                        if len(added) >= k:
                            break
                new_v = count_violations(temp_set)
                new_size = len(temp_set)
                if new_v == 0 and new_size >= len(current_set) - 1:
                    current_set = temp_set
                    current_missing = all_edges_set - current_set
                    current_v = 0
                    cur_list = list(current_set)
                    mis_list = list(current_missing)
            continue

        # ── 改良③: 2-opt近傍 (remove2 + add3) ──────────────────────
        elif r < 0.15:
            if len(cur_list) < 2 or len(mis_list) < 3:
                pass
            else:
                e1, e2 = random.sample(cur_list, 2)
                temp = current_set - {e1, e2}
                temp_mis = current_missing | {e1, e2}
                # 追加できる辺を3本探す
                candidates = [e for e in random.sample(
                    list(temp_mis), min(30, len(temp_mis)))
                    if delta_add(e, temp) == 0]
                if len(candidates) >= 3:
                    add3 = random.sample(candidates, 3)
                    new_set = temp | set(add3)
                    new_v = count_violations(new_set)
                    delta = len(new_set) - len(current_set)  # +1
                    if new_v == 0 and delta >= 0:
                        current_set = new_set
                        current_missing = all_edges_set - current_set
                        current_v = 0
                        cur_list = list(current_set)
                        mis_list = list(current_missing)
                    elif new_v <= 1 and random.random() < safe_exp(-(new_v * 5) / T):
                        current_set = new_set
                        current_missing = all_edges_set - current_set
                        current_v = new_v
                        cur_list = list(current_set)
                        mis_list = list(current_missing)

        # ── 既存: 追加 ───────────────────────────────────────────────
        elif r < 0.45:
            if not mis_list:
                continue
            e = random.choice(mis_list)
            dv = delta_add(e, current_set)
            new_v = current_v + dv
            if new_v == 0:
                current_set.add(e)
                current_missing.discard(e)
                current_v = 0
            elif new_v <= 2 and random.random() < safe_exp(-new_v * 3 / T):
                current_set.add(e)
                current_missing.discard(e)
                current_v = new_v

        # ── 既存: 削除 ───────────────────────────────────────────────
        elif r < 0.65:
            if not cur_list:
                continue
            if current_v > 0:
                violating = []
                for sq in all_squares:
                    if all(e in current_set for e in sq):
                        violating.extend(sq)
                e = random.choice(violating) if violating else random.choice(cur_list)
            else:
                e = random.choice(cur_list)
            dv = -delta_remove(e, current_set)
            new_v = current_v + dv
            if new_v <= current_v or random.random() < safe_exp(-(new_v - current_v) / T):
                current_set.discard(e)
                current_missing.add(e)
                current_v = new_v

        # ── 既存: 1-swap ─────────────────────────────────────────────
        else:
            if not cur_list or not mis_list:
                continue
            e_rem = random.choice(cur_list)
            e_add = random.choice(mis_list)
            temp = current_set - {e_rem}
            dv_rem = -delta_remove(e_rem, current_set)
            dv_add = delta_add(e_add, temp)
            new_v = current_v + dv_rem + dv_add
            if new_v <= current_v or random.random() < safe_exp(-(new_v - current_v) * 2 / T):
                current_set = temp
                current_set.add(e_add)
                current_missing = (current_missing | {e_rem}) - {e_add}
                current_v = new_v

        # ── 改善チェック ─────────────────────────────────────────────
        if current_v == 0 and len(current_set) > best_count:
            best_count = len(current_set)
            best_set = current_set.copy()
            elapsed = time.time() - start_time
            print(f"  ★ run{run} iter{it:,}: {best_count}辺! ({elapsed:.1f}s)", flush=True)
            save_best(best_set, best_count)

        if current_v == 0 and len(current_set) > local_best:
            local_best = len(current_set)

    elapsed = time.time() - start_time
    print(f"  run{run}完了: ローカル最高{local_best}辺, 全体最高{best_count}辺, {elapsed:.1f}s",
          flush=True)

print(f"\n=== 最終結果 ===")
print(f"最良解: {best_count}辺")
print(f"総イテレーション: {total_iter:,}")
print(f"総時間: {time.time()-start_time:.1f}秒")
print(f"C4違反数: {count_violations(best_set)}")
