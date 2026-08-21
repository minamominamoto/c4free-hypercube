import highspy
import json
import os
import numpy as np

# パラメータ定義
n = 7
num_vertices = 1 << n

# 辺生成
edges = []
edge_dict = {}
for u in range(num_vertices):
    for bit in range(n):
        v = u ^ (1 << bit)
        if u < v:
            edges.append((u, v))
            edge_dict[(u, v)] = len(edges) - 1

num_edges = len(edges)
print(f"頂点数: {num_vertices}, 辺数: {num_edges}")

# HiGHSモデル
h = highspy.Highs()
h.setOptionValue('time_limit', 7200)
h.setOptionValue('presolve', 'on')
h.setOptionValue('mip_max_nodes', 10000000)

# 変数追加
for i in range(num_edges):
    h.addVar(0, 1)

integrality = [highspy.HighsVarType.kInteger] * num_edges
indices = list(range(num_edges))
h.changeColsIntegrality(num_edges, indices, integrality)

cost = [-1.0] * num_edges
h.changeColsCost(num_edges, indices, cost)

# C4制約
num_constraints = 0
for k1 in range(n):
    for k2 in range(k1 + 1, n):
        free_bits = [i for i in range(n) if i not in (k1, k2)]
        for m in range(1 << (n - 2)):
            base = 0
            for j in range(n - 2):
                if m & (1 << j):
                    base |= (1 << free_bits[j])
            v1 = base
            v2 = base ^ (1 << k1)
            v3 = base ^ (1 << k1) ^ (1 << k2)
            v4 = base ^ (1 << k2)
            e1 = (min(v1, v2), max(v1, v2))
            e2 = (min(v2, v3), max(v2, v3))
            e3 = (min(v3, v4), max(v3, v4))
            e4 = (min(v4, v1), max(v4, v1))
            idx = [edge_dict[e1], edge_dict[e2], edge_dict[e3], edge_dict[e4]]
            coeff = [1.0] * 4
            h.addRow(-highspy.kHighsInf, 3, 4, idx, coeff)
            num_constraints += 1

print(f"制約数: {num_constraints}")

# ウォームスタート (修正版: HighsSolutionオブジェクトを使用)
warm_start_file = 'selected_edges_best.json'
if os.path.exists(warm_start_file):
    with open(warm_start_file, 'r') as f:
        saved = json.load(f)
    saved_edges = set(tuple(e) for e in saved['edges'])
    sol_values = [1.0 if edges[i] in saved_edges else 0.0 for i in range(num_edges)]
    print(f"ウォームスタート: {len(saved_edges)}辺の解を読み込み (value={saved['value']})")
    try:
        sol = highspy.HighsSolution()
        sol.col_value = sol_values
        sol.value_valid = True
        status = h.setSolution(sol)
        print(f"ウォームスタート設定完了 (status={status})")
    except Exception as e:
        print(f"ウォームスタート設定失敗 (無視): {e}")

# 解決
print("最適化開始...")
h.run()

model_status = h.getModelStatus()
print(f"ステータス: {model_status}")

if model_status in [highspy.HighsModelStatus.kOptimal,
                     highspy.HighsModelStatus.kObjectiveBound,
                     highspy.HighsModelStatus.kTimeLimit]:
    solution = h.getSolution()
    obj_val = h.getObjectiveValue()
    max_edges_count = round(-obj_val)

    info = h.getInfoValue('mip_dual_bound')
    dual_bound = -info[1] if info[0] == highspy.HighsStatus.kOk else float('nan')

    print(f"\n=== 結果 ===")
    print(f"下界 (最良整数解): {max_edges_count}")
    print(f"上界 (双対境界):   {dual_bound:.4f}")
    print(f"ギャップ: {(dual_bound - max_edges_count) / dual_bound * 100:.2f}%")

    selected_edges = [edges[i] for i in range(num_edges) if solution.col_value[i] > 0.5]
    print(f"選択辺数: {len(selected_edges)}")

    # C4検証
    violations = 0
    set_selected = set(selected_edges)
    for k1 in range(n):
        for k2 in range(k1 + 1, n):
            free_bits = [i for i in range(n) if i not in (k1, k2)]
            for m in range(1 << (n - 2)):
                base = 0
                for j in range(n - 2):
                    if m & (1 << j):
                        base |= (1 << free_bits[j])
                v1 = base; v2 = base^(1<<k1); v3 = base^(1<<k1)^(1<<k2); v4 = base^(1<<k2)
                e1=(min(v1,v2),max(v1,v2)); e2=(min(v2,v3),max(v2,v3))
                e3=(min(v3,v4),max(v3,v4)); e4=(min(v4,v1),max(v4,v1))
                count = sum(1 for e in [e1,e2,e3,e4] if e in set_selected)
                if count == 4:
                    violations += 1
    print(f"C4違反数: {violations}")

    # 次元別辺数
    print("\n=== 次元別辺数 ===")
    dim_count = [0] * n
    for (u, v) in selected_edges:
        diff = u ^ v
        for bit in range(n):
            if diff == (1 << bit):
                dim_count[bit] += 1; break
    for bit in range(n):
        print(f"  次元 {bit}: {dim_count[bit]}辺")

    # 頂点次数分布
    from collections import Counter
    deg = [0] * num_vertices
    for (u, v) in selected_edges:
        deg[u] += 1; deg[v] += 1
    deg_dist = Counter(deg)
    print("\n=== 頂点次数分布 ===")
    for d in sorted(deg_dist.keys()):
        print(f"  次数 {d}: {deg_dist[d]}頂点")

    # 解を保存
    save = True
    if os.path.exists(warm_start_file):
        with open(warm_start_file, 'r') as f:
            prev = json.load(f)
        if prev['value'] >= len(selected_edges):
            save = False
            print(f"\n既存解 ({prev['value']}) の方が良いため保存スキップ")

    if save and violations == 0:
        with open(warm_start_file, 'w') as f:
            json.dump({'value': len(selected_edges), 'n': n,
                       'edges': [list(e) for e in selected_edges]}, f)
        print(f"\n解を {warm_start_file} に保存 ({len(selected_edges)}辺)")

    with open(f'selected_edges_{len(selected_edges)}.txt', 'w') as f:
        for (u, v) in selected_edges:
            f.write(f"{u},{v}\n")
    print(f"辺リストを selected_edges_{len(selected_edges)}.txt に保存")

else:
    print(f"解決失敗: {model_status}")