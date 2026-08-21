import argparse
import time
import sys
import numpy as np
import networkx as nx
from pulp import *

print("=== Q7 C4-free 最大部分グラフ 攻撃開始 ===")
print("Erdős $100問題  n=7版")

# ====================== ILP ======================
def solve_ilp_n7(timelimit=3600*8, solver_name='CBC'):
    print(f"\nILP開始 (タイムリミット: {timelimit//3600}時間)")
    G = nx.hypercube_graph(7)
    mapping = {node: int(''.join(map(str, node)), 2) for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    edges = list(G.edges())
    total = len(edges)

    # C4列挙（高速化版）
    c4s = []
    for u in G:
        for v in G[u]:
            if v > u:
                common = set(G[u]) & set(G[v])
                for w in common:
                    for x in common:
                        if w < x and G.has_edge(w, x):
                            c4 = [(u,v),(v,w),(w,x),(x,u)]
                            c4s.append([tuple(sorted(e)) for e in c4])

    c4s = [list(dict.fromkeys(c)) for c in c4s]  # 重複除去

    prob = LpProblem("MaxC4Free_Q7", LpMaximize)
    x = {e: LpVariable(f"x_{e[0]}_{e[1]}", cat='Binary') for e in edges}
    prob += lpSum(x.values())

    for c4 in c4s:
        prob += lpSum(x[e] for e in c4 if e in x) <= 3

    print(f"C4数: {len(c4s)}")

    if solver_name == 'HiGHS':
        solver = HiGHS_CMD(msg=1, timeLimit=timelimit)
    else:
        solver = PULP_CBC_CMD(msg=1, timeLimit=timelimit, options=['threads 0'])

    prob.solve(solver)

    selected = [e for e in edges if value(x[e]) > 0.5]
    density = len(selected) / total
    print(f"\n結果: {len(selected)}辺 / {total}  (密度 {density:.5f})")
    print(f"Brass下界 {0.5 + 1/(2*np.sqrt(7)):.5f} との比較")
    return len(selected), density

# ====================== メイン ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['ilp', 'heuristic'], default='ilp')
    parser.add_argument('--timelimit', type=int, default=28800)  # 8時間
    parser.add_argument('--solver', choices=['CBC', 'HiGHS'], default='CBC')
    args = parser.parse_args()

    if args.mode == 'ilp':
        solve_ilp_n7(args.timelimit, args.solver)
    else:
        print("heuristicモードは別途実装中（必要なら言ってください）")

print("\n実行例:")
print("  python3 hypercube_n7.py --mode ilp --timelimit 28800 --solver HiGHS")