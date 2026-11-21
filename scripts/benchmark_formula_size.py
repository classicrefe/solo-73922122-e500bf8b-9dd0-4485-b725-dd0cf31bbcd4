#!/usr/bin/env python
"""差异公式规模基准：旧（按候选行反复计数）vs 新（计数映射）编码。

用法：
    python scripts/benchmark_formula_size.py

在每个行数界 m（每表 0~m 行，生产默认 m=3）上，对一对 LEFT JOIN 查询
（候选行数最多的情形：每侧 m²+m 行）构造两种差异公式，统计不同 z3 AST
节点数并拟合 log-log 斜率。斜率 ~2 表示随两侧候选行总数平方增长，
~1 表示线性增长。
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.sql_parser import parse_query
from app.z3check import build_diff_formula, count_ast_nodes

QUERY_A = ("SELECT s.id, r.value FROM stations s LEFT JOIN readings r "
           "ON s.id = r.station_id")
QUERY_B = ("SELECT s.id, r.value FROM stations s LEFT JOIN readings r "
           "ON s.id = r.station_id WHERE r.value IS NOT NULL OR r.value IS NULL")

BOUNDS = [2, 3, 4, 5, 6, 7, 8]


def loglog_slope(xs, ys):
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    var = sum((v - mx) ** 2 for v in lx)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / var


def main() -> None:
    q1, q2 = parse_query(QUERY_A), parse_query(QUERY_B)
    print(f"{'m':>3} {'候选行/侧':>9} {'候选行总数':>10} "
          f"{'旧(计数)':>9} {'新(计数映射)':>12} {'缩小倍数':>8}")
    rows = []
    for m in BOUNDS:
        n_side = m * m + m
        old = count_ast_nodes(
            build_diff_formula(q1, q2, max_rows=m, method="pairwise")
        )
        new = count_ast_nodes(
            build_diff_formula(q1, q2, max_rows=m, method="bagmap")
        )
        rows.append((2 * n_side, old, new))
        print(f"{m:>3} {n_side:>9} {2 * n_side:>10} "
              f"{old:>9} {new:>12} {old / new:>7.2f}x")
    ns = [r[0] for r in rows]
    print(f"\nlog-log 斜率（节点数 ~ 候选行总数^k）：")
    print(f"  旧编码（按候选行反复计数）: k = {loglog_slope(ns, [r[1] for r in rows]):.3f}")
    print(f"  新编码（计数映射）        : k = {loglog_slope(ns, [r[2] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
