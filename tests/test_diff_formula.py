"""差异公式编码测试：计数映射打包单射性、新旧编码等价、节点数增长渐近。

不依赖数据库。增长渐近测试通过 max_rows 参数（仅测试用，生产默认仍为 3）
在多个行数界上构造差异公式并统计不同 AST 节点数。
"""
import math

import pytest
from z3 import Solver, unsat

from app.ir import MAX_ROWS
from app.sql_parser import parse_query
from app.z3check import (
    _CELL_RADIX,
    _NULL_SLOT,
    _build_multisets,
    _multiset_diff,
    _multiset_diff_pairwise,
    _pack_key,
    build_diff_formula,
    count_ast_nodes,
)

# ---------- 打包键 ----------

def test_pack_key_injective_slots():
    """NULL 槽位落在 int4 平移值域之外，基数大于单元格取值数（打包单射）。"""
    # int4 值 v 平移为 v + 2^31 ∈ [0, 2^32)
    assert _NULL_SLOT == 2**32
    assert _CELL_RADIX > _NULL_SLOT


def test_pack_key_distinguishes_null_from_value():
    """NULL 单元格的打包键与任何非 NULL 单元格的打包键不可相等。"""
    from z3 import Bool, Int, Not

    n1, n2 = Bool("pk_n1"), Bool("pk_n2")
    v1, v2 = Int("pk_v1"), Int("pk_v2")
    k_null = _pack_key(((n1, v1),))
    k_val = _pack_key(((n2, v2),))
    solver = Solver()
    solver.add(n1, Not(n2))  # 左侧 NULL，右侧非 NULL
    solver.add(-(2**31) <= v2, v2 <= 2**31 - 1)  # int4 值域约束
    assert solver.check(k_null == k_val) == unsat


# ---------- 新旧编码在全部合法状态上等价 ----------

# 覆盖：单表过滤 / NULL 语义 / 重复行计数 / INNER / LEFT JOIN / 投影差异
_QUERY_PAIRS = [
    ("SELECT id FROM stations WHERE threshold > 0",
     "SELECT id FROM stations WHERE threshold >= 1"),
    ("SELECT id FROM stations WHERE threshold = 5",
     "SELECT id FROM stations WHERE threshold = 5 OR threshold IS NULL"),
    ("SELECT id FROM stations WHERE NOT (threshold <= 5)",
     "SELECT id FROM stations WHERE threshold > 5 OR threshold IS NULL"),
    ("SELECT station_id FROM readings",
     "SELECT station_id FROM readings WHERE value IS NOT NULL"),
    ("SELECT id, threshold FROM stations",
     "SELECT id, id FROM stations"),
    ("SELECT s.id FROM stations s LEFT JOIN readings r ON s.id = r.station_id",
     "SELECT s.id FROM stations s INNER JOIN readings r ON s.id = r.station_id"),
    ("SELECT s.id FROM stations s LEFT JOIN readings r ON s.id = r.station_id",
     "SELECT id FROM stations"),
    ("SELECT s.id, r.value FROM stations s INNER JOIN readings r "
     "ON s.id = r.station_id WHERE r.value >= 10",
     "SELECT s.id, r.value FROM stations s INNER JOIN readings r "
     "ON r.station_id = s.id WHERE NOT (r.value < 10)"),
    ("SELECT s.id FROM stations s LEFT JOIN readings r ON s.id = r.station_id "
     "WHERE r.station_id IS NULL",
     "SELECT id FROM stations"),
]


@pytest.mark.parametrize("a, b", _QUERY_PAIRS)
def test_bagmap_encoding_matches_pairwise(a, b):
    """在完整性约束下，两种差异公式不可同时取不同值（布尔不等式 UNSAT）。"""
    q1, q2 = parse_query(a), parse_query(b)
    db, ms1, ms2 = _build_multisets(q1, q2, MAX_ROWS)
    f_new = _multiset_diff(ms1, ms2)
    f_old = _multiset_diff_pairwise(ms1, ms2)
    solver = Solver()
    solver.set("timeout", 30000)
    solver.add(*db.constraints())
    solver.add(f_new != f_old)
    assert solver.check() == unsat


# ---------- 节点数增长渐近 ----------

# 两侧均为 LEFT JOIN：候选行数最多（每侧 m²+m）
_GROWTH_A = ("SELECT s.id, r.value FROM stations s LEFT JOIN readings r "
             "ON s.id = r.station_id")
_GROWTH_B = ("SELECT s.id, r.value FROM stations s LEFT JOIN readings r "
             "ON s.id = r.station_id WHERE r.value IS NOT NULL OR r.value IS NULL")
_GROWTH_BOUNDS = [2, 3, 4, 5, 6, 7]


@pytest.fixture(scope="module")
def node_counts():
    """各阶行数界下两种编码的差异公式节点数（模块内只构造一次）。"""
    q1, q2 = parse_query(_GROWTH_A), parse_query(_GROWTH_B)
    counts = []
    for m in _GROWTH_BOUNDS:
        n_total = 2 * (m * m + m)  # 两侧候选行总数
        old = count_ast_nodes(
            build_diff_formula(q1, q2, max_rows=m, method="pairwise")
        )
        new = count_ast_nodes(
            build_diff_formula(q1, q2, max_rows=m, method="bagmap")
        )
        counts.append((n_total, old, new))
    return counts


def _loglog_slope(xs, ys):
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    var = sum((v - mx) ** 2 for v in lx)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / var


def test_bagmap_encoding_smaller_at_every_bound(node_counts):
    for n_total, old, new in node_counts:
        assert new < old, f"候选行总数 {n_total}: bagmap={new} 应小于 pairwise={old}"


def test_bagmap_encoding_grows_slower_than_quadratic(node_counts):
    ns = [c[0] for c in node_counts]
    slope_old = _loglog_slope(ns, [c[1] for c in node_counts])
    slope_new = _loglog_slope(ns, [c[2] for c in node_counts])
    # 新编码为线性增长（斜率 ~1），须显著低于 2（平方），且低于旧编码
    assert slope_new < 1.5, f"新编码 log-log 斜率 {slope_new:.3f} 应 < 1.5"
    assert slope_old > slope_new + 0.3, (
        f"旧编码斜率 {slope_old:.3f} 应明显高于新编码 {slope_new:.3f}"
    )
    # 最大界上节点数减半以上
    n_total, old, new = node_counts[-1]
    assert 2 * new < old, f"候选行总数 {n_total}: bagmap={new}, pairwise={old}"


def test_production_bound_uses_three_rows():
    """生产默认行数界保持每表 0~3 行。"""
    assert MAX_ROWS == 3
    q1, q2 = parse_query(_GROWTH_A), parse_query(_GROWTH_B)
    default = count_ast_nodes(build_diff_formula(q1, q2))
    explicit = count_ast_nodes(build_diff_formula(q1, q2, max_rows=3))
    assert default == explicit
