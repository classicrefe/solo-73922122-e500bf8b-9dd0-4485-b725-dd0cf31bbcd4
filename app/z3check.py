"""基于 z3 的有界等价性判定。

在每表 0~MAX_ROWS 行、列值取 PostgreSQL INTEGER 全域（含 NULL）的所有合法
数据库状态下，判定两条查询的结果多重集（按列位置、忽略顺序、保留重复）是否
存在差异。差异条件：存在某个元组值，其在两侧结果中的出现次数不同。

差异公式默认采用计数映射编码（`_multiset_diff`）：把每个候选结果行按列位置
打包为单个整数键（NULL 映射到 int4 值域外的哨兵），每侧把"键 -> 出现次数"
累积成 z3 数组（每行一次 Store），两侧数组外延不等即存在差异。不同 AST
节点数为 Θ(k·n)（k 为投影列数，n 为两侧候选行总数），固定投影列数时随候选
行总数线性（而非平方）增长；旧实现 `_multiset_diff_pairwise`（按候选行反复
构造两侧计数，Θ(k·n²)）保留用于对照与规模基准。
"""
from __future__ import annotations

from dataclasses import dataclass

from z3 import (
    And as ZAnd,
    Bool,
    BoolVal,
    If,
    Implies,
    Int,
    IntSort,
    IntVal,
    K,
    Not as ZNot,
    Or as ZOr,
    Select,
    Solver,
    Store,
    is_true,
    sat,
    unknown as z3_unknown,
    unsat,
)
from z3.z3core import Z3_get_ast_id

from .ir import (
    FOREIGN_KEYS,
    INT_MAX,
    INT_MIN,
    MAX_ROWS,
    PRIMARY_KEY,
    SCHEMA,
    And,
    BoolNode,
    Compare,
    IsNull,
    Not,
    Or,
    QueryIR,
)

# 符号单元格：(is_null: z3 Bool, value: z3 Int)
Cell = tuple[object, object]
# 符号行：(存在条件, {(table, column): Cell})
Row = tuple[object, dict[tuple[str, str], "Cell"]]
# 结果多重集元素：(保留条件, 投影元组)
OutRow = tuple[object, tuple["Cell", ...]]


@dataclass
class CounterexampleData:
    """z3 模型提取出的具体数据库状态。"""

    stations: list[tuple[int, int | None]]          # (id, threshold)
    readings: list[tuple[int | None, int | None]]   # (station_id, value)


@dataclass
class Z3Outcome:
    status: str  # 'equivalent' | 'counterexample' | 'unknown'
    witness: CounterexampleData | None = None
    detail: str = ""


class SymbolicDB:
    """stations / readings 各 max_rows 行的符号化数据库状态及完整性约束。"""

    def __init__(self, max_rows: int = MAX_ROWS) -> None:
        self.max_rows = max_rows
        self.present: dict[str, list] = {}
        self.cells: dict[str, dict[str, list[Cell]]] = {}
        for tname, cols in SCHEMA.items():
            self.present[tname] = [Bool(f"{tname}_p{i}") for i in range(max_rows)]
            tcells: dict[str, list[Cell]] = {}
            for col in cols:
                nulls = [Bool(f"{tname}_{col}_n{i}") for i in range(max_rows)]
                vals = [Int(f"{tname}_{col}_v{i}") for i in range(max_rows)]
                tcells[col] = list(zip(nulls, vals))
            self.cells[tname] = tcells

    def constraints(self) -> list:
        cs: list = []
        for tname, cols in SCHEMA.items():
            pk = PRIMARY_KEY.get(tname)
            for col in cols:
                for i in range(self.max_rows):
                    is_null, val = self.cells[tname][col][i]
                    # int4 值域（NULL 标记与取值解耦，NULL 时取值无关）
                    cs.append(ZAnd(INT_MIN <= val, val <= INT_MAX))
                    if col == pk:
                        cs.append(ZNot(is_null))  # 主键非空
            if pk is not None:
                pkcells = self.cells[tname][pk]
                present = self.present[tname]
                for i in range(self.max_rows):
                    for j in range(i + 1, self.max_rows):
                        cs.append(
                            Implies(
                                ZAnd(present[i], present[j]),
                                pkcells[i][1] != pkcells[j][1],
                            )
                        )
        for child, fkcol, parent, pkcol in FOREIGN_KEYS:
            for i in range(self.max_rows):
                is_null, val = self.cells[child][fkcol][i]
                refs = ZOr(
                    *[
                        ZAnd(
                            self.present[parent][j],
                            self.cells[parent][pkcol][j][1] == val,
                        )
                        for j in range(self.max_rows)
                    ]
                )
                cs.append(
                    Implies(ZAnd(self.present[child][i], ZNot(is_null)), refs)
                )
        return cs


def _build_multisets(
    q1: QueryIR, q2: QueryIR, max_rows: int = MAX_ROWS
) -> tuple[SymbolicDB, list[OutRow], list[OutRow]]:
    """在同一符号数据库上展开两条查询的候选结果多重集。"""
    db = SymbolicDB(max_rows)
    return db, _result_multiset(q1, db), _result_multiset(q2, db)


def check_bounded_equivalence(
    q1: QueryIR, q2: QueryIR, timeout_ms: int = 10000, max_rows: int = MAX_ROWS
) -> Z3Outcome:
    """判定两条查询在界内是否等价；返回等价 / 反例 / 未知。"""
    db, ms1, ms2 = _build_multisets(q1, q2, max_rows)
    solver = Solver()
    solver.set("timeout", timeout_ms)
    solver.add(*db.constraints())
    solver.add(_multiset_diff(ms1, ms2))
    result = solver.check()
    if result == unsat:
        return Z3Outcome(
            "equivalent",
            detail=f"每表 0~{max_rows} 行的所有合法取值下结果多重集相同",
        )
    if result == z3_unknown:
        return Z3Outcome(
            "unknown", detail=f"求解器无法判定（原因: {solver.reason_unknown()}）"
        )
    if result != sat:  # pragma: no cover - 防御
        return Z3Outcome("unknown", detail=f"求解器返回异常状态: {result}")
    return Z3Outcome("counterexample", witness=_extract(solver.model(), db))


def build_diff_formula(
    q1: QueryIR, q2: QueryIR, max_rows: int = MAX_ROWS, method: str = "bagmap"
):
    """构造两条查询的结果多重集差异公式（供规模度量与编码对照）。

    method="bagmap"：计数映射编码（生产默认）；method="pairwise"：旧的按候选行
    反复计数的平方级编码（仅用于对照）。
    """
    db, ms1, ms2 = _build_multisets(q1, q2, max_rows)
    if method == "bagmap":
        return _multiset_diff(ms1, ms2)
    if method == "pairwise":
        return _multiset_diff_pairwise(ms1, ms2)
    raise ValueError(f"未知编码方式: {method}")


def count_ast_nodes(expr) -> int:
    """公式中不同 z3 AST 节点数（按 z3 内部唯一 id 计，共享子式只计一次）。"""
    seen: set[int] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        nid = Z3_get_ast_id(node.ctx_ref(), node.as_ast())
        if nid in seen:
            continue
        seen.add(nid)
        stack.extend(node.children())
    return len(seen)


# ---------- 查询编码 ----------

def _row_cells(db: SymbolicDB, tname: str, i: int) -> dict[tuple[str, str], Cell]:
    return {(tname, c): db.cells[tname][c][i] for c in SCHEMA[tname]}


def _null_cells(tname: str) -> dict[tuple[str, str], Cell]:
    return {(tname, c): (BoolVal(True), IntVal(0)) for c in SCHEMA[tname]}


def _base_rows(q: QueryIR, db: SymbolicDB) -> list[Row]:
    """FROM/JOIN 展开为符号行列表（含存在条件）。"""
    n = db.max_rows
    t1 = q.tables[0]
    if q.join is None:
        return [(db.present[t1][i], _row_cells(db, t1, i)) for i in range(n)]
    t2 = q.tables[1]
    jc1, jc2 = q.join.left_col, q.join.right_col

    def join_match(i: int, j: int):
        n1, v1 = db.cells[t1][jc1.column][i]
        n2, v2 = db.cells[t2][jc2.column][j]
        # SQL 等值连接：任一侧为 NULL 则不匹配
        return ZAnd(ZNot(n1), ZNot(n2), v1 == v2)

    rows: list[Row] = []
    for i in range(n):
        for j in range(n):
            present = ZAnd(db.present[t1][i], db.present[t2][j], join_match(i, j))
            cells = {**_row_cells(db, t1, i), **_row_cells(db, t2, j)}
            rows.append((present, cells))
    if q.join.kind == "left":
        # 左表无匹配行时补 NULL 扩展行
        for i in range(n):
            any_match = ZOr(
                *[ZAnd(db.present[t2][j], join_match(i, j)) for j in range(n)]
            )
            present = ZAnd(db.present[t1][i], ZNot(any_match))
            cells = {**_row_cells(db, t1, i), **_null_cells(t2)}
            rows.append((present, cells))
    return rows


def _encode_bool(node: BoolNode, cells: dict[tuple[str, str], Cell]):
    """SQL 三值逻辑编码，返回 (defined, value)。

    TRUE=(True,True)，FALSE=(True,False)，UNKNOWN=(False,*)。
    """
    if isinstance(node, Compare):
        is_null, val = cells[(node.col.table, node.col.column)]
        cmp_expr = _apply_cmp(node.op, val, node.value)
        return ZNot(is_null), cmp_expr
    if isinstance(node, IsNull):
        is_null, _ = cells[(node.col.table, node.col.column)]
        return BoolVal(True), ZNot(is_null) if node.negated else is_null
    if isinstance(node, And):
        da, va = _encode_bool(node.left, cells)
        db_, vb = _encode_bool(node.right, cells)
        defined = ZOr(ZAnd(da, db_), ZAnd(da, ZNot(va)), ZAnd(db_, ZNot(vb)))
        return defined, ZAnd(va, vb)
    if isinstance(node, Or):
        da, va = _encode_bool(node.left, cells)
        db_, vb = _encode_bool(node.right, cells)
        defined = ZOr(ZAnd(da, db_), ZAnd(da, va), ZAnd(db_, vb))
        return defined, ZOr(va, vb)
    if isinstance(node, Not):
        d, v = _encode_bool(node.child, cells)
        return d, ZNot(v)
    raise TypeError(f"未知布尔节点: {node!r}")


def _apply_cmp(op: str, val, const: int):
    if op == "=":
        return val == const
    if op == "!=":
        return val != const
    if op == "<":
        return val < const
    if op == "<=":
        return val <= const
    if op == ">":
        return val > const
    if op == ">=":
        return val >= const
    raise ValueError(op)  # pragma: no cover


def _result_multiset(q: QueryIR, db: SymbolicDB) -> list[OutRow]:
    out: list[OutRow] = []
    for present, cells in _base_rows(q, db):
        if q.where is not None:
            defined, val = _encode_bool(q.where, cells)
            kept = ZAnd(present, defined, val)
        else:
            kept = present
        proj = tuple(cells[(c.table, c.column)] for c in q.projections)
        out.append((kept, proj))
    return out


# ---------- 多重集差异编码 ----------

def _cell_eq(c1: Cell, c2: Cell):
    # 结果比较按值语义：NULL 与 NULL 视为相同
    return ZAnd(c1[0] == c2[0], ZOr(c1[0], c1[1] == c2[1]))


def _tuple_eq(t1: tuple[Cell, ...], t2: tuple[Cell, ...]):
    return ZAnd(*[_cell_eq(a, b) for a, b in zip(t1, t2)])


def _count(ms: list[OutRow], tup: tuple[Cell, ...]):
    terms = [If(ZAnd(kept, _tuple_eq(row, tup)), 1, 0) for kept, row in ms]
    total = 0
    for t in terms:
        total = total + t
    return total


def _multiset_diff_pairwise(ms1: list[OutRow], ms2: list[OutRow]):
    """旧编码（对照用）：存在某个出现的元组值，两侧计数不同。

    对每个候选行分别在两侧构造计数，元组相等比较共 (n1+n2)² 对，
    不同 AST 节点数为 Θ(k·(n1+n2)²)。
    """
    clauses = []
    for kept, tup in ms1:
        clauses.append(ZAnd(kept, _count(ms1, tup) != _count(ms2, tup)))
    for kept, tup in ms2:
        clauses.append(ZAnd(kept, _count(ms1, tup) != _count(ms2, tup)))
    return ZOr(*clauses) if clauses else BoolVal(False)


# ---- 计数映射编码（默认） ----
#
# 思路：把每个候选结果行按列位置打包为单个整数键——
#   NULL 单元格 -> _NULL_SLOT（int4 值域之外的哨兵，NULL 与 NULL 相等）；
#   其余单元格 -> 原 int4 值平移到 [0, 2^32)（v + 2^31）。
# 键按混合进制（基数 _CELL_RADIX） Horner 展开，因此在完整性约束下打包是
# 单射：键相等 ⟺ 结果元组按值语义相等。每侧把"键 -> 出现次数"累积成
# z3 数组（每行一次条件 Store；未保留行不贡献计数），两侧数组外延不等
# ⟺ 存在某个键两侧计数不同 ⟺ 结果多重集不同。每个候选行只贡献 O(k) 个
# 节点，故不同 AST 节点数为 Θ(k·n)，固定 k 时随 (n1+n2) 线性增长。
_NULL_SLOT = 2**32          # NULL 单元格的打包槽位（> 任何平移后的 int4 值）
_CELL_RADIX = 2**32 + 1     # 混合进制基数：2^32 个 int4 平移值 + NULL 槽位


def _pack_key(tup: tuple[Cell, ...]):
    """把候选结果行的投影元组打包为单个整数键（见上述说明）。"""
    key = None
    for is_null, val in tup:
        cell = If(is_null, IntVal(_NULL_SLOT), val + 2**31)
        key = cell if key is None else key * _CELL_RADIX + cell
    return key


def _count_map(ms: list[OutRow]):
    """把候选结果多重集累积为"键 -> 出现次数"的符号数组。"""
    counts = K(IntSort(), IntVal(0))  # 任意键 |-> 0
    for kept, tup in ms:
        key = _pack_key(tup)
        counts = If(kept, Store(counts, key, Select(counts, key) + 1), counts)
    return counts


def _multiset_diff(ms1: list[OutRow], ms2: list[OutRow]):
    """计数映射编码：两侧计数映射外延不等即存在差异。"""
    return _count_map(ms1) != _count_map(ms2)


# ---------- 模型提取 ----------

def _extract(model, db: SymbolicDB) -> CounterexampleData:
    def ev_bool(b) -> bool:
        return is_true(model.eval(b, model_completion=True))

    def ev_cell(cell: Cell):
        is_null, val = cell
        if ev_bool(is_null):
            return None
        return model.eval(val, model_completion=True).as_long()

    stations: list[tuple[int, int | None]] = []
    for i in range(db.max_rows):
        if ev_bool(db.present["stations"][i]):
            stations.append(
                (
                    model.eval(
                        db.cells["stations"]["id"][i][1], model_completion=True
                    ).as_long(),
                    ev_cell(db.cells["stations"]["threshold"][i]),
                )
            )
    readings: list[tuple[int | None, int | None]] = []
    for i in range(db.max_rows):
        if ev_bool(db.present["readings"][i]):
            readings.append(
                (
                    ev_cell(db.cells["readings"]["station_id"][i]),
                    ev_cell(db.cells["readings"]["value"][i]),
                )
            )
    return CounterexampleData(stations=stations, readings=readings)
