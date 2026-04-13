from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from formulation.common import (
    TAU,
    C_b,
    R_b,
    Wh_b,
    depot_b,
    f_m,
    l_s,
    make_depot_end_copy,
    make_depot_start_copy,
    p_m,
    s_m,
    tau_m,
)
from formulation.formulation_3.problem3_definition import (
    MILES_TO_KILOMETERS,
    Formulation3,
)


@dataclass(frozen=True, slots=True)
class SparseTriplet:
    n_rows: int
    n_cols: int
    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray

    def to_payload(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}_n_rows": np.asarray(self.n_rows, dtype=np.int64),
            f"{prefix}_n_cols": np.asarray(self.n_cols, dtype=np.int64),
            f"{prefix}_rows": self.rows,
            f"{prefix}_cols": self.cols,
            f"{prefix}_vals": self.vals,
        }


@dataclass(frozen=True, slots=True)
class Formulation3NumericInstance:
    ALPHA: float
    BETA: float
    PHI: float
    EPSILON: float
    H_RIDE: float
    LAMBDA_ROUND: float
    M_TIME: float
    M_CAPACITY: int
    T_horizon: float
    kappa_tau: np.ndarray

    nB: int
    nM: int
    nS: int
    nP: int
    nN: int
    nA: int
    nQ: int
    nTau: int

    arc_src: np.ndarray
    arc_dst: np.ndarray
    arc_time: np.ndarray
    arc_distance: np.ndarray

    pickup_node_p: np.ndarray
    pickup_node_of_m: np.ndarray
    school_node_of_m: np.ndarray
    school_index_of_m: np.ndarray
    tau_of_m: np.ndarray
    is_flagged_m: np.ndarray
    needs_wheelchair_m: np.ndarray

    capacity_b: np.ndarray
    cap_upper_b: np.ndarray
    range_b: np.ndarray
    wheelchair_ok_b: np.ndarray
    start_depot_node_b: np.ndarray
    end_depot_node_b: np.ndarray

    school_node_s: np.ndarray
    school_copy_node_s: np.ndarray
    latest_arrival_s: np.ndarray

    node_out_arc: SparseTriplet
    node_in_arc: SparseTriplet
    bus_start_arc: SparseTriplet
    bus_end_arc: SparseTriplet
    school_copy_out_arc: SparseTriplet
    school_copy_in_arc: SparseTriplet
    student_to_pickup: SparseTriplet
    student_to_school: SparseTriplet
    student_to_school_type: SparseTriplet

    bus_names: np.ndarray
    student_names: np.ndarray
    node_names: np.ndarray
    node_ids: np.ndarray
    school_ids: np.ndarray

    def to_payload(self) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "ALPHA": np.asarray(self.ALPHA, dtype=np.float64),
            "BETA": np.asarray(self.BETA, dtype=np.float64),
            "PHI": np.asarray(self.PHI, dtype=np.float64),
            "EPSILON": np.asarray(self.EPSILON, dtype=np.float64),
            "H_RIDE": np.asarray(self.H_RIDE, dtype=np.float64),
            "LAMBDA_ROUND": np.asarray(self.LAMBDA_ROUND, dtype=np.float64),
            "M_TIME": np.asarray(self.M_TIME, dtype=np.float64),
            "M_CAPACITY": np.asarray(self.M_CAPACITY, dtype=np.int64),
            "T_horizon": np.asarray(self.T_horizon, dtype=np.float64),
            "kappa_tau": self.kappa_tau,
            "nB": np.asarray(self.nB, dtype=np.int64),
            "nM": np.asarray(self.nM, dtype=np.int64),
            "nS": np.asarray(self.nS, dtype=np.int64),
            "nP": np.asarray(self.nP, dtype=np.int64),
            "nN": np.asarray(self.nN, dtype=np.int64),
            "nA": np.asarray(self.nA, dtype=np.int64),
            "nQ": np.asarray(self.nQ, dtype=np.int64),
            "nTau": np.asarray(self.nTau, dtype=np.int64),
            "arc_src": self.arc_src,
            "arc_dst": self.arc_dst,
            "arc_time": self.arc_time,
            "arc_distance": self.arc_distance,
            "pickup_node_p": self.pickup_node_p,
            "pickup_node_of_m": self.pickup_node_of_m,
            "school_node_of_m": self.school_node_of_m,
            "school_index_of_m": self.school_index_of_m,
            "tau_of_m": self.tau_of_m,
            "is_flagged_m": self.is_flagged_m,
            "needs_wheelchair_m": self.needs_wheelchair_m,
            "capacity_b": self.capacity_b,
            "cap_upper_b": self.cap_upper_b,
            "range_b": self.range_b,
            "wheelchair_ok_b": self.wheelchair_ok_b,
            "start_depot_node_b": self.start_depot_node_b,
            "end_depot_node_b": self.end_depot_node_b,
            "school_node_s": self.school_node_s,
            "school_copy_node_s": self.school_copy_node_s,
            "latest_arrival_s": self.latest_arrival_s,
            "bus_names": _np_bytes_array(self.bus_names.tolist()),
            "student_names": _np_bytes_array(self.student_names.tolist()),
            "node_names": _np_bytes_array(self.node_names.tolist()),
            "node_ids": self.node_ids,
            "school_ids": _np_bytes_array(self.school_ids.tolist()),
        }
        for name in (
            "node_out_arc",
            "node_in_arc",
            "bus_start_arc",
            "bus_end_arc",
            "school_copy_out_arc",
            "school_copy_in_arc",
            "student_to_pickup",
            "student_to_school",
            "student_to_school_type",
        ):
            payload.update(getattr(self, name).to_payload(name))
        return payload


def _student_identifier(student: Any) -> str:
    return str(getattr(student, "id", student.name))


def _np_bytes_array(values: list[str]) -> np.ndarray:
    return np.asarray([value.encode("utf-8") for value in values], dtype=np.bytes_)


def _triplet_from_memberships(
    n_rows: int,
    n_cols: int,
    memberships: list[list[int]],
) -> SparseTriplet:
    rows: list[int] = []
    cols: list[int] = []
    for row_idx, members in enumerate(memberships, start=1):
        rows.extend([row_idx] * len(members))
        cols.extend(member + 1 for member in members)
    vals = np.ones(len(rows), dtype=np.float64)
    return SparseTriplet(
        n_rows=n_rows,
        n_cols=n_cols,
        rows=np.asarray(rows, dtype=np.int64),
        cols=np.asarray(cols, dtype=np.int64),
        vals=vals,
    )


def _triplet_from_single_mapping(
    n_rows: int,
    n_cols: int,
    row_for_col: list[int],
) -> SparseTriplet:
    cols = np.arange(1, len(row_for_col) + 1, dtype=np.int64)
    rows = np.asarray([row_idx + 1 for row_idx in row_for_col], dtype=np.int64)
    vals = np.ones(len(row_for_col), dtype=np.float64)
    return SparseTriplet(n_rows=n_rows, n_cols=n_cols, rows=rows, cols=cols, vals=vals)


def build_formulation3_numeric_instance(
    problem: Formulation3,
) -> Formulation3NumericInstance:
    B = problem.B
    M = problem.M
    S = problem.S
    S_PLUS = problem.S_PLUS
    P = problem.P
    A = problem.A
    N = problem.N

    A_list = list(A.keys())
    node_to_idx = {node: idx for idx, node in enumerate(N)}
    stop_to_idx = {stop: idx for idx, stop in enumerate(P)}
    school_to_idx = {school: idx for idx, school in enumerate(S)}
    school_type_to_idx = {school_type: idx for idx, school_type in enumerate(TAU)}

    arcs_from_node = [[] for _ in N]
    arcs_to_node = [[] for _ in N]
    arc_src = np.empty(len(A_list), dtype=np.int64)
    arc_dst = np.empty(len(A_list), dtype=np.int64)
    arc_time = np.empty(len(A_list), dtype=np.float64)
    arc_distance = np.empty(len(A_list), dtype=np.float64)

    for arc_idx, (start_node, end_node) in enumerate(A_list):
        start_idx = node_to_idx[start_node]
        end_idx = node_to_idx[end_node]
        arcs_from_node[start_idx].append(arc_idx)
        arcs_to_node[end_idx].append(arc_idx)
        arc_src[arc_idx] = start_idx + 1
        arc_dst[arc_idx] = end_idx + 1
        arc_time[arc_idx] = problem.t_ij(start_node, end_node)
        arc_distance[arc_idx] = problem.d_ij(start_node, end_node)

    pickup_node_p = np.asarray([node_to_idx[stop] + 1 for stop in P], dtype=np.int64)
    school_node_s = np.asarray(
        [node_to_idx[school] + 1 for school in S], dtype=np.int64
    )
    school_copy_node_s = np.asarray(
        [node_to_idx[school_copy] + 1 for school_copy in S_PLUS], dtype=np.int64
    )

    pickup_node_of_m: list[int] = []
    school_node_of_m: list[int] = []
    school_index_of_m: list[int] = []
    tau_of_m: list[int] = []
    is_flagged_m: list[int] = []
    needs_wheelchair_m: list[int] = []
    pickup_row_for_student: list[int] = []
    school_row_for_student: list[int] = []
    tau_row_for_student: list[int] = []

    for student in M:
        pickup_stop = p_m(student)
        school = s_m(student)
        school_type = tau_m(student)

        pickup_node_of_m.append(node_to_idx[pickup_stop])
        school_node_of_m.append(node_to_idx[school])
        school_index = school_to_idx[school]
        school_index_of_m.append(school_index)
        tau_index = school_type_to_idx[school_type]
        tau_of_m.append(tau_index)
        is_flagged_m.append(f_m(student))
        needs_wheelchair_m.append(int(student.requires_wheelchair))
        pickup_row_for_student.append(stop_to_idx[pickup_stop])
        school_row_for_student.append(school_index)
        tau_row_for_student.append(tau_index)

    start_depot_node_b = np.asarray(
        [node_to_idx[make_depot_start_copy(depot_b(bus))] + 1 for bus in B],
        dtype=np.int64,
    )
    end_depot_node_b = np.asarray(
        [node_to_idx[make_depot_end_copy(depot_b(bus))] + 1 for bus in B],
        dtype=np.int64,
    )

    bus_start_memberships = [
        arcs_from_node[start_node_idx - 1] for start_node_idx in start_depot_node_b
    ]
    bus_end_memberships = [
        arcs_to_node[end_node_idx - 1] for end_node_idx in end_depot_node_b
    ]
    school_copy_out_memberships = [
        arcs_from_node[node_idx - 1] for node_idx in school_copy_node_s
    ]
    school_copy_in_memberships = [
        arcs_to_node[node_idx - 1] for node_idx in school_copy_node_s
    ]

    capacity_b = np.asarray([C_b(bus) for bus in B], dtype=np.int64)
    cap_upper_b = np.asarray([problem.C_CAP_B(bus) for bus in B], dtype=np.float64)
    range_b = np.asarray(
        [R_b(bus) * MILES_TO_KILOMETERS for bus in B],
        dtype=np.float64,
    )
    wheelchair_ok_b = np.asarray([Wh_b(bus) for bus in B], dtype=np.int64)
    latest_arrival_s = np.asarray([l_s(school) for school in S], dtype=np.float64)
    kappa_tau = np.asarray(
        [problem.KAPPA[school_type] for school_type in TAU],
        dtype=np.float64,
    )

    return Formulation3NumericInstance(
        ALPHA=problem.ALPHA,
        BETA=problem.BETA,
        PHI=problem.PHI,
        EPSILON=problem.EPSILON,
        H_RIDE=problem.H_RIDE,
        LAMBDA_ROUND=problem.LAMBDA_ROUND,
        M_TIME=problem.M_TIME,
        M_CAPACITY=problem.M_CAPACITY,
        T_horizon=problem.T_horizon,
        kappa_tau=kappa_tau,
        nB=len(B),
        nM=len(M),
        nS=len(S),
        nP=len(P),
        nN=len(N),
        nA=len(A_list),
        nQ=len(problem.Q),
        nTau=len(TAU),
        arc_src=arc_src,
        arc_dst=arc_dst,
        arc_time=arc_time,
        arc_distance=arc_distance,
        pickup_node_p=pickup_node_p,
        pickup_node_of_m=np.asarray(
            [node_idx + 1 for node_idx in pickup_node_of_m], dtype=np.int64
        ),
        school_node_of_m=np.asarray(
            [node_idx + 1 for node_idx in school_node_of_m], dtype=np.int64
        ),
        school_index_of_m=np.asarray(
            [school_idx + 1 for school_idx in school_index_of_m], dtype=np.int64
        ),
        tau_of_m=np.asarray([tau_idx + 1 for tau_idx in tau_of_m], dtype=np.int64),
        is_flagged_m=np.asarray(is_flagged_m, dtype=np.int64),
        needs_wheelchair_m=np.asarray(needs_wheelchair_m, dtype=np.int64),
        capacity_b=capacity_b,
        cap_upper_b=cap_upper_b,
        range_b=range_b,
        wheelchair_ok_b=wheelchair_ok_b,
        start_depot_node_b=start_depot_node_b,
        end_depot_node_b=end_depot_node_b,
        school_node_s=school_node_s,
        school_copy_node_s=school_copy_node_s,
        latest_arrival_s=latest_arrival_s,
        node_out_arc=_triplet_from_memberships(len(N), len(A_list), arcs_from_node),
        node_in_arc=_triplet_from_memberships(len(N), len(A_list), arcs_to_node),
        bus_start_arc=_triplet_from_memberships(
            len(B), len(A_list), bus_start_memberships
        ),
        bus_end_arc=_triplet_from_memberships(len(B), len(A_list), bus_end_memberships),
        school_copy_out_arc=_triplet_from_memberships(
            len(S), len(A_list), school_copy_out_memberships
        ),
        school_copy_in_arc=_triplet_from_memberships(
            len(S), len(A_list), school_copy_in_memberships
        ),
        student_to_pickup=_triplet_from_single_mapping(
            len(P), len(M), pickup_row_for_student
        ),
        student_to_school=_triplet_from_single_mapping(
            len(S), len(M), school_row_for_student
        ),
        student_to_school_type=_triplet_from_single_mapping(
            len(TAU), len(M), tau_row_for_student
        ),
        bus_names=np.asarray([bus.name for bus in B], dtype=str),
        student_names=np.asarray(
            [_student_identifier(student) for student in M], dtype=str
        ),
        node_names=np.asarray([node.name for node in N], dtype=str),
        node_ids=np.asarray([node.node_id for node in N], dtype=np.int64),
        school_ids=np.asarray([school.id for school in S], dtype=str),
    )


def export_formulation3_instance(problem: Formulation3, path: str | Path) -> Path:
    instance = build_formulation3_numeric_instance(problem)
    requested_path = Path(path)
    if requested_path.suffix == ".npz":
        output_path = requested_path
    elif requested_path.suffix:
        output_path = requested_path.with_suffix(requested_path.suffix + ".npz")
    else:
        output_path = requested_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **instance.to_payload())
    return output_path
