from dataclasses import dataclass
import pickle
import json
from typing import Any
import datetime as dt
from pathlib import Path

import gurobipy as gp
from gurobipy import tupledict, Var
import numpy as np


def _gurobi_status_name(status: int) -> str:
    status_names = {
        gp.GRB.LOADED: "LOADED",
        gp.GRB.OPTIMAL: "OPTIMAL",
        gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
        gp.GRB.UNBOUNDED: "UNBOUNDED",
        gp.GRB.CUTOFF: "CUTOFF",
        gp.GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        gp.GRB.NODE_LIMIT: "NODE_LIMIT",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
        gp.GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        gp.GRB.INTERRUPTED: "INTERRUPTED",
        gp.GRB.NUMERIC: "NUMERIC",
        gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
        gp.GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return status_names.get(status, str(status))


PORTABLE_SOLUTION_SCHEMA_VERSION = 1
PORTABLE_VARIABLE_ARITIES = {
    "z_b": 1,
    "z_bq": 2,
    "y_bqtau": 3,
    "x_bqij": 3,
    "v_bqi": 3,
    "a_mbq": 3,
    "T_bqi": 3,
    "L_bqi": 3,
    "e_bqs": 3,
    "r_bmon": 1,
}


def _encode_utf8_array(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8)


def _decode_utf8_array(value: np.ndarray) -> str:
    return bytes(np.asarray(value, dtype=np.uint8).tolist()).decode("utf-8")


def _portable_meta_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _portable_meta_value(inner_value)
            for key, inner_value in value.items()
        }
    if isinstance(value, tuple | list | set):
        return [_portable_meta_value(item) for item in value]
    return str(value)


def _portable_index_tuple(key: Any, arity: int) -> tuple[int, ...]:
    if arity == 1:
        return (int(key),)
    if not isinstance(key, tuple):
        raise TypeError(f"expected tuple key for arity={arity}, got {type(key).__name__}")
    if len(key) != arity:
        raise ValueError(f"expected key of length {arity}, got {len(key)}")
    return tuple(int(idx) for idx in key)


def _portable_variable_payload(
    name: str,
    values: dict[Any, float | None],
) -> dict[str, np.ndarray]:
    arity = PORTABLE_VARIABLE_ARITIES[name]
    items: list[tuple[tuple[int, ...], float]] = []
    for key, value in values.items():
        if value is None or abs(value) <= 1e-9:
            continue
        items.append((_portable_index_tuple(key, arity), float(value)))
    items.sort(key=lambda item: item[0])

    if items:
        indices = np.asarray([key for key, _ in items], dtype=np.int64)
        if arity == 1:
            indices = indices.reshape(-1, 1)
        values_array = np.asarray([value for _, value in items], dtype=np.float64)
    else:
        indices = np.zeros((0, arity), dtype=np.int64)
        values_array = np.zeros((0,), dtype=np.float64)

    return {
        f"{name}_indices": indices,
        f"{name}_values": values_array,
    }


def _portable_solution_payload(solution: "Formulation3Solution") -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PORTABLE_SOLUTION_SCHEMA_VERSION, dtype=np.int64),
        "status": np.asarray(solution.status, dtype=np.int64),
        "status_name_utf8": _encode_utf8_array(solution.status_name),
        "has_objective_value": np.asarray(
            0 if solution.objective_value is None else 1,
            dtype=np.int64,
        ),
        "objective_value": np.asarray(
            np.nan if solution.objective_value is None else solution.objective_value,
            dtype=np.float64,
        ),
        "runtime_seconds": np.asarray(solution.runtime_seconds, dtype=np.float64),
        "meta_backend_utf8": _encode_utf8_array(str(solution.meta.get("backend", "python"))),
        "meta_json_utf8": _encode_utf8_array(
            json.dumps(_portable_meta_value(solution.meta), sort_keys=True)
        ),
    }
    for name in PORTABLE_VARIABLE_ARITIES:
        payload.update(_portable_variable_payload(name, solution.variables.get(name, {})))
    return payload


def _load_portable_variables(data: Any) -> dict[str, dict[Any, float | None]]:
    variables: dict[str, dict[Any, float | None]] = {}
    for name, arity in PORTABLE_VARIABLE_ARITIES.items():
        indices_key = f"{name}_indices"
        values_key = f"{name}_values"
        if indices_key not in data or values_key not in data:
            variables[name] = {}
            continue

        indices = np.asarray(data[indices_key], dtype=np.int64)
        values = np.asarray(data[values_key], dtype=np.float64)
        if indices.size == 0:
            variables[name] = {}
            continue
        if indices.ndim == 1:
            indices = indices.reshape(-1, arity)

        family_values: dict[Any, float | None] = {}
        for key_row, value in zip(indices, values, strict=True):
            key = int(key_row[0]) if arity == 1 else tuple(int(idx) for idx in key_row)
            family_values[key] = float(value)
        variables[name] = family_values
    return variables


def _load_portable_solution(path: Path) -> "Formulation3Solution":
    with np.load(path, allow_pickle=False) as data:
        if int(np.asarray(data["schema_version"]).item()) != PORTABLE_SOLUTION_SCHEMA_VERSION:
            raise ValueError(f"unsupported portable solution schema in {path}")

        objective_value = (
            None
            if int(np.asarray(data["has_objective_value"]).item()) == 0
            else float(np.asarray(data["objective_value"]).item())
        )
        meta_json = ""
        if "meta_json_utf8" in data:
            meta_json = _decode_utf8_array(data["meta_json_utf8"])
        meta = {} if meta_json == "" else json.loads(meta_json)
        if "meta_backend_utf8" in data:
            meta["backend"] = _decode_utf8_array(data["meta_backend_utf8"])
        for name in ("node_ids", "arc_src", "arc_dst"):
            key = f"meta_{name}"
            if key in data:
                meta[name] = np.asarray(data[key], dtype=np.int64).tolist()

        return Formulation3Solution(
            status=int(np.asarray(data["status"]).item()),
            status_name=_decode_utf8_array(data["status_name_utf8"]),
            objective_value=objective_value,
            runtime_seconds=float(np.asarray(data["runtime_seconds"]).item()),
            variables=_load_portable_variables(data),
            meta=meta,
        )


def _snapshot_tupledict(
    variables: tupledict[tuple[Any, ...], Var],
    *,
    has_values: bool,
) -> dict[Any, float | None]:
    return {
        key: (float(var.X) if has_values else None)
        for key, var in variables.items()
    }


@dataclass(slots=True)
class Formulation3Solution:
    status: int
    status_name: str
    objective_value: float | None
    runtime_seconds: float
    variables: dict[str, dict[Any, float | None]]
    meta: dict[str, Any]

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".npz":
            payload = _portable_solution_payload(self)
            np.savez_compressed(str(output_path), allow_pickle=False, **payload)
            return output_path

        with output_path.open("wb") as handle:
            pickle.dump(self, handle)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "Formulation3Solution":
        load_path = Path(path)
        if load_path.suffix == ".npz":
            return _load_portable_solution(load_path)

        with load_path.open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded


@dataclass(slots=True)
class Formulation3ModelBundle:
    model: gp.Model
    variables: dict[str, Any]
    meta: dict[str, Any]

    def snapshot(self) -> Formulation3Solution:
        has_values = self.model.SolCount > 0
        objective_value = float(self.model.ObjVal) if has_values else None
        return Formulation3Solution(
            status=self.model.Status,
            status_name=_gurobi_status_name(self.model.Status),
            objective_value=objective_value,
            runtime_seconds=float(self.model.Runtime),
            variables={
                name: _snapshot_tupledict(variable_set, has_values=has_values)
                for name, variable_set in self.variables.items()
            },
            meta=dict(self.meta),
        )


@dataclass(frozen=True, slots=True)
class _SnapshotVar:
    X: float | None


class _SnapshotVarSet(dict):
    def __init__(
        self,
        values: dict[Any, float | None],
        *,
        default_value: float | None = 0.0,
    ):
        super().__init__(values)
        self.default_value = default_value

    def __getitem__(self, key: Any) -> _SnapshotVar:
        return _SnapshotVar(X=super().get(key, self.default_value))


@dataclass(frozen=True, slots=True)
class _SnapshotModelView:
    status: int
    Status: int
    SolCount: int
    ObjVal: float | None
    Runtime: float


def _coerce_reporting_inputs(
    prob: gp.Model | Formulation3Solution,
    model_vars: dict[str, Any] | None,
) -> tuple[gp.Model | _SnapshotModelView, dict[str, Any]]:
    if isinstance(prob, Formulation3Solution):
        wrapped_vars = {
            name: _SnapshotVarSet(values)
            for name, values in prob.variables.items()
        }
        return (
            _SnapshotModelView(
                status=prob.status,
                Status=prob.status,
                SolCount=1 if prob.objective_value is not None else 0,
                ObjVal=prob.objective_value,
                Runtime=prob.runtime_seconds,
            ),
            {**wrapped_vars, "meta": prob.meta},
        )

    if model_vars is None:
        raise ValueError("model_vars are required when prob is a live gurobi model")
    return prob, model_vars
