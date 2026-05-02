from __future__ import annotations

import unittest
from dataclasses import dataclass

from experiments.all_gurobi_julia import _build_run_scopes
from formulation.common import SchoolType


@dataclass(frozen=True)
class FakeSchool:
    id: str
    name: str
    type: SchoolType


@dataclass(frozen=True)
class FakeStudent:
    school: FakeSchool


class FakeProblemData:
    def __init__(self, schools: list[FakeSchool], students: list[FakeStudent]) -> None:
        self.schools = schools
        self.students = students
        self.school_calls: list[str] = []
        self.school_type_calls: list[SchoolType] = []

    def restrict_to_school(self, school_id: str):
        self.school_calls.append(school_id)
        return f"school:{school_id}"

    def restrict_to_school_type(self, school_type: SchoolType):
        self.school_type_calls.append(school_type)
        return f"type:{school_type.name}"


class AllGurobiJuliaScopeTests(unittest.TestCase):
    def test_default_scope_keeps_full_problem(self) -> None:
        school = FakeSchool(id="FUL", name="Fuller", type=SchoolType.E)
        problem_data = FakeProblemData([school], [FakeStudent(school=school)])

        scopes = _build_run_scopes(
            problem_data,
            per_school=False,
            per_school_type=False,
        )

        self.assertEqual(len(scopes), 1)
        self.assertEqual(scopes[0].label_suffix, "")
        self.assertIs(scopes[0].problem_data, problem_data)
        self.assertEqual(problem_data.school_calls, [])
        self.assertEqual(problem_data.school_type_calls, [])

    def test_per_school_skips_schools_without_students(self) -> None:
        fuller = FakeSchool(id="FUL", name="Fuller", type=SchoolType.E)
        mccarthy = FakeSchool(id="MCC", name="McCarthy", type=SchoolType.MS)
        empty = FakeSchool(id="EMP", name="Empty", type=SchoolType.HS)
        problem_data = FakeProblemData(
            [fuller, mccarthy, empty],
            [FakeStudent(school=fuller), FakeStudent(school=mccarthy)],
        )

        scopes = _build_run_scopes(
            problem_data,
            per_school=True,
            per_school_type=False,
        )

        self.assertEqual(
            [scope.label_suffix for scope in scopes],
            ["school_ful", "school_mcc"],
        )
        self.assertEqual(
            [scope.problem_data for scope in scopes],
            ["school:FUL", "school:MCC"],
        )
        self.assertEqual(problem_data.school_calls, ["FUL", "MCC"])
        self.assertEqual(problem_data.school_type_calls, [])

    def test_per_school_type_uses_active_types_only(self) -> None:
        elementary = FakeSchool(id="FUL", name="Fuller", type=SchoolType.E)
        high = FakeSchool(id="FHS", name="Framingham High", type=SchoolType.HS)
        unused_middle = FakeSchool(id="MCC", name="McCarthy", type=SchoolType.MS)
        problem_data = FakeProblemData(
            [elementary, high, unused_middle],
            [FakeStudent(school=high), FakeStudent(school=elementary)],
        )

        scopes = _build_run_scopes(
            problem_data,
            per_school=False,
            per_school_type=True,
        )

        self.assertEqual(
            [scope.label_suffix for scope in scopes],
            ["school_type_e", "school_type_hs"],
        )
        self.assertEqual(
            [scope.problem_data for scope in scopes],
            ["type:E", "type:HS"],
        )
        self.assertEqual(problem_data.school_calls, [])
        self.assertEqual(problem_data.school_type_calls, [SchoolType.E, SchoolType.HS])


if __name__ == "__main__":
    unittest.main()
