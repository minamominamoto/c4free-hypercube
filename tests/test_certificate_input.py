"""Strict certificate input checks shared by both independent verifiers.

Run with ``python -m unittest discover -s tests -v``.  Standard library only;
the test file and the verifier interfaces support Python 3.8+.
"""

import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cross_verify
import verify


class IntegerSubclass(int):
    pass


INVALID_OUTER_CONTAINERS = (None, {}, {"edges": [[0, 1]]}, "01", 1)
INVALID_PAIRS = (
    [None], [0], ["01"], [{0: 0, 1: 1}], [{0, 1}],
    [[]], [[0]], [[0, 1, 2]], [(0,)], [(0, 1, 2)],
)
INVALID_ENDPOINTS = (
    [[False, 1]], [[0, True]], [[0.0, 1]], [[0, 1.9]],
    [["0", 1]], [[0, "1"]], [[None, 1]], [[0, None]],
    [[IntegerSubclass(0), 1]], [[0, IntegerSubclass(1)]],
)
SQUARE = [[0, 1], [0, 2], [1, 3], [2, 3]]


class CertificateInputTests(unittest.TestCase):
    def test_main_does_not_analyse_a_rejected_q8_record(self):
        with mock.patch.object(verify, "TARGETS", [(8, 680, 1, ["invalid.json"])]), \
             mock.patch.object(verify, "ODDSQUARE_TARGETS", []), \
             mock.patch.object(verify, "sha256", return_value="test-digest"), \
             mock.patch.object(verify, "load_solutions", return_value=[[[False, True]]]), \
             mock.patch.object(verify, "nonedge_violation_distribution") as margins, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(verify.main(), 1)
        margins.assert_not_called()
        self.assertIn("[FAIL] solution 0", output.getvalue())
        self.assertIn("FAILURES DETECTED", output.getvalue())

    def assert_reported_failure(self, function, edges, expected_edges=1):
        # Malformed certificate data must become a diagnostic, not escape as
        # a conversion, indexing, unpacking, or duplicate-edge exception.
        with contextlib.redirect_stdout(io.StringIO()):
            result = function(edges, 2, expected_edges)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], False)
        self.assertIsInstance(result[1], str)
        self.assertTrue(result[1])

    def test_build_accepts_list_and_tuple_pairs_in_either_order(self):
        for edges in ([[0, 1], (3, 1)], ((1, 0), [1, 3])):
            with self.subTest(edges=edges):
                expected = {(0, 1), (1, 3)}
                self.assertEqual(verify.build_edge_set(edges), expected)
                self.assertEqual(cross_verify.check_edges_valid(edges, 2), expected)

    def test_build_accepts_empty_list_and_tuple(self):
        for edges in ([], ()):
            with self.subTest(edges=edges):
                self.assertEqual(verify.build_edge_set(edges), set())
                self.assertEqual(cross_verify.check_edges_valid(edges, 2), set())

    def test_build_rejects_invalid_outer_containers(self):
        for edges in INVALID_OUTER_CONTAINERS:
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                verify.build_edge_set(edges)

    def test_build_rejects_non_pairs_and_extra_endpoints(self):
        for edges in INVALID_PAIRS:
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                verify.build_edge_set(edges)

    def test_build_rejects_non_integer_endpoint_types_without_coercion(self):
        for edges in INVALID_ENDPOINTS:
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                verify.build_edge_set(edges)

    def test_build_rejects_loops_and_reversed_duplicate_edges(self):
        for edges in ([[0, 0]], [[0, 1], [0, 1]], [[0, 1], [1, 0]]):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                verify.build_edge_set(edges)

    def test_public_verifiers_report_all_malformed_inputs_consistently(self):
        malformed = INVALID_OUTER_CONTAINERS + INVALID_PAIRS + INVALID_ENDPOINTS
        malformed += ([[0, 0]], [[0, 1], [1, 0]])
        for edges in malformed:
            for function in (verify.verify_solution, verify.verify_odd_square):
                with self.subTest(function=function.__name__, edges=edges):
                    self.assert_reported_failure(function, edges)
            with self.subTest(function="check_edges_valid", edges=edges):
                self.assertIsNone(cross_verify.check_edges_valid(edges, 2))

    def test_out_of_range_and_non_hypercube_edges_are_rejected(self):
        for edges in ([[-1, 0]], [[0, 4]], [[4, 5]], [[0, 3]]):
            for function in (verify.verify_solution, verify.verify_odd_square):
                with self.subTest(function=function.__name__, edges=edges):
                    self.assert_reported_failure(function, edges)
            with self.subTest(function="check_edges_valid", edges=edges):
                self.assertIsNone(cross_verify.check_edges_valid(edges, 2))

    def test_claimed_edge_count_is_checked(self):
        for function in (verify.verify_solution, verify.verify_odd_square):
            with self.subTest(function=function.__name__):
                self.assert_reported_failure(function, [[0, 1]], expected_edges=2)

    def test_empty_q2_is_c4_free_but_not_odd_square(self):
        self.assertEqual(verify.verify_solution([], 2, 0), (True, "ok"))
        self.assert_reported_failure(verify.verify_odd_square, [], expected_edges=0)
        self.assertEqual(cross_verify.analyse(set(), 2, cross_verify.dist2_masks(2)),
                         (True, False))

    def test_complete_q2_has_a_four_cycle(self):
        for function in (verify.verify_solution, verify.verify_odd_square):
            with self.subTest(function=function.__name__):
                self.assert_reported_failure(function, SQUARE, expected_edges=4)
        edges = cross_verify.check_edges_valid(SQUARE, 2)
        self.assertIsNotNone(edges)
        self.assertEqual(cross_verify.analyse(edges, 2, cross_verify.dist2_masks(2)),
                         (False, False))

    def test_three_q2_edges_are_c4_free_and_odd_square(self):
        edges = SQUARE[:-1]
        self.assertEqual(verify.verify_solution(edges, 2, 3), (True, "ok"))
        self.assertEqual(verify.verify_odd_square(edges, 2, 3), (True, "ok"))
        normalized = cross_verify.check_edges_valid(edges, 2)
        self.assertEqual(cross_verify.analyse(normalized, 2, cross_verify.dist2_masks(2)),
                         (True, True))

    @unittest.skipUnless((ROOT / "q6_odd_square_132.json").is_file(),
                         "published Q6 witness is not present")
    def test_published_q6_odd_square_witness_still_passes(self):
        with (ROOT / "q6_odd_square_132.json").open(encoding="utf-8") as handle:
            edges = json.load(handle)["edges"]
        self.assertEqual(len(edges), 132)
        self.assertEqual(verify.verify_solution(edges, 6, 132), (True, "ok"))
        self.assertEqual(verify.verify_odd_square(edges, 6, 132), (True, "ok"))
        normalized = cross_verify.check_edges_valid(edges, 6)
        self.assertIsNotNone(normalized)
        self.assertEqual(len(normalized), 132)
        self.assertEqual(cross_verify.analyse(normalized, 6, cross_verify.dist2_masks(6)),
                         (True, True))

    @unittest.skipUnless((ROOT / "selected_edges_best.json").is_file(),
                         "published Q7 witness is not present")
    def test_published_q7_witness_still_passes(self):
        with (ROOT / "selected_edges_best.json").open(encoding="utf-8") as handle:
            edges = json.load(handle)["edges"]
        self.assertEqual(len(edges), 304)
        self.assertEqual(verify.verify_solution(edges, 7, 304), (True, "ok"))
        normalized = cross_verify.check_edges_valid(edges, 7)
        self.assertIsNotNone(normalized)
        self.assertEqual(len(normalized), 304)
        self.assertIs(cross_verify.analyse(normalized, 7, cross_verify.dist2_masks(7))[0], True)


if __name__ == "__main__":
    unittest.main()
