"""Tests for the GlobalSpillGraph minimax solve (B3)."""

from __future__ import annotations

import math

import numpy as np

from digitalrivers._outofcore.spillgraph import OUTLET, GlobalSpillGraph


class TestEdges:
    def test_keeps_minimum_per_pair_and_is_order_independent(self):
        g = GlobalSpillGraph()
        g.add_edge(2, 1, 9.0)
        g.add_edge(1, 2, 4.0)  # lower, should win
        g.add_edge(1, 2, 7.0)  # higher, ignored
        assert g.edges[(1, 2)] == 4.0

    def test_self_edges_ignored(self):
        g = GlobalSpillGraph()
        g.add_edge(3, 3, 1.0)
        assert g.edges == {}


class TestSolve:
    def test_direct_outlet(self):
        g = GlobalSpillGraph()
        g.add_outlet(1, 5.0)
        drain = g.solve()
        assert drain[OUTLET] == -math.inf
        assert drain[1] == 5.0

    def test_minimax_path(self):
        # label 2 reaches the outlet only via label 1; its drain is the higher saddle on the path.
        g = GlobalSpillGraph()
        g.add_outlet(1, 5.0)
        g.add_edge(1, 2, 8.0)
        drain = g.solve()
        assert drain[1] == 5.0
        assert drain[2] == 8.0

    def test_picks_lower_bottleneck_path(self):
        # label 3 can reach outlet via (10) or via 2->1 (max 6); should pick 6.
        g = GlobalSpillGraph()
        g.add_outlet(1, 2.0)
        g.add_edge(1, 2, 6.0)
        g.add_edge(2, 3, 4.0)
        g.add_outlet(3, 10.0)
        drain = g.solve()
        assert drain[3] == 6.0

    def test_unreachable_label_absent(self):
        g = GlobalSpillGraph()
        g.add_outlet(1, 1.0)
        g.add_edge(5, 6, 2.0)  # disconnected component
        drain = g.solve()
        assert 5 not in drain and 6 not in drain

    def test_add_adjacency_records_saddles(self):
        labels = np.array([[1, 1, 2], [1, 1, 2]], dtype=np.int64)
        filled = np.array([[1.0, 3.0, 9.0], [1.0, 5.0, 9.0]], dtype=np.float64)
        g = GlobalSpillGraph()
        g.add_adjacency(labels, filled)
        # 1 and 2 meet; lowest saddle = max(3,9)=9 at the top, max(5,9)=9 — min is 9
        assert g.edges[(1, 2)] == 9.0
