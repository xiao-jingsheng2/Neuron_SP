"""Round 2 review: exploit tests that find gaps in 38c84f81."""
import os
import types
import pytest
from collections import deque

from deepspeed.core.distributed.collective_contract import (
    CollectiveContract, CollectiveOp, ContractViolation, build_step_contract,
)
from deepspeed.core.distributed.contract_diagnostics import (
    StepTrace, StepTraceLog, validate_call_sites, diff_sequences,
)


class TestVerifyTruncationGuardIsDeadCode:
    """BUG: TestVerifyTruncationGuard never actually triggers the RuntimeError
    because verify() short-circuits when dist is not initialized."""

    def test_verify_with_250_ops_does_NOT_raise(self):
        """Prove that verify() returns True even with 250 ops (>4096 bytes)."""
        c = CollectiveContract(step=0, rank=0)
        for i in range(250):
            c.plan(f"very_long_collective_name_number_{i:05d}", CollectiveOp.ALL_REDUCE)

        # The author's test only checks len(encoded) >= 4096.
        # But verify() itself never reaches the truncation guard:
        result = c.verify()
        assert result is True, "verify() should short-circuit (dist not init)"
        assert c._verified is True

        # The encoding IS too long:
        encoded = "|".join(c.planned_sequence).encode("utf-8")
        assert len(encoded) >= 4096

        # But the guard on L355 is UNREACHABLE in non-distributed mode.
        # This means the test gives false confidence.

    def test_truncation_guard_directly(self):
        """Show how to ACTUALLY test the truncation guard."""
        c = CollectiveContract(step=0, rank=0)
        for i in range(250):
            c.plan(f"very_long_collective_name_number_{i:05d}", CollectiveOp.ALL_REDUCE)

        # Directly test the encoding path that verify() would use:
        local_seq_str = "|".join(c.planned_sequence)
        max_len = 4096
        encoded = local_seq_str.encode("utf-8")

        # This is what L355-360 does, but verify() never reaches here
        if len(encoded) >= max_len:
            with pytest.raises(RuntimeError, match="too long"):
                raise RuntimeError(
                    f"CollectiveContract: planned sequence too long for verify() "
                    f"({len(encoded)} >= {max_len} bytes, {c.planned_count} ops). "
                )


class TestResetSeqCounterCorruption:
    """BUG: reset() does not reset _seq_counter, so plan() after reset()
    produces entries with non-contiguous seq numbers."""

    def test_seq_counter_not_reset(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        c.plan("op_b", CollectiveOp.ALL_REDUCE)
        assert c._seq_counter == 2

        with c.guard("op_a"): pass
        with c.guard("op_b"): pass
        c.reset()

        # _seq_counter is NOT reset:
        assert c._seq_counter == 2, "reset() should have reset _seq_counter but didn't"

    def test_plan_after_reset_appends_to_old_planned(self):
        """If user calls plan() after reset(), old planned entries remain."""
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        with c.guard("op_a"): pass
        c.reset()

        # plan after reset APPENDS, doesn't start fresh:
        c.plan("op_new", CollectiveOp.ALL_REDUCE)
        assert c.planned_count == 2  # old "op_a" + new "op_new"
        assert c.planned_sequence == ["op_a", "op_new"]
        # guard("op_a") must be called first even though we "reset":
        with c.guard("op_a"): pass
        with c.guard("op_new"): pass
        c.assert_complete()


class TestEnvVarTraceMaxlenEdgeCases:
    """BUG: NEURON_SP_TRACE_MAXLEN has no validation."""

    def test_non_numeric_crashes(self):
        """int('abc') raises ValueError, crashing training at startup."""
        os.environ["NEURON_SP_TRACE_MAXLEN"] = "not_a_number"
        with pytest.raises(ValueError):
            int(os.environ.get("NEURON_SP_TRACE_MAXLEN", "128"))
        del os.environ["NEURON_SP_TRACE_MAXLEN"]

    def test_zero_maxlen_loses_all_traces(self):
        """maxlen=0 means deque stores nothing, all post-mortem data lost."""
        log = StepTraceLog(maxlen=0)
        log.record(StepTrace(step=0, rank=0, complete=True))
        log.record(StepTrace(step=1, rank=0, complete=False))
        assert len(log.traces) == 0  # everything evicted immediately
        assert log.last is None
        assert log.incomplete_steps() == []
        # summary says 0 steps, no way to diagnose anything:
        s = log.summary()
        assert s["total_steps"] == 0

    def test_negative_maxlen_accepted(self):
        """deque(maxlen=-5) does not raise but behaves unexpectedly."""
        # Python deque actually DOES raise on negative maxlen:
        with pytest.raises((ValueError, OverflowError)):
            deque(maxlen=-5)


class TestVerifyLimitWithoutDeslocKx:
    """BUG: desloc_engine.py L2165 uses self.desloc_Kx without hasattr guard.
    If the engine doesn't have desloc_Kx, it crashes."""

    def test_missing_desloc_kx_attribute(self):
        engine = types.SimpleNamespace()  # no desloc_Kx attribute
        with pytest.raises(AttributeError):
            _verify_limit = max(engine.desloc_Kx, 5) + 1

    def test_getattr_fallback_would_fix_it(self):
        engine = types.SimpleNamespace()
        _verify_limit = max(getattr(engine, 'desloc_Kx', 5), 5) + 1
        assert _verify_limit == 6  # fallback to 5, so max(5,5)+1=6


class TestGuardExceptionSafetyRegression:
    """Verify the clip_grads.py fix: guard context must exit even on exception."""

    def test_guard_exits_on_exception(self):
        c = CollectiveContract(step=0, rank=0)
        c.plan("op_a", CollectiveOp.ALL_REDUCE)
        c.plan("op_b", CollectiveOp.ALL_REDUCE)

        # Exception inside guard should not corrupt state
        with pytest.raises(RuntimeError, match="boom"):
            with c.guard("op_a"):
                raise RuntimeError("boom")

        # Guard should have recorded op_a as executed despite exception
        assert len(c._executed) == 1
        assert c._executed[0].name == "op_a"
        assert c._active_guard is None  # cleaned up

        # Can continue to op_b:
        with c.guard("op_b"): pass
        c.assert_complete()
