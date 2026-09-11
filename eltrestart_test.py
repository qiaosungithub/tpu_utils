r"""Negative-controlled proof that auto warm-restart picks the RIGHT resume
mechanism per checkpoint layout -- restart_from+restart_step for ELT, load_from
for everyone else.

★WHY THIS TEST EXISTS. route_lib.build_warm_restart_entry is the live
translation point (route_check.run_reconcile calls it once plan_pruned_restart
returns RESUME_WARM) that turns a surviving checkpoint into a new queue row's
launch_kwargs. ELT/EqR-jax training MUST resume via restart_from(=workdir)+
restart_step(=N); handed $LOAD_FROM instead, main_eqr does `workdir = LOAD_FROM`
and read+writes the rescued dir, so orbax deletes the checkpoints it resumed
from (measured 2026-09-09: XID 288109183/288109952 died at step 0). Every other
fleet layout honours $LOAD_FROM. This test pins that split and the invariant
that the two mutually-exclusive mechanisms are NEVER both set.

★WHAT WOULD MAKE THIS TEST WORTHLESS. Only checking the ELT cell. A translator
that emitted restart_from for EVERYTHING would break every torch/paligemma
resume, so the non-ELT cells are positive assertions, and a negative control
proves the parser's verdict actually depends on the path shape (not a constant).

Banner: ELT_RESTART_OK (unique). Runs on CPU in milliseconds: route_lib is a
pure module with no I/O.

This is a SEPARATE test file on purpose -- route_lib_test.py is owned by the
adjacent scheduler-rewrite change; keeping the ELT-resume assertions here avoids
a merge collision while still testing the shared route_lib surface.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import route_lib as R  # noqa: E402  # pyrefly: ignore[missing-import]

_ELT_LEAF = ("/cns/li-d/home/qiaos/eqr_data/logs/elt-dit/"
             "xid_287351065_20260907_x/checkpoints/456000")
_ELT_WORKDIR = "/cns/li-d/home/qiaos/eqr_data/logs/elt-dit/xid_287351065_20260907_x"
_TORCH_LEAF = "/cns/is-d/home/qiaos/lyy_parcae_runs/parcae/steps/step_6144.pt"
_PALI_LEAF = "/cns/oi-d/home/qiaos/x/checkpoint_1000"


def _dead(**lk):
  """A minimal dead ELT-ish row; launch_kwargs overridden per case."""
  return R.QueueEntry(
      job_id="v7-32-dead", power="v7-32", allowed_archs=["v7", "v6p"],
      tier="PROD", allowed_metros=["lpp"], launch_kwargs=lk,
      xid="288109183", auto_resumes=0)


def check_inverse_parser():
  """_elt_restart_from_checkpoint recognises ONLY the ELT bare-int leaf."""
  assert R._elt_restart_from_checkpoint(_ELT_LEAF) == (_ELT_WORKDIR, 456000)
  assert R._elt_restart_from_checkpoint(_ELT_LEAF + "/") == (_ELT_WORKDIR, 456000)
  for other in (_TORCH_LEAF, _PALI_LEAF, "/cns/x/xid_y",
                "/cns/x/checkpoints/latest", "", None):
    assert R._elt_restart_from_checkpoint(other) is None, other
  print("  [1] inverse parser: ELT leaf -> (workdir, step); all others -> None  OK")


def check_elt_training_gets_restart_from():
  """ELT leaf -> restart_from+restart_step, and any stale load_from is dropped."""
  new = R.build_warm_restart_entry(
      _dead(config="prod_train_persite2_dw1", exp_name="elt_unroll_dw1",
            load_from="STALE_MUST_BE_DROPPED"),
      _ELT_LEAF, "v7-32-new")
  lk = new.launch_kwargs
  assert lk.get("restart_from") == _ELT_WORKDIR, lk
  assert lk.get("restart_step") == "456000", lk           # string, launcher-ready
  assert "load_from" not in lk, lk
  print("  [2] ELT training -> restart_from + restart_step, load_from cleared    OK")


def check_non_elt_gets_load_from():
  """A torch leaf -> load_from; any stale restart_* is dropped."""
  new = R.build_warm_restart_entry(
      _dead(config="configs/unroll_right.yml", exp_name="parcae",
            restart_from="STALE", restart_step="9"),
      _TORCH_LEAF, "h100-new")
  lk = new.launch_kwargs
  assert lk.get("load_from") == _TORCH_LEAF, lk
  assert "restart_from" not in lk and "restart_step" not in lk, lk
  print("  [3] non-ELT (torch) -> load_from, restart_* cleared                   OK")


def check_mutual_exclusion_invariant():
  """★The launcher trips main_eqr's guard if BOTH families of keys are set.

  Prove neither branch can ever emit both, across both layouts.
  """
  for ckpt in (_ELT_LEAF, _TORCH_LEAF, _PALI_LEAF):
    lk = R.build_warm_restart_entry(_dead(config="prod_train"), ckpt, "j").launch_kwargs
    both = ("load_from" in lk) and ("restart_from" in lk or "restart_step" in lk)
    assert not both, (ckpt, lk)
  print("  [4] mutual exclusion: never load_from AND restart_* together          OK")


def check_common_fields_preserved():
  """The restart is the SAME run: config/exp_name/workdir/prior_xids/budget."""
  dead = _dead(config="prod_train_8n4l_persite4_dw1", exp_name="elt_x")
  dead.workdir = "/some/checkout"
  new = R.build_warm_restart_entry(dead, _ELT_LEAF, "v7-32-new")
  assert new.launch_kwargs.get("config") == "prod_train_8n4l_persite4_dw1"
  assert new.workdir == "/some/checkout"
  assert new.auto_resumes == 1                     # budget incremented
  assert "288109183" in new.prior_xids             # dead xid recorded
  assert new.state == R.JobState.QUEUED
  print("  [5] clone preserves config/workdir, bumps budget, records prior xid   OK")


def check_negative_control_parser_not_constant():
  """The parser's verdict DEPENDS on the path shape (else cells 2-3 are vacuous)."""
  recognised = R._elt_restart_from_checkpoint(_ELT_LEAF) is not None
  rejected = R._elt_restart_from_checkpoint(_TORCH_LEAF) is None
  assert recognised and rejected, "parser must split ELT from non-ELT, not be constant"
  print("  [6] negative control: parser splits ELT vs non-ELT (not a constant)   OK")


def check_out_dir_from_log_reads_elt_banner():
  """Evidence layer must recover the ELT out_dir from main_eqr's boot banner.

  ELT names no launcher `out_dir (post-locality)` line; its durable workdir is
  only in `[main_eqr] TRAINING run: redirecting workdir -> $CHECKPOINT_BUCKET
  <path> ...`. Without parsing it the evidence layer finds no out_dir and every
  ELT auto-resume HOLDs. Negative-controlled: a non-ELT line must NOT match here.
  """
  banner = ("[main_eqr] TRAINING run: redirecting workdir -> $CHECKPOINT_BUCKET "
            + _ELT_WORKDIR + " (durable CNS; the launcher's --workdir /tmp/x is "
            "VM-local and wiped on restart).")
  assert R.out_dir_from_log(banner) == _ELT_WORKDIR, R.out_dir_from_log(banner)
  # still recovers the torch layout, and still None on a plain progress line
  assert R.out_dir_from_log(
      "checkpoint saved -> /cns/is-d/x/run/steps/step_10.pt") == "/cns/is-d/x/run"
  assert R.out_dir_from_log("[parcae-torch] step 1 loss 10.6") is None
  print("  [7] out_dir_from_log: reads ELT boot banner, torch still works        OK")


def check_elt_checkpoint_leaf_step():
  """Bare-int ELT leaf -> step; orbax tmp / prefixed / empty -> -1.

  This is what lets _latest_complete_checkpoint scan ELT's checkpoints/<N>/. The
  tmp case is the completeness guard: orbax renames <N>.orbax-checkpoint-tmp-*
  to <N> on finalize, so a tmp leaf is an in-flight save and must score -1.
  """
  assert R.elt_checkpoint_leaf_step("456000") == 456000
  assert R.elt_checkpoint_leaf_step("/cns/x/checkpoints/456000/") == 456000
  assert R.elt_checkpoint_leaf_step("456000.orbax-checkpoint-tmp-abc") == -1
  assert R.elt_checkpoint_leaf_step("step_10") == -1   # torch shape is not ELT's
  assert R.elt_checkpoint_leaf_step("") == -1
  print("  [8] elt_checkpoint_leaf_step: bare int -> step, tmp/prefixed -> -1     OK")


def check_evidence_to_restart_roundtrip():
  """★The closed loop: the leaf the evidence layer returns is exactly what the
  restart builder inverts. _latest_complete_checkpoint hands back
  `<out_dir>/checkpoints/<N>`; feeding THAT into build_warm_restart_entry must
  yield restart_from=<out_dir> + restart_step=N. If these two ever drift apart,
  ELT auto-resume silently falls back to load_from -- the original bug.
  """
  leaf = _ELT_WORKDIR + "/checkpoints/461000"
  lk = R.build_warm_restart_entry(
      _dead(config="prod_train", exp_name="elt_x"), leaf, "v7-32-new").launch_kwargs
  assert lk.get("restart_from") == _ELT_WORKDIR, lk
  assert lk.get("restart_step") == "461000", lk
  assert "load_from" not in lk, lk
  print("  [9] evidence leaf -> restart_from+restart_step round-trip             OK")


def main():
  print("=== eltrestart_test ===")
  check_inverse_parser()
  check_elt_training_gets_restart_from()
  check_non_elt_gets_load_from()
  check_mutual_exclusion_invariant()
  check_common_fields_preserved()
  check_negative_control_parser_not_constant()
  check_out_dir_from_log_reads_elt_banner()
  check_elt_checkpoint_leaf_step()
  check_evidence_to_restart_roundtrip()
  print("ELT_RESTART_OK")


if __name__ == "__main__":
  main()
