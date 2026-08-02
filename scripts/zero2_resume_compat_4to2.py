"""Compatibility helpers for restoring a 4-rank ZeRO-2 checkpoint on 2 GPUs.

The source checkpoint is a legacy ZeRO-2 checkpoint produced with four data
parallel ranks. DeepSpeed's elastic loader can repartition its optimizer
state, but older checkpoints need their flattened optimizer state converted to
the elastic representation first.
"""

from typing import Any, Dict, List


def _as_int(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _convert_legacy_zero2_state(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one legacy ZeRO-2 optimizer shard to elastic format."""

    base_state = state_dict.get("base_optimizer_state")
    if not isinstance(base_state, dict) or "param_groups" not in base_state:
        return state_dict

    parameter_states = base_state.get("state", {})
    group_states: List[Dict[str, Any]] = []
    for group in base_state["param_groups"]:
        parameter_ids = group.get("params", [])
        if len(parameter_ids) != 1:
            raise ValueError(
                "Expected one flattened parameter per ZeRO-2 optimizer group; "
                f"got {parameter_ids}"
            )
        parameter_id = parameter_ids[0]
        if parameter_id not in parameter_states:
            raise ValueError(f"Missing optimizer state for flattened parameter {parameter_id}")
        group_states.append(parameter_states[parameter_id])

    if not group_states:
        raise ValueError("The ZeRO-2 optimizer state contains no parameter groups")
    steps = {_as_int(group_state["step"]) for group_state in group_states}
    if len(steps) != 1:
        raise ValueError(f"ZeRO-2 optimizer groups disagree on step: {sorted(steps)}")

    converted = dict(state_dict)
    converted["base_optimizer_state"] = group_states
    converted["base_optimizer_state_step"] = steps.pop()
    return converted


def install_zero2_4to2_resize_compat() -> None:
    """Enable the specific 4-rank ZeRO-2 -> 2-rank restore path."""

    import deepspeed
    from deepspeed.runtime.engine import DeepSpeedEngine
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    del deepspeed

    if getattr(DeepSpeedEngine, "_fragllm_4to2_resize", False):
        return

    original_load_zero_checkpoint = DeepSpeedEngine._load_zero_checkpoint
    original_load_legacy_checkpoint = DeepSpeedZeroOptimizer._load_legacy_checkpoint

    def load_zero_checkpoint_with_resize(
        self, load_dir, tag, load_optimizer_states=True
    ):
        saved_world_size = self.loaded_checkpoint_dp_world_size
        current_world_size = self.seq_dp_world_size
        if (
            load_optimizer_states
            and saved_world_size is not None
            and current_world_size != saved_world_size
        ):
            if (saved_world_size, current_world_size) != (4, 2):
                raise RuntimeError(
                    "This compatibility layer only supports 4-rank ZeRO-2 "
                    f"checkpoints restored on 2 ranks, got {saved_world_size}->{current_world_size}"
                )
            # Keep the saved DP size while DeepSpeed discovers all source
            # shards; restore the live DP size immediately afterwards.
            self.seq_dp_world_size = saved_world_size
            try:
                return original_load_zero_checkpoint(
                    self,
                    load_dir,
                    tag,
                    load_optimizer_states=load_optimizer_states,
                )
            finally:
                self.seq_dp_world_size = current_world_size

        return original_load_zero_checkpoint(
            self,
            load_dir,
            tag,
            load_optimizer_states=load_optimizer_states,
        )

    def load_legacy_checkpoint_as_elastic(
        self,
        state_dict_list,
        load_optimizer_states=True,
        load_from_fp32_weights=False,
    ):
        if self.elastic_checkpoint:
            state_dict_list = [
                _convert_legacy_zero2_state(state_dict)
                for state_dict in state_dict_list
            ]
        return original_load_legacy_checkpoint(
            self,
            state_dict_list,
            load_optimizer_states=load_optimizer_states,
            load_from_fp32_weights=load_from_fp32_weights,
        )

    DeepSpeedEngine._load_zero_checkpoint = load_zero_checkpoint_with_resize
    DeepSpeedZeroOptimizer._load_legacy_checkpoint = load_legacy_checkpoint_as_elastic
    DeepSpeedEngine._fragllm_4to2_resize = True
    print(
        "[posttrain-2gpu] Installed ZeRO-2 4-rank -> 2-rank "
        "in-memory optimizer repartition compatibility"
    )
