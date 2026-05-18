# CALVIN 0% Success Pivot

## Summary

The current official-protocol results show both `frozen_openvla` and `ce_ais` at 0% on CALVIN. This should be interpreted as an action-prior mismatch, not as direct evidence that CE-AIS is ineffective.

OpenVLA is an off-the-shelf VLA trained with non-CALVIN action conventions. CALVIN expects native 7D `rel_actions` with its own coordinate frame, scale, and gripper convention. CE-AIS steers candidate actions in action space; if the base policy never enters task-relevant action regions, the steering module has no useful prior to refine.

## Diagnosis

- OpenVLA is not a CALVIN-native policy.
- CALVIN long-horizon evaluation expects actions compatible with its native `rel_actions` format.
- CE-AIS assumes a plausible base action proposal and is not designed to solve arbitrary action-space remapping from a failed zero-shot policy.
- The expert replay sanity check is the right diagnostic for environment/oracle/action plumbing: replayed validation `rel_actions` should solve the corresponding annotated task if reset and oracle logic are correct.

## Revised paper framing

Use CE-AIS as a gradient-free test-time steering module for action-compatible VLA or imitation policies, not as a universal converter for mismatched action spaces.

Recommended framing:

1. Main comparison: CALVIN-trained frozen policy vs. CALVIN-trained policy + CE-AIS.
2. Robustness focus: clean, visual OOD, physics OOD, and recovery settings after the frozen CALVIN policy has nonzero success.
3. OpenVLA zero-shot CALVIN: report only as a weak off-the-shelf transfer diagnostic, not as the main result.
4. Claims should be tied to measurable settings: success rate, average completed subtasks, recovery after perturbation, and latency.

## Implementation roadmap

1. Add a CALVIN policy adapter that loads a provided CALVIN-native checkpoint or fails clearly if none is provided.
2. Preserve raw CALVIN observations in the wrapper so official CALVIN policies can consume the expected observation format.
3. Reset the wrapped policy at each subtask boundary, matching the official CALVIN evaluation protocol.
4. Add an expert replay diagnostic using validation `rel_actions` to verify reset, oracle, and action plumbing.
5. Run CE-AIS only after a frozen CALVIN-native base policy reaches nonzero clean success.

## Experimental decision tree

- Expert replay fails: fix environment reset, oracle, task annotation, or action replay plumbing.
- Expert replay succeeds but no CALVIN policy checkpoint is available: train or download a CALVIN-native policy before making main paper claims.
- Frozen CALVIN policy succeeds but CE-AIS fails: debug CE-WM energy calibration, action scale, steering step size, and gating.
- CE-AIS matches clean success and improves OOD/recovery: this supports the paper direction.

## Recommended next experiments

1. Run expert replay on several validation language segments.
2. Obtain a CALVIN-native policy checkpoint and run `frozen_calvin_policy` on official chains.
3. Use the same checkpoint as the CE-AIS base policy with `--vla-type calvin`.
4. Compare clean and OOD settings only after the frozen base policy is validated.
