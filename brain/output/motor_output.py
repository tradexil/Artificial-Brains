"""Motor output pipeline: decode motor neuron activations into action decisions.

Reads the motor region (130000–139999) to determine what the brain
wants to do: approach, withdraw, idle, or conflict.

Motor sub-populations:
  130000–134999: approach (move toward, grab, engage)
  135000–137999: withdraw excitatory (retreat, release, avoid)
  138000–139999: inhibitory (regulation)
"""

from __future__ import annotations

from dataclasses import dataclass

import brain_core


@dataclass
class MotorAction:
    """Decoded motor action from brain state."""
    action_type: str     # "idle", "approach", "withdraw", "conflict"
    approach: float      # approach strength 0.0–1.0
    withdraw: float      # withdraw strength 0.0–1.0
    peak_neurons: list[tuple[int, float]]  # top active motor neurons
    motor_activation: float  # overall motor activity level


class MotorOutput:
    """Decodes motor region activations into action decisions.

    Usage:
        decoder = MotorOutput()
        action = decoder.read()
        print(action.action_type, action.approach)
    """

    def __init__(self, top_k: int = 20, suppression: float = 0.7):
        self.top_k = top_k
        self.suppression = suppression

    def read(self, apply_inhibition: bool = True) -> MotorAction:
        """Read current motor state and decode into action.

        Args:
            apply_inhibition: if True, apply lateral inhibition before reading.

        Returns:
            MotorAction with decoded action type and strengths.
        """
        # Optionally apply lateral inhibition to sharpen
        if apply_inhibition:
            brain_core.motor_lateral_inhibition(self.suppression)

        # Decode motor action via Rust
        action_type, approach, withdraw = brain_core.decode_motor_action()

        # Read peak neurons
        peaks = brain_core.get_peak_motor_neurons(self.top_k)
        motor_activation = brain_core.get_motor_activation()

        return MotorAction(
            action_type=action_type,
            approach=approach,
            withdraw=withdraw,
            peak_neurons=peaks,
            motor_activation=motor_activation,
        )

    def read_raw(self, top_k: int = 50) -> list[tuple[int, float]]:
        """Read raw peak motor neuron activations without decoding."""
        return brain_core.get_peak_motor_neurons(top_k)

    def get_approach_withdraw(self) -> tuple[float, float]:
        """Get raw approach/withdraw strengths without inhibition."""
        return brain_core.get_approach_vs_withdraw()
