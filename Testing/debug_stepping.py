"""Debug: Show which channels are visible at each step direction."""

from point_counter import FlatTargetHitCounter
import numpy as np

# Setup
counter = FlatTargetHitCounter(1.8, 1.3, 10, 0, 0, np.radians(45)/180, np.radians(30)/127)
counter.set_calibration(0, np.radians(-11.281))

print("="*70)
print("STEPPING ANALYSIS AT 10m")
print("="*70)
print(f"Target: 1.3m tall at 10m = {np.degrees(np.arctan(1.3/10)):.2f}° vertical span")
print(f"Sensor FOV: 30° (much larger than target!)")
print(f"At calibration: highest beams (channel 0) point at top of target")
print()

macro_step = np.radians(7.28)

print("DOWNWARD stepping (negative angles):")
print("-" * 70)
for step in range(6):
    dtheta_rel = -step * macro_step
    dphi_abs, dtheta_abs = counter.relative_to_absolute_offset(0, dtheta_rel)
    
    X, Y, _, _ = counter.project_with_angles(dphi_abs, dtheta_abs)
    mask = counter.inside_mask(X, Y)
    
    if np.any(mask):
        beam_idx = np.where(mask)[0]
        channels = sorted(set((7 - beam_idx // 16).tolist()))
        print(f"  Step {step} (rel={np.degrees(dtheta_rel):+6.2f}°, abs={np.degrees(dtheta_abs):+6.2f}°): "
              f"beams {beam_idx.min():3d}-{beam_idx.max():3d} → channels {channels}")
    else:
        print(f"  Step {step} (rel={np.degrees(dtheta_rel):+6.2f}°, abs={np.degrees(dtheta_abs):+6.2f}°): "
              f"NO HITS (FOV moved below target)")

print()
print("UPWARD stepping (positive angles):")
print("-" * 70)
for step in range(1, 6):
    dtheta_rel = +step * macro_step
    dphi_abs, dtheta_abs = counter.relative_to_absolute_offset(0, dtheta_rel)
    
    X, Y, _, _ = counter.project_with_angles(dphi_abs, dtheta_abs)
    mask = counter.inside_mask(X, Y)
    
    if np.any(mask):
        beam_idx = np.where(mask)[0]
        channels = sorted(set((7 - beam_idx // 16).tolist()))
        print(f"  Step {step} (rel={np.degrees(dtheta_rel):+6.2f}°, abs={np.degrees(dtheta_abs):+6.2f}°): "
              f"beams {beam_idx.min():3d}-{beam_idx.max():3d} → channels {channels}")
    else:
        print(f"  Step {step} (rel={np.degrees(dtheta_rel):+6.2f}°, abs={np.degrees(dtheta_abs):+6.2f}°): "
              f"NO HITS (FOV moved above target)")

print()
print("="*70)
print("KEY INSIGHT:")
print("="*70)
print("The sensor FOV (30°) is MUCH larger than the target (7.44°).")
print("Channels 0-2 need the sensor tilted DOWN (to point high beams at target)")
print("Channels 5-7 need the sensor tilted UP (to point low beams at target)")
print("Channels 3-4 are visible around calibration position.")
print("="*70)
