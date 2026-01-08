"""
SIMPLE VERIFICATION: Do we have 300 samples per channel with equal elevation distribution?
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

# Load sample data
samples = []
with open('sample_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ch = int(row['channel'])
        elev = int(row['elevation'])
        samples.append((ch, elev))

# Count per channel and elevation
counts = np.zeros((8, 16), dtype=int)
for ch, elev in samples:
    counts[ch, elev] += 1

# Print summary
print("\n" + "="*80)
print("VERIFICATION RESULTS")
print("="*80)
for ch in range(8):
    total = counts[ch].sum()
    min_val = counts[ch].min()
    max_val = counts[ch].max()
    print(f"Channel {ch}: Total={total:3d} | Min={min_val:2d} | Max={max_val:2d} | Target=300")

print("\n" + "="*80)
print("VISUAL VERIFICATION")
print("="*80)

# Create simple plots
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.ravel()

colors = ['red', 'orange', 'gold', 'green', 'cyan', 'blue', 'purple', 'magenta']

for ch in range(8):
    ax = axes[ch]
    
    # Bar chart for this channel
    ax.bar(range(16), counts[ch], color=colors[ch], edgecolor='black', linewidth=1.5)
    ax.axhline(y=19, color='green', linestyle='--', linewidth=2, label='Target (19)')
    ax.set_ylim(0, 25)
    ax.set_xlabel('Elevation (0-15)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
    ax.set_title(f'CHANNEL {ch}', fontsize=14, fontweight='bold', color=colors[ch])
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(range(16))
    
    # Add total in corner
    total = counts[ch].sum()
    ax.text(0.98, 0.98, f'Total: {total}', transform=ax.transAxes,
            fontsize=14, fontweight='bold', verticalalignment='top', 
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=2))
    
    if ch == 0:
        ax.legend(loc='upper left', fontsize=10)

plt.suptitle('SAMPLE DISTRIBUTION: 8 Channels × 16 Elevations\nAll bars should be ~19 samples', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('VERIFY_COVERAGE.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization: VERIFY_COVERAGE.png")
print("\nCheck the plot - every elevation should have ~19 samples (green line)")
print("="*80 + "\n")

plt.show()
