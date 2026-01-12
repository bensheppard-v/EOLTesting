"""
Simple visualization of actual sample data collected by autofill.
Reads sample_data.csv and creates clear visualizations.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load sample data
samples = []
with open('sample_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        x = float(row['x_m'])
        y = float(row['y_m'])
        ch = int(row['channel'])
        elev = int(row['elevation'])
        samples.append((x, y, ch, elev))

print(f"Loaded {len(samples)} samples")

# ============================================================================
# PLOT 1: Channel×Elevation Heatmap
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(14, 7))

# Count samples per (channel, elevation)
counts = np.zeros((8, 16), dtype=int)
for _, _, ch, elev in samples:
    counts[ch, elev] += 1

im = ax1.imshow(counts, aspect='auto', cmap='YlOrRd', interpolation='nearest')
cbar = plt.colorbar(im, ax=ax1)
cbar.set_label('Sample Count', rotation=270, labelpad=20)

ax1.set_xlabel('Elevation (0-15)', fontsize=12)
ax1.set_ylabel('Channel (0-7)', fontsize=12)
ax1.set_title('Sample Distribution: 8 Channels × 16 Elevations\n(All cells should show 19-21 samples)', fontsize=14, fontweight='bold')

ax1.set_xticks(range(16))
ax1.set_yticks(range(8))

# Add text annotations
for ch in range(8):
    for elev in range(16):
        color = 'white' if counts[ch, elev] > 40 else 'black'
        ax1.text(elev, ch, f'{counts[ch, elev]}',
                ha="center", va="center", color=color, fontsize=9, fontweight='bold')

plt.tight_layout()
fig1.savefig('plot_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ Saved: plot_heatmap.png")

# ============================================================================
# PLOT 2: Per-Channel Histograms
# ============================================================================
fig2, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

channel_colors = ['red', 'orange', 'gold', 'green', 'cyan', 'blue', 'purple', 'magenta']

for ch in range(8):
    ax = axes[ch]
    
    # Get samples for this channel
    elevations = [elev for _, _, c, elev in samples if c == ch]
    
    # Count per elevation
    counts_ch = [elevations.count(e) for e in range(16)]
    
    # Plot bar chart
    ax.bar(range(16), counts_ch, color=channel_colors[ch], alpha=0.8, edgecolor='black')
    ax.set_title(f'Channel {ch}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Elevation', fontsize=10)
    ax.set_ylabel('Sample Count', fontsize=10)
    ax.set_xticks(range(16))
    ax.set_ylim(0, max(counts_ch) * 1.1 if counts_ch else 25)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=19, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target (19)')
    
    # Add stats
    total = sum(counts_ch)
    min_val = min(counts_ch) if counts_ch else 0
    max_val = max(counts_ch) if counts_ch else 0
    avg_val = total / 16 if total > 0 else 0
    ax.text(0.98, 0.95, f'Total={total}\nMin={min_val}\nMax={max_val}\nAvg={avg_val:.1f}',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    if ch == 0:
        ax.legend(loc='upper left', fontsize=8)

plt.suptitle('Sample Distribution per Channel\n(Each elevation should have ~19-21 samples)', 
             fontsize=15, fontweight='bold')
plt.tight_layout()
fig2.savefig('plot_histograms.png', dpi=150, bbox_inches='tight')
print("✓ Saved: plot_histograms.png")

# ============================================================================
# PLOT 3: Target Surface
# ============================================================================
fig3, ax3 = plt.subplots(figsize=(14, 10))

# Draw target outline
target_width = 1.8
target_height = 1.3
target = Rectangle((-target_width/2, -target_height/2), 
                   target_width, target_height, 
                   fill=False, edgecolor='black', linewidth=3)
ax3.add_patch(target)

# Plot each channel with distinct color and elevation-based brightness
for ch in range(8):
    ch_samples = [(x, y, elev) for x, y, c, elev in samples if c == ch]
    if not ch_samples:
        continue
    
    xs = [s[0] for s in ch_samples]
    ys = [s[1] for s in ch_samples]
    elevs = [s[2] for s in ch_samples]
    
    # Elevation controls brightness: 0=darker, 15=lighter
    alphas = [0.4 + 0.6 * (e / 15.0) for e in elevs]
    
    for x, y, alpha in zip(xs, ys, alphas):
        ax3.scatter(x, y, c=channel_colors[ch], s=20, alpha=alpha, edgecolors='none')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=channel_colors[ch], label=f'Ch {ch}') for ch in range(8)]
ax3.legend(handles=legend_elements, loc='upper right', title='Channels', fontsize=10)

ax3.set_xlabel('X Position (m)', fontsize=12)
ax3.set_ylabel('Y Position (m)', fontsize=12)
ax3.set_title('Target Surface Coverage\n(Color = Channel, Brightness = Elevation within channel)', 
             fontsize=14, fontweight='bold')
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
fig3.savefig('plot_target_surface.png', dpi=150, bbox_inches='tight')
print("✓ Saved: plot_target_surface.png")

print("\n" + "="*80)
print("✓ All visualizations complete!")
print("="*80)

plt.show()
