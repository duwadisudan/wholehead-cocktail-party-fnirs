"""
Per-subject MNI registration error vs the Brodmann template.

Loads the template MNI optode positions and the subject-specific scanner-
registered MNI positions, computes the per-channel Euclidean registration
error per subject, renders 2D and 3D scalp visualisations of the error
magnitude, and reports cross-subject variability statistics.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Code refactoring, documentation, and commenting were AI-assisted;
       all scientific decisions and accountability remain with the author.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import List, Optional
import pyvista as pv

from wholehead_cocktail_party.paths import load_paths, require

_PATHS = load_paths()
require(_PATHS, "raw_root", "roi_csv")

# CONFIGURATION - MODIFY THESE

# List of subject IDs to include in analysis
SUBJECT_IDS = [
    "sub-630",
    "sub-633",
    "sub-635",
    "sub-640",
    "sub-649",
    "sub-650",
    "sub-651",
    "sub-653",
    "sub-660",
    "sub-663",
]

# Base directory for data
BASE_DIR = _PATHS.raw_root

# Template MNI file (ground truth - where channels SHOULD be).
# Lives next to roi_master.csv in the ROIs/ folder.
TEMPLATE_MNI_PATH = _PATHS.roi_csv.parent / "roi_coor_system_2_w_mni.csv"

# Output directory
OUTPUT_DIR = BASE_DIR / "derivatives" / "mni_registration_error_analysis"

# Subject MNI filename (no header, columns: x, y, z)
SUBJECT_MNI_FILENAME = "just_mni.csv"

# Number of channels expected
N_CHANNELS = 567

# END CONFIGURATION


def load_template_mni(template_path: Path) -> np.ndarray:
    """Load template MNI coordinates (ground truth positions).
    
    Returns: (N_CHANNELS, 3) array of [x, y, z] coordinates
    """
    print(f"\nLoading template MNI from: {template_path}")
    df = pd.read_csv(template_path)
    
    # Extract MNI columns (MNIX, MNIY, MNIZ)
    template_coords = df[['MNIX', 'MNIY', 'MNIZ']].values
    print(f"  Loaded {len(template_coords)} template channel positions")
    
    return template_coords


def load_subject_mni(subject_id: str, base_dir: Path) -> Optional[np.ndarray]:
    """Load subject-specific registered MNI coordinates.
    
    Returns: (N_CHANNELS, 3) array of [x, y, z] coordinates, or None if file not found
    """
    csv_path = base_dir / subject_id / "nirs" / "atlasviewer_mni" / SUBJECT_MNI_FILENAME
    
    if not csv_path.exists():
        print(f"  WARNING: File not found for {subject_id}: {csv_path}")
        return None
    
    try:
        # No header, columns are x, y, z
        df = pd.read_csv(csv_path, header=None, names=['x', 'y', 'z'])
        coords = df[['x', 'y', 'z']].values
        
        if len(coords) != N_CHANNELS:
            print(f"  WARNING: {subject_id} has {len(coords)} channels, expected {N_CHANNELS}")
        
        print(f"  Loaded {len(coords)} channels from {subject_id}")
        return coords
        
    except Exception as e:
        print(f"  ERROR loading {csv_path}: {e}")
        return None


def load_all_subjects(subject_ids: List[str], base_dir: Path) -> dict:
    """Load MNI coordinates from all subjects.
    
    Returns: dict mapping subject_id -> (N_CHANNELS, 3) array
    """
    print("\nLoading subject data...")
    
    subject_data = {}
    for subject_id in subject_ids:
        coords = load_subject_mni(subject_id, base_dir)
        if coords is not None:
            subject_data[subject_id] = coords
    
    if not subject_data:
        raise ValueError("No data loaded from any subject!")
    
    print(f"\nLoaded data from {len(subject_data)} subjects")
    return subject_data


def compute_registration_errors(template: np.ndarray, subject_data: dict) -> pd.DataFrame:
    """Compute per-channel, per-subject registration error.
    
    Error = Euclidean distance from subject position to template position
    
    Returns DataFrame with columns:
    - channel_idx: channel index (0 to N_CHANNELS-1)
    - subject: subject ID
    - error: Euclidean distance (mm)
    - template_x/y/z: template position
    - subject_x/y/z: subject position
    - delta_x/y/z: difference in each axis
    """
    print("\nComputing registration errors...")
    
    records = []
    for subject_id, subject_coords in subject_data.items():
        n_channels = min(len(template), len(subject_coords))
        
        for ch_idx in range(n_channels):
            t = template[ch_idx]
            s = subject_coords[ch_idx]
            delta = s - t
            error = np.linalg.norm(delta)
            
            records.append({
                'channel_idx': ch_idx,
                'subject': subject_id,
                'error': error,
                'template_x': t[0],
                'template_y': t[1],
                'template_z': t[2],
                'subject_x': s[0],
                'subject_y': s[1],
                'subject_z': s[2],
                'delta_x': delta[0],
                'delta_y': delta[1],
                'delta_z': delta[2],
            })
    
    errors_df = pd.DataFrame(records)
    
    print(f"\nRegistration Error Summary (mm):")
    print(f"  Overall Mean:  {errors_df['error'].mean():.2f}")
    print(f"  Overall Std:   {errors_df['error'].std():.2f}")
    print(f"  Overall Min:   {errors_df['error'].min():.2f}")
    print(f"  Overall Max:   {errors_df['error'].max():.2f}")
    
    return errors_df


def compute_channel_statistics(errors_df: pd.DataFrame, template: np.ndarray) -> pd.DataFrame:
    """Compute per-channel statistics across all subjects.
    
    Returns DataFrame with per-channel summary statistics.
    """
    print("\nComputing per-channel statistics...")
    
    stats_list = []
    for ch_idx, group in errors_df.groupby('channel_idx'):
        stats_list.append({
            'channel_idx': ch_idx,
            'n_subjects': len(group),
            'mean_error': group['error'].mean(),
            'std_error': group['error'].std(),
            'min_error': group['error'].min(),
            'max_error': group['error'].max(),
            'template_x': template[ch_idx, 0],
            'template_y': template[ch_idx, 1],
            'template_z': template[ch_idx, 2],
            # Mean subject position
            'mean_subject_x': group['subject_x'].mean(),
            'mean_subject_y': group['subject_y'].mean(),
            'mean_subject_z': group['subject_z'].mean(),
            # Variability (spread) across subjects
            'std_subject_x': group['subject_x'].std(),
            'std_subject_y': group['subject_y'].std(),
            'std_subject_z': group['subject_z'].std(),
        })
    
    stats_df = pd.DataFrame(stats_list)
    
    print(f"\nPer-Channel Error Summary (mm):")
    print(f"  Mean of mean errors:  {stats_df['mean_error'].mean():.2f}")
    print(f"  Worst channel error:  {stats_df['mean_error'].max():.2f} (channel {stats_df.loc[stats_df['mean_error'].idxmax(), 'channel_idx']})")
    print(f"  Best channel error:   {stats_df['mean_error'].min():.2f} (channel {stats_df.loc[stats_df['mean_error'].idxmin(), 'channel_idx']})")
    
    return stats_df


def compute_subject_statistics(errors_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-subject statistics across all channels.
    
    Returns DataFrame with per-subject summary (which subjects have worst fit).
    """
    print("\nComputing per-subject statistics...")
    
    stats_list = []
    for subject_id, group in errors_df.groupby('subject'):
        stats_list.append({
            'subject': subject_id,
            'n_channels': len(group),
            'mean_error': group['error'].mean(),
            'std_error': group['error'].std(),
            'min_error': group['error'].min(),
            'max_error': group['error'].max(),
            'median_error': group['error'].median(),
        })
    
    stats_df = pd.DataFrame(stats_list).sort_values('mean_error', ascending=False)
    
    print(f"\nPer-Subject Error Summary (mm):")
    for _, row in stats_df.iterrows():
        print(f"  {row['subject']}: mean={row['mean_error']:.2f}, max={row['max_error']:.2f}")
    
    return stats_df


# VISUALIZATION FUNCTIONS

def plot_2d_error_map(channel_stats: pd.DataFrame, output_path: Path) -> None:
    """Create 2D scatter plot showing registration error per channel.
    
    Similar to scalp plot style in photogrammetric code.
    """
    print("\nGenerating 2D error map...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    x = channel_stats['template_x'].values
    y = channel_stats['template_y'].values
    error = channel_stats['mean_error'].values
    
    # Plot 1: Template positions colored by mean error
    ax1 = axes[0]
    sc1 = ax1.scatter(x, y, c=error, s=50, cmap='jet', alpha=0.8,
                      edgecolors='black', linewidths=0.3)
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Mean Registration Error (mm)', fontsize=12)
    
    ax1.set_xlabel('MNI X (mm)', fontsize=12)
    ax1.set_ylabel('MNI Y (mm)', fontsize=12)
    ax1.set_title('Channel Registration Error\n(Template Positions)', fontsize=14)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Top-down view (X vs Y) with Z as color intensity overlay
    ax2 = axes[1]
    z = channel_stats['template_z'].values
    sc2 = ax2.scatter(x, y, c=error, s=50, cmap='plasma', alpha=0.8,
                      edgecolors='black', linewidths=0.3,
                      vmin=0, vmax=np.percentile(error, 95))
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label('Mean Registration Error (mm)', fontsize=12)
    
    ax2.set_xlabel('MNI X (mm)', fontsize=12)
    ax2.set_ylabel('MNI Y (mm)', fontsize=12)
    ax2.set_title('Registration Error (95th percentile clipped)', fontsize=14)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_error_by_axis(channel_stats: pd.DataFrame, output_path: Path) -> None:
    """Create 2D plots showing error contribution from each axis."""
    print("\nGenerating axis-wise error plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x = channel_stats['template_x'].values
    y = channel_stats['template_y'].values
    
    # Compute mean absolute delta per channel (need to recalculate from errors_df)
    # For now, use std as proxy for axis contribution
    for ax, (col, title) in zip(axes, [
        ('std_subject_x', 'Variability in X'),
        ('std_subject_y', 'Variability in Y'),
        ('std_subject_z', 'Variability in Z'),
    ]):
        vals = channel_stats[col].values
        sc = ax.scatter(x, y, c=vals, s=50, cmap='plasma', alpha=0.8,
                        edgecolors='black', linewidths=0.3)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(f'{title} (mm)', fontsize=10)
        
        ax.set_xlabel('MNI X (mm)', fontsize=12)
        ax.set_ylabel('MNI Y (mm)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_error_histograms(errors_df: pd.DataFrame, channel_stats: pd.DataFrame,
                          subject_stats: pd.DataFrame, output_path: Path) -> None:
    """Create histograms summarizing error distributions."""
    print("\nGenerating error histograms...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Overall error distribution
    ax1 = axes[0, 0]
    ax1.hist(errors_df['error'], bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(errors_df['error'].mean(), color='red', linestyle='--',
                label=f"Mean: {errors_df['error'].mean():.2f} mm")
    ax1.axvline(errors_df['error'].median(), color='orange', linestyle=':',
                label=f"Median: {errors_df['error'].median():.2f} mm")
    ax1.set_xlabel('Registration Error (mm)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of All Registration Errors', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Per-channel mean error distribution
    ax2 = axes[0, 1]
    ax2.hist(channel_stats['mean_error'], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax2.axvline(channel_stats['mean_error'].mean(), color='red', linestyle='--',
                label=f"Mean: {channel_stats['mean_error'].mean():.2f} mm")
    ax2.set_xlabel('Mean Error per Channel (mm)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Per-Channel Mean Errors', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Per-subject mean error (bar chart)
    ax3 = axes[1, 0]
    subjects = subject_stats['subject'].values
    mean_errors = subject_stats['mean_error'].values
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(subjects)))
    ax3.barh(subjects, mean_errors, color=colors, edgecolor='black')
    ax3.set_xlabel('Mean Registration Error (mm)', fontsize=12)
    ax3.set_ylabel('Subject', fontsize=12)
    ax3.set_title('Mean Error by Subject', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Top 20 worst channels
    ax4 = axes[1, 1]
    worst20 = channel_stats.nlargest(20, 'mean_error')
    ax4.barh(range(20), worst20['mean_error'].values, edgecolor='black', color='salmon')
    ax4.set_yticks(range(20))
    ax4.set_yticklabels([f"Ch {int(idx)}" for idx in worst20['channel_idx'].values])
    ax4.set_xlabel('Mean Registration Error (mm)', fontsize=12)
    ax4.set_ylabel('Channel', fontsize=12)
    ax4.set_title('Top 20 Channels with Highest Error', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_3d_error_pyvista(channel_stats: pd.DataFrame, errors_df: pd.DataFrame,
                          output_path: Path) -> None:
    """Create 3D visualization showing template vs subject positions with error.
    
    Uses pyvista similar to photogrammetric code visualization.
    """
    print("\nGenerating 3D error visualization (pyvista)...")
    
    # Get template positions and errors
    template_points = channel_stats[['template_x', 'template_y', 'template_z']].values
    mean_errors = channel_stats['mean_error'].values
    
    # Create colormap normalization
    norm = mcolors.Normalize(vmin=mean_errors.min(), vmax=np.percentile(mean_errors, 95))
    cmap = plt.cm.jet
    
    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=[1400, 1000])
    
    # Add template points as spheres colored by error
    for i, (point, error) in enumerate(zip(template_points, mean_errors)):
        sphere = pv.Sphere(radius=2.0, center=point)
        color = cmap(norm(error))[:3]
        plotter.add_mesh(sphere, color=color, smooth_shading=True)
    
    # Add colorbar using dummy mesh
    dummy = pv.PolyData(np.array([[0, 0, 0]]))
    dummy["error"] = [0]
    plotter.add_mesh(
        dummy,
        scalars="error",
        cmap='jet',
        scalar_bar_args={'title': 'Mean Error (mm)', 'title_font_size': 16, 'label_font_size': 14},
        clim=(mean_errors.min(), np.percentile(mean_errors, 95)),
        show_scalar_bar=True,
        opacity=0,
    )
    
    plotter.add_axes()
    plotter.camera_position = 'yz'
    
    # Save screenshot
    plotter.screenshot(str(output_path))
    plotter.close()
    print(f"  Saved: {output_path}")
    
    # Save additional views
    for view, suffix in [('xy', 'top'), ('xz', 'front'), ('iso', 'iso')]:
        plotter2 = pv.Plotter(off_screen=True, window_size=[1400, 1000])
        
        for point, error in zip(template_points, mean_errors):
            sphere = pv.Sphere(radius=2.0, center=point)
            color = cmap(norm(error))[:3]
            plotter2.add_mesh(sphere, color=color, smooth_shading=True)
        
        plotter2.add_mesh(
            dummy, scalars="error", cmap='jet',
            scalar_bar_args={'title': 'Mean Error (mm)'},
            clim=(mean_errors.min(), np.percentile(mean_errors, 95)),
            show_scalar_bar=True, opacity=0,
        )
        plotter2.add_axes()
        plotter2.camera_position = view
        
        view_path = output_path.with_stem(f"{output_path.stem}_{suffix}")
        plotter2.screenshot(str(view_path))
        plotter2.close()
        print(f"  Saved: {view_path}")


def plot_3d_template_vs_subjects(template: np.ndarray, subject_data: dict,
                                  channel_stats: pd.DataFrame, output_path: Path) -> None:
    """Create 3D plot showing template (black) and all subject positions (colored by error).
    
    This shows the "cloud" of subject positions around each template point.
    """
    print("\nGenerating 3D template vs subjects plot...")
    
    mean_errors = channel_stats['mean_error'].values
    norm = mcolors.Normalize(vmin=mean_errors.min(), vmax=np.percentile(mean_errors, 95))
    cmap = plt.cm.jet
    
    plotter = pv.Plotter(off_screen=True, window_size=[1400, 1000])
    
    # Add template points (black, larger)
    for point in template:
        sphere = pv.Sphere(radius=2.5, center=point)
        plotter.add_mesh(sphere, color='black', smooth_shading=True, opacity=0.8)
    
    # Add subject points (smaller, colored by that channel's mean error)
    for subject_id, coords in subject_data.items():
        for ch_idx, point in enumerate(coords):
            if ch_idx < len(mean_errors):
                sphere = pv.Sphere(radius=1.2, center=point)
                color = cmap(norm(mean_errors[ch_idx]))[:3]
                plotter.add_mesh(sphere, color=color, smooth_shading=True, opacity=0.4)
    
    # Add colorbar
    dummy = pv.PolyData(np.array([[0, 0, 0]]))
    dummy["error"] = [0]
    plotter.add_mesh(
        dummy, scalars="error", cmap='jet',
        scalar_bar_args={'title': 'Mean Error (mm)'},
        clim=(mean_errors.min(), np.percentile(mean_errors, 95)),
        show_scalar_bar=True, opacity=0,
    )
    
    plotter.add_axes()
    plotter.camera_position = 'yz'
    
    plotter.screenshot(str(output_path))
    plotter.close()
    print(f"  Saved: {output_path}")


def plot_3d_error_vectors(template: np.ndarray, subject_data: dict,
                          output_path: Path, subject_to_show: str = None) -> None:
    """Create 3D plot showing error vectors from template to subject positions.
    
    If subject_to_show is specified, show only that subject's vectors.
    Otherwise, show mean displacement vectors.
    """
    print("\nGenerating 3D error vectors plot...")
    
    plotter = pv.Plotter(off_screen=True, window_size=[1400, 1000])
    
    if subject_to_show and subject_to_show in subject_data:
        # Show vectors for one specific subject
        subject_coords = subject_data[subject_to_show]
        n_channels = min(len(template), len(subject_coords))
        
        for ch_idx in range(n_channels):
            t = template[ch_idx]
            s = subject_coords[ch_idx]
            error = np.linalg.norm(s - t)
            
            # Template point (black)
            plotter.add_mesh(pv.Sphere(radius=1.5, center=t), color='black', smooth_shading=True)
            
            # Subject point (red)
            plotter.add_mesh(pv.Sphere(radius=1.5, center=s), color='red', smooth_shading=True, opacity=0.7)
            
            # Arrow from template to subject
            if error > 1:  # Only show arrows for errors > 1mm
                arrow = pv.Arrow(start=t, direction=s-t, scale=error)
                plotter.add_mesh(arrow, color='blue', opacity=0.5)
        
        plotter.add_text(f"Subject: {subject_to_show}", position='upper_left', font_size=14)
    else:
        # Show mean displacement vectors
        all_subjects = np.stack(list(subject_data.values()), axis=0)  # (n_subjects, n_channels, 3)
        mean_subject = all_subjects.mean(axis=0)  # (n_channels, 3)
        
        n_channels = min(len(template), len(mean_subject))
        
        for ch_idx in range(n_channels):
            t = template[ch_idx]
            s = mean_subject[ch_idx]
            error = np.linalg.norm(s - t)
            
            # Color by error magnitude
            norm_err = min(error / 15.0, 1.0)  # Normalize to 0-1, cap at 15mm
            color = plt.cm.jet(norm_err)[:3]
            
            # Template point
            plotter.add_mesh(pv.Sphere(radius=2.0, center=t), color=color, smooth_shading=True)
            
            # Arrow from template to mean subject position
            if error > 1:
                arrow = pv.Arrow(start=t, direction=s-t, scale=error * 0.8)
                plotter.add_mesh(arrow, color=color, opacity=0.6)
        
        plotter.add_text("Mean displacement vectors", position='upper_left', font_size=14)
    
    plotter.add_axes()
    plotter.camera_position = 'yz'
    
    plotter.screenshot(str(output_path))
    plotter.close()
    print(f"  Saved: {output_path}")


def save_statistics(errors_df: pd.DataFrame, channel_stats: pd.DataFrame,
                    subject_stats: pd.DataFrame, output_dir: Path) -> None:
    """Save all statistics to CSV files."""
    print("\nSaving statistics...")
    
    # All individual errors
    errors_path = output_dir / "registration_errors_all.csv"
    errors_df.to_csv(errors_path, index=False, float_format='%.3f')
    print(f"  Saved: {errors_path}")
    
    # Per-channel statistics
    channel_path = output_dir / "registration_errors_by_channel.csv"
    channel_stats.to_csv(channel_path, index=False, float_format='%.3f')
    print(f"  Saved: {channel_path}")
    
    # Per-subject statistics
    subject_path = output_dir / "registration_errors_by_subject.csv"
    subject_stats.to_csv(subject_path, index=False, float_format='%.3f')
    print(f"  Saved: {subject_path}")


def main():
    """Main function to run MNI registration error analysis."""
    print("=" * 70)
    print("MNI REGISTRATION ERROR ANALYSIS")
    print("Comparing subject channel positions to template positions")
    print("=" * 70)
    print(f"\nSubjects: {SUBJECT_IDS}")
    print(f"Template: {TEMPLATE_MNI_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load template (ground truth)
    template = load_template_mni(TEMPLATE_MNI_PATH)
    
    # Load subject data
    subject_data = load_all_subjects(SUBJECT_IDS, BASE_DIR)
    
    # Compute errors
    errors_df = compute_registration_errors(template, subject_data)
    
    # Compute statistics
    channel_stats = compute_channel_statistics(errors_df, template)
    subject_stats = compute_subject_statistics(errors_df)
    
    # Save statistics
    save_statistics(errors_df, channel_stats, subject_stats, OUTPUT_DIR)
    
    # Generate visualizations
    plot_2d_error_map(channel_stats, OUTPUT_DIR / "registration_error_2d_map.png")
    plot_error_by_axis(channel_stats, OUTPUT_DIR / "registration_error_by_axis.png")
    plot_error_histograms(errors_df, channel_stats, subject_stats, 
                          OUTPUT_DIR / "registration_error_histograms.png")
    
    # 3D visualizations
    plot_3d_error_pyvista(channel_stats, errors_df, OUTPUT_DIR / "registration_error_3d.png")
    plot_3d_template_vs_subjects(template, subject_data, channel_stats,
                                  OUTPUT_DIR / "template_vs_subjects_3d.png")
    plot_3d_error_vectors(template, subject_data, OUTPUT_DIR / "error_vectors_3d.png")
    
    # Also save individual subject vector plots
    for subject_id in list(subject_data.keys())[:3]:  # First 3 subjects as examples
        plot_3d_error_vectors(template, subject_data,
                              OUTPUT_DIR / f"error_vectors_{subject_id}.png",
                              subject_to_show=subject_id)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
