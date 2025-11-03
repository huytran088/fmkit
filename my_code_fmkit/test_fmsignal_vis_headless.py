#!/usr/bin/env python3
"""
Simple test script for headless/Colab environments.
Creates static plots and animations saved as files.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add code_fmkit to Python path
# Adjust the path based on where code_fmkit is relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
code_fmkit_path = os.path.join(current_dir, 'code_fmkit')
if os.path.exists(code_fmkit_path):
    sys.path.insert(0, code_fmkit_path)
else:
    # Try parent directory
    code_fmkit_path = os.path.join(current_dir, '..', 'code_fmkit')
    if os.path.exists(code_fmkit_path):
        sys.path.insert(0, code_fmkit_path)

# Import from fmsignal_vis
try:
    from fmsignal_vis import plot_xyz_axes
    # print("Successfully imported plot_xyz_axes from fmsignal_vis")
except ImportError as e:
    print(f"Could not import modulde: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

def create_spiral_trajectory(n_points=100):
    """Create a simple spiral trajectory for testing."""
    t = np.linspace(0, 4*np.pi, n_points)
    x = np.cos(t) * t * 10
    y = np.sin(t) * t * 10
    z = t * 5
    return np.column_stack([x, y, z]), t

def create_rotation_matrices(t_values):
    """Create rotation matrices that evolve along the trajectory."""
    rotms = []
    for t in t_values:
        # Rotation around Z-axis that increases with time
        angle = t / 2
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Add some pitch variation
        pitch = np.sin(t / 4) * 0.3
        cos_p = np.cos(pitch)
        sin_p = np.sin(pitch)
        
        # Combined rotation matrix (yaw + pitch)
        R = np.array([
            [cos_a * cos_p, -sin_a, cos_a * sin_p],
            [sin_a * cos_p, cos_a, sin_a * sin_p],
            [-sin_p, 0, cos_p]
        ], dtype=np.float32)
        rotms.append(R)
    return np.array(rotms)

def save_trajectory_plot(trajectory, filename='trajectory_plot.png'):
    """Save a static 3D plot of the trajectory."""
    fig = plt.figure(figsize=(12, 10))
    
    # Create 4 subplots with different views
    views = [
        (30, 45, '3D Perspective'),
        (90, 0, 'Top View (XY)'),
        (0, 0, 'Front View (XZ)'),
        (0, 90, 'Side View (YZ)')
    ]
    
    for idx, (elev, azim, title) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                'b-', linewidth=2, alpha=0.7)
        
        # Mark start and end points
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                  c='green', s=100, marker='o', label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                  c='red', s=100, marker='s', label='End')
        
        # Set view angle
        ax.view_init(elev=elev, azim=azim)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        
        # Set equal aspect ratio
        max_range = np.array([
            trajectory[:, 0].max() - trajectory[:, 0].min(),
            trajectory[:, 1].max() - trajectory[:, 1].min(),
            trajectory[:, 2].max() - trajectory[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (trajectory[:, 0].max() + trajectory[:, 0].min()) * 0.5
        mid_y = (trajectory[:, 1].max() + trajectory[:, 1].min()) * 0.5
        mid_z = (trajectory[:, 2].max() + trajectory[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.suptitle('Trajectory Visualization', fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved static plot: {filename}")
    return filename

def save_animation_frames(trajectory, output_dir='animation_frames', num_frames=10):
    """Save animation frames as individual images."""
    os.makedirs(output_dir, exist_ok=True)
    
    frame_indices = np.linspace(2, len(trajectory), num_frames, dtype=int)
    
    for frame_num, end_idx in enumerate(frame_indices, 1):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory up to current frame
        ax.plot(trajectory[:end_idx, 0], 
                trajectory[:end_idx, 1], 
                trajectory[:end_idx, 2], 
                'b-', linewidth=2, alpha=0.7)
        
        # Highlight current position
        ax.scatter(trajectory[end_idx-1, 0], 
                  trajectory[end_idx-1, 1], 
                  trajectory[end_idx-1, 2], 
                  c='red', s=100, marker='o')
        
        # Set view and labels
        ax.view_init(elev=30, azim=45)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Animation Frame {frame_num}/{num_frames}')
        
        # Set consistent limits for all frames
        max_range = 65
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(0, 70)
        
        # Save frame
        filename = os.path.join(output_dir, f'frame_{frame_num:03d}.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {num_frames} animation frames in '{output_dir}/'")
    return output_dir

def save_orientation_frames(trajectory, rotms, output_dir='orientation_frames', num_frames=10):
    """Save orientation animation frames showing coordinate axes."""
    os.makedirs(output_dir, exist_ok=True)
    
    frame_indices = np.linspace(2, len(trajectory), num_frames, dtype=int)
    scale = 10  # Scale for the axes
    
    # Pre-compute orientation trail
    xv = np.asarray((scale, 0, 0), np.float32).reshape((3, 1))
    points_x = np.zeros((len(trajectory), 3))
    for i in range(len(trajectory)):
        R = rotms[i]
        xv_R = np.matmul(R, xv)
        points_x[i] = xv_R[:, 0]
    
    for frame_num, end_idx in enumerate(frame_indices, 1):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot orientation trail
        if end_idx > 1:
            ax.plot(points_x[:end_idx, 0], 
                   points_x[:end_idx, 1], 
                   points_x[:end_idx, 2], 
                   'k-', linewidth=1, alpha=0.5)
        
        # Plot current orientation axes
        R = rotms[end_idx-1]
        t = np.asarray((0, 0, 0)).reshape((3, 1))
        plot_xyz_axes(ax, R, t, scale=scale)
        
        # Set view and labels
        ax.view_init(elev=30, azim=160)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Orientation Frame {frame_num}/{num_frames}')
        
        # Set consistent limits for all frames
        lim = scale * 1.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        
        # Save frame
        filename = os.path.join(output_dir, f'orientation_{frame_num:03d}.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {num_frames} orientation frames in '{output_dir}/'")
    return output_dir

def create_gif_from_frames(frame_dir='animation_frames', output_file='animation.gif', pattern='frame_*.png'):
    """Create a GIF from saved frames using matplotlib."""
    try:
        from PIL import Image
        import glob
        
        # Get all frame files
        frame_files = sorted(glob.glob(os.path.join(frame_dir, pattern)))
        
        if not frame_files:
            print(f"No frames found in {frame_dir} matching {pattern}")
            return None
        
        # Load images
        images = []
        for filename in frame_files:
            images.append(Image.open(filename))
        
        # Save as GIF
        images[0].save(
            output_file,
            save_all=True,
            append_images=images[1:],
            duration=200,  # milliseconds per frame
            loop=0
        )
        
        print(f"Created GIF animation: {output_file}")
        return output_file
        
    except ImportError:
        print("PIL/Pillow not installed. Install with: pip install Pillow")
        return None

def main():
    """Run all visualization tests."""
    print("=" * 60)
    print("Testing Visualization in Headless Environment")
    print("=" * 60)
    
    # Create test trajectory and rotation matrices
    print("\n1. Creating test trajectory and orientations...")
    trajectory, t_values = create_spiral_trajectory(100)
    rotms = create_rotation_matrices(t_values)
    print(f"   Trajectory shape: {trajectory.shape}")
    print(f"   Rotation matrices shape: {rotms.shape}")
    
    # Save static plot
    print("\n2. Creating static trajectory visualization...")
    save_trajectory_plot(trajectory, 'test_trajectory.png')
    
    # Save trajectory animation frames
    print("\n3. Creating trajectory animation frames...")
    save_animation_frames(trajectory, 'test_frames', num_frames=20)
    
    # Save orientation animation frames
    print("\n4. Creating orientation animation frames...")
    save_orientation_frames(trajectory, rotms, 'test_orientation_frames', num_frames=20)
    
    # Try to create GIFs
    print("\n5. Attempting to create GIF animations...")
    create_gif_from_frames('test_frames', 'test_trajectory_animation.gif', pattern='frame_*.png')
    create_gif_from_frames('test_orientation_frames', 'test_orientation_animation.gif', pattern='orientation_*.png')
    
    print("\n" + "=" * 60)
    print("Complete! Check the following files:")
    print("  - test_trajectory.png (static views)")
    print("  - test_frames/ (trajectory animation frames)")
    print("  - test_orientation_frames/ (orientation animation frames)")
    print("  - test_trajectory_animation.gif (if PIL is installed)")
    print("  - test_orientation_animation.gif (if PIL is installed)")
    print("=" * 60)

if __name__ == "__main__":
    main()