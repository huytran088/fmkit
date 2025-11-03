'''
Modified visualization module for headless/remote environments.
Saves animations as GIF or MP4 files instead of displaying them.

Author: Huy Tran <<huyfuv81212@gmail.com>>
From Duo Lu's fmsignal_vis.py
Modified for headless environments

Version: 0.3 (Headless-compatible version)
License: MIT
'''

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import os
import sys

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
    from fmsignal import FMSignal
    from fmsignal import FMSignalLeap
    from fmsignal import FMSignalGlove
    from fmsignal_vis import plot_xyz_axes
    # print("Successfully imported plot_xyz_axes from fmsignal_vis")
except ImportError as e:
    print(f"Could not import modulde: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


def trajectory_animation_to_file(signal, output_file='trajectory_animation.gif', 
                                 speed=1, seg_length=-1, show_hand=False, 
                                 fps=20, dpi=100):
    '''Save the signal trajectory animation to a file (GIF or MP4).

    Args:
        signal (FMSignal): The signal to be visualized.
        output_file (str): Output filename (.gif or .mp4)
        speed (float): Animation speed. "speed=1" is 1x.
        seg_length (int): The animated segment length in number of samples.
        show_hand (bool): Whether to show hand geometry.
        fps (int): Frames per second for the output file.
        dpi (int): DPI for the output file.

    Returns:
        str: Path to the saved animation file.
    '''

    if isinstance(signal, FMSignal):
        trajectory = signal.data[:, 0:3]
    elif isinstance(signal, FMSignalLeap) or isinstance(signal, FMSignalGlove):
        trajectory = signal.trajectory
        assert trajectory is not None
    else:
        raise ValueError('Wrong signal: %s.' % signal.__class__)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=160)

    tx = trajectory[:, 0]
    ty = trajectory[:, 1]
    tz = trajectory[:, 2]

    # Calculate limits
    dx = max(abs(np.max(tx)), abs(np.min(tx)))
    dy = max(abs(np.max(ty)), abs(np.min(ty)))
    dz = max(abs(np.max(tz)), abs(np.min(tz)))
    dist = max(dx, dy, dz)
    lim = dist * 1.2

    l = signal.length
    
    # Prepare frames for animation
    if speed > 1:
        frame_indices = list(range(1, l, int(round(speed))))
    else:
        frame_indices = list(range(1, l))

    def update(frame_idx):
        ax.clear()
        
        i = frame_indices[frame_idx]
        
        if seg_length <= 0:
            s = 0
        else:
            s = i - seg_length
            if s < 0:
                s = 0
        
        if i > s:
            ax.plot(tx[s:i], ty[s:i], tz[s:i], 'k-', linewidth=1.5)
            # Add a marker for the current position
            ax.scatter(tx[i-1], ty[i-1], tz[i-1], c='red', s=50, marker='o')
        
        if isinstance(signal, FMSignalLeap) and show_hand:
            from fmsignal_vis import plot_handgeo
            plot_handgeo(ax, signal, i)
        
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {i}/{l}')
        
        return [ax]

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(frame_indices), 
                        interval=50/speed if speed >= 1 else 50, 
                        blit=False)

    # Save animation
    if output_file.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer, dpi=dpi)
        print(f"Animation saved as GIF: {output_file}")
    elif output_file.endswith('.mp4'):
        try:
            writer = FFMpegWriter(fps=fps)
            anim.save(output_file, writer=writer, dpi=dpi)
            print(f"Animation saved as MP4: {output_file}")
        except:
            print("FFmpeg not available, saving as GIF instead")
            output_file = output_file.replace('.mp4', '.gif')
            writer = PillowWriter(fps=fps)
            anim.save(output_file, writer=writer, dpi=dpi)
            print(f"Animation saved as GIF: {output_file}")
    else:
        print(f"Unknown format, saving as GIF")
        output_file = output_file + '.gif'
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer, dpi=dpi)
        print(f"Animation saved as GIF: {output_file}")

    plt.close(fig)
    return output_file

def orientation_animation_to_file(signal, output_file='orientation_animation.gif',
                                  speed=1, seg_length=-1, fps=20, dpi=100):
    '''Save the orientation animation to a file (GIF or MP4).

    Args:
        signal (FMSignal): The signal to be visualized.
        output_file (str): Output filename (.gif or .mp4)
        speed (float): Animation speed. "speed=1" is 1x.
        seg_length (int): The animated segment length in number of samples.
        fps (int): Frames per second for the output file.
        dpi (int): DPI for the output file.

    Returns:
        str: Path to the saved animation file.
    '''

    if isinstance(signal, FMSignal):
        rotms, _qs = signal.get_orientation()
    elif isinstance(signal, FMSignalLeap) or isinstance(signal, FMSignalGlove):
        rotms = signal.rotms
    else:
        raise ValueError('Wrong signal: %s.' % signal.__class__)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=30, azim=160)
    
    l = signal.length
    scale = 100
    lim = scale * 1.2

    # Pre-compute all orientation points
    xv = np.asarray((scale, 0, 0), np.float32).reshape((3, 1))
    points_x = np.zeros((l, 3))

    for i in range(l):
        R = rotms[i]
        xv_R = np.matmul(R, xv)
        points_x[i] = xv_R[:, 0]

    ox = points_x[:, 0]
    oy = points_x[:, 1]
    oz = points_x[:, 2]

    # Prepare frames for animation
    if speed > 1:
        frame_indices = list(range(1, l, int(round(speed))))
    else:
        frame_indices = list(range(1, l))

    def update(frame_idx):
        ax.clear()
        
        i = frame_indices[frame_idx]
        
        if seg_length <= 0:
            s = 0
        else:
            s = i - seg_length
            if s < 0:
                s = 0
        
        # Plot trajectory
        if i > s:
            ax.plot(ox[s:i], oy[s:i], oz[s:i], color='k', linewidth=1.5)
        
        # Plot orientation axes
        R = rotms[i]
        t = np.asarray((0, 0, 0)).reshape((3, 1))
        plot_xyz_axes(ax, R, t, scale=scale)
        
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {i}/{l}')
        
        return [ax]

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(frame_indices),
                        interval=50/speed if speed >= 1 else 50,
                        blit=False)

    # Save animation
    if output_file.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer, dpi=dpi)
        print(f"Animation saved as GIF: {output_file}")
    elif output_file.endswith('.mp4'):
        try:
            writer = FFMpegWriter(fps=fps)
            anim.save(output_file, writer=writer, dpi=dpi)
            print(f"Animation saved as MP4: {output_file}")
        except:
            print("FFmpeg not available, saving as GIF instead")
            output_file = output_file.replace('.mp4', '.gif')
            writer = PillowWriter(fps=fps)
            anim.save(output_file, writer=writer, dpi=dpi)
            print(f"Animation saved as GIF: {output_file}")
    else:
        print(f"Unknown format, saving as GIF")
        output_file = output_file + '.gif'
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer, dpi=dpi)
        print(f"Animation saved as GIF: {output_file}")

    plt.close(fig)
    return output_file


def trajectory_static_views(signal, output_file='trajectory_views.png', show_hand=False):
    '''Save multiple static views of the trajectory to a single image.

    Args:
        signal (FMSignal): The signal to be visualized.
        output_file (str): Output filename for the image.
        show_hand (bool): Whether to show hand geometry.

    Returns:
        str: Path to the saved image file.
    '''

    if isinstance(signal, FMSignal):
        trajectory = signal.data[:, 0:3]
    elif isinstance(signal, FMSignalLeap) or isinstance(signal, FMSignalGlove):
        trajectory = signal.trajectory
        assert trajectory is not None
    else:
        raise ValueError('Wrong signal: %s.' % signal.__class__)

    fig = plt.figure(figsize=(20, 5))
    
    tx = trajectory[:, 0]
    ty = trajectory[:, 1]
    tz = trajectory[:, 2]
    
    # Calculate limits
    dx = max(abs(np.max(tx)), abs(np.min(tx)))
    dy = max(abs(np.max(ty)), abs(np.min(ty)))
    dz = max(abs(np.max(tz)), abs(np.min(tz)))
    dist = max(dx, dy, dz)
    lim = dist * 1.2
    
    # View 1: Default perspective
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.plot(tx, ty, tz, 'b-', linewidth=1.5)
    ax1.scatter(tx[0], ty[0], tz[0], c='green', s=100, marker='o', label='Start')
    ax1.scatter(tx[-1], ty[-1], tz[-1], c='red', s=100, marker='s', label='End')
    ax1.view_init(elev=30, azim=45)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_zlim(-lim, lim)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Perspective View')
    ax1.legend()
    
    # View 2: Top view (XY plane)
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.plot(tx, ty, tz, 'b-', linewidth=1.5)
    ax2.scatter(tx[0], ty[0], tz[0], c='green', s=100, marker='o')
    ax2.scatter(tx[-1], ty[-1], tz[-1], c='red', s=100, marker='s')
    ax2.view_init(elev=90, azim=0)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_zlim(-lim, lim)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Top View (XY)')
    
    # View 3: Front view (XZ plane)
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.plot(tx, ty, tz, 'b-', linewidth=1.5)
    ax3.scatter(tx[0], ty[0], tz[0], c='green', s=100, marker='o')
    ax3.scatter(tx[-1], ty[-1], tz[-1], c='red', s=100, marker='s')
    ax3.view_init(elev=0, azim=0)
    ax3.set_xlim(-lim, lim)
    ax3.set_ylim(-lim, lim)
    ax3.set_zlim(-lim, lim)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Front View (XZ)')
    
    # View 4: Side view (YZ plane)
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.plot(tx, ty, tz, 'b-', linewidth=1.5)
    ax4.scatter(tx[0], ty[0], tz[0], c='green', s=100, marker='o')
    ax4.scatter(tx[-1], ty[-1], tz[-1], c='red', s=100, marker='s')
    ax4.view_init(elev=0, azim=90)
    ax4.set_xlim(-lim, lim)
    ax4.set_ylim(-lim, lim)
    ax4.set_zlim(-lim, lim)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('Side View (YZ)')
    
    plt.suptitle(f'Trajectory Views - {signal.user} {signal.cid} {signal.seq}')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Static views saved as: {output_file}")
    return output_file


def create_trajectory_frames(signal, output_dir='frames', seg_length=-1, 
                            num_frames=10, show_hand=False):
    '''Save key frames of the trajectory animation as separate images.

    Args:
        signal (FMSignal): The signal to be visualized.
        output_dir (str): Directory to save frame images.
        seg_length (int): The animated segment length in number of samples.
        num_frames (int): Number of frames to extract.
        show_hand (bool): Whether to show hand geometry.

    Returns:
        list: List of paths to saved frame files.
    '''
    
    if isinstance(signal, FMSignal):
        trajectory = signal.data[:, 0:3]
    elif isinstance(signal, FMSignalLeap) or isinstance(signal, FMSignalGlove):
        trajectory = signal.trajectory
        assert trajectory is not None
    else:
        raise ValueError('Wrong signal: %s.' % signal.__class__)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    tx = trajectory[:, 0]
    ty = trajectory[:, 1]
    tz = trajectory[:, 2]
    
    # Calculate limits
    dx = max(abs(np.max(tx)), abs(np.min(tx)))
    dy = max(abs(np.max(ty)), abs(np.min(ty)))
    dz = max(abs(np.max(tz)), abs(np.min(tz)))
    dist = max(dx, dy, dz)
    lim = dist * 1.2
    
    l = signal.length
    frame_indices = np.linspace(1, l-1, num_frames, dtype=int)
    
    saved_files = []
    
    for frame_num, i in enumerate(frame_indices):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=160)
        
        if seg_length <= 0:
            s = 0
        else:
            s = i - seg_length
            if s < 0:
                s = 0
        
        # Plot trajectory up to current point
        if i > s:
            ax.plot(tx[s:i], ty[s:i], tz[s:i], 'b-', linewidth=1.5, alpha=0.7)
            # Highlight current position
            ax.scatter(tx[i-1], ty[i-1], tz[i-1], c='red', s=100, marker='o')
        
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame_num+1}/{num_frames} (Sample {i}/{l})')
        
        filename = os.path.join(output_dir, f'frame_{frame_num+1:03d}.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        saved_files.append(filename)
        print(f"Saved frame {frame_num+1}/{num_frames}: {filename}")
    
    return saved_files
