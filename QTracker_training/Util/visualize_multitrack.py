"""
Animate the multi-track finding process as a video.

Shows:
- Hit matrix evolution (input at each step)
- Predicted mu+ and mu- tracks overlaid
- Confidence score (if available)
- Track subtraction visualization
"""

import argparse
import os
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.layers import AxialAttention
from models.data_loader import load_data


def load_single_event(root_file: str, event_idx: int):
    """Load a single event's data."""
    X, y_mup, y_mum = load_data(root_file)
    if X is None:
        return None, None, None
    return X[event_idx:event_idx+1], y_mup[event_idx], y_mum[event_idx]


def run_multitrack_step_by_step(model, X_init, max_steps=5, confidence_threshold=0.5):
    """
    Run multi-track finder step by step, recording state at each iteration.
    
    Returns:
        List of dicts, one per step, containing:
        - X: current hit matrix (N, 62, 201, 1)
        - mu_plus_pred: predicted mu+ track (N, 62)
        - mu_minus_pred: predicted mu- track (N, 62)
        - mu_plus_softmax: softmax for mu+ (N, 62, 201)
        - mu_minus_softmax: softmax for mu- (N, 62, 201)
        - confidence: confidence score (N,) or None
        - active: whether event is still active (bool)
    """
    X = X_init.copy()
    active = True
    steps_data = []
    
    # Detect if model has confidence head
    has_confidence = False
    if isinstance(model.output, (list, tuple)):
        has_confidence = len(model.output) >= 3
    
    for step in range(max_steps):
        if not active:
            # Pad with zeros
            steps_data.append({
                'X': np.zeros_like(X_init),
                'mu_plus_pred': np.zeros(62, dtype=np.int32),
                'mu_minus_pred': np.zeros(62, dtype=np.int32),
                'mu_plus_softmax': np.zeros((62, 201), dtype=np.float32),
                'mu_minus_softmax': np.zeros((62, 201), dtype=np.float32),
                'confidence': 0.0,
                'active': False,
            })
            continue
        
        # Forward pass
        outputs = model.predict(tf.cast(X, tf.float32), verbose=0)
        
        if has_confidence:
            _, y_pred, confidence = outputs[0], outputs[1], outputs[2]
            confidence_val = float(confidence[0, 0] if confidence.ndim > 1 else confidence[0])
            if confidence_val < confidence_threshold:
                active = False
        else:
            _, y_pred = outputs[0], outputs[1]
            confidence_val = None
        
        # Extract predictions
        mp_softmax = y_pred[0, 0, :, :]  # (62, 201)
        mm_softmax = y_pred[0, 1, :, :]  # (62, 201)
        
        mu_plus_pred = np.argmax(mp_softmax, axis=-1).astype(np.int32)  # (62,)
        mu_minus_pred = np.argmax(mm_softmax, axis=-1).astype(np.int32)  # (62,)
        
        # Store this step's data
        steps_data.append({
            'X': X.copy(),
            'mu_plus_pred': mu_plus_pred,
            'mu_minus_pred': mu_minus_pred,
            'mu_plus_softmax': mp_softmax,
            'mu_minus_softmax': mm_softmax,
            'confidence': confidence_val,
            'active': active,
        })
        
        # Soft subtraction for next iteration
        X_current = X[0, :, :, 0]  # (62, 201)
        for det_id in range(62):
            elem_plus = mu_plus_pred[det_id]
            elem_minus = mu_minus_pred[det_id]
            if elem_plus > 0:
                prob = mp_softmax[det_id, elem_plus]
                X_current[det_id, elem_plus] = max(0, X_current[det_id, elem_plus] - prob)
            if elem_minus > 0:
                prob = mm_softmax[det_id, elem_minus]
                X_current[det_id, elem_minus] = max(0, X_current[det_id, elem_minus] - prob)
        X = X_current[:, :, np.newaxis, np.newaxis]  # Restore shape
    
    return steps_data


def create_animation(steps_data, y_mup_true, y_mum_true, output_file, event_idx=0):
    """
    Create animated visualization of multi-track finding.
    
    Args:
        steps_data: List of dicts from run_multitrack_step_by_step
        y_mup_true: Ground truth mu+ track (62,)
        y_mum_true: Ground truth mu- track (62,)
        output_file: Output video filename (.mp4 or .gif)
        event_idx: Event number for title
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Multi-Track Finding Process - Event {event_idx}', fontsize=16)
    
    # Custom colormap for hit matrix
    cmap_hits = LinearSegmentedColormap.from_list('hits', ['white', 'blue', 'red'], N=256)
    
    def update(frame):
        for ax in axes.flat:
            ax.clear()
        
        step_data = steps_data[frame]
        X_current = step_data['X'][0, :, :, 0]  # (62, 201)
        mu_plus_pred = step_data['mu_plus_pred']
        mu_minus_pred = step_data['mu_minus_pred']
        confidence = step_data['confidence']
        active = step_data['active']
        
        # Top-left: Hit matrix with predictions overlaid
        ax = axes[0, 0]
        im = ax.imshow(X_current.T, aspect='auto', origin='lower', 
                      cmap=cmap_hits, vmin=0, vmax=1,
                      extent=[0, 62, 0, 201])
        
        # Overlay predictions
        mup_nonzero = mu_plus_pred > 0
        ax.scatter(np.arange(62)[mup_nonzero], mu_plus_pred[mup_nonzero], 
                  c='lime', s=30, marker='o', label='μ+ pred', edgecolors='black', linewidths=0.5)
        
        mum_nonzero = mu_minus_pred > 0
        ax.scatter(np.arange(62)[mum_nonzero], mu_minus_pred[mum_nonzero], 
                  c='yellow', s=30, marker='s', label='μ− pred', edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel('Detector ID')
        ax.set_ylabel('Element ID')
        status = "ACTIVE" if active else "STOPPED"
        ax.set_title(f'Step {frame+1}/{len(steps_data)} - {status}')
        ax.legend(loc='upper right')
        plt.colorbar(im, ax=ax, label='Hit Probability')
        
        # Top-right: Mu+ softmax
        ax = axes[0, 1]
        im = ax.imshow(step_data['mu_plus_softmax'].T, aspect='auto', origin='lower',
                      cmap='viridis', vmin=0, vmax=0.5,
                      extent=[0, 62, 0, 201])
        ax.scatter(np.arange(62)[mup_nonzero], mu_plus_pred[mup_nonzero], 
                  c='red', s=20, marker='x', label='Predicted')
        ax.set_xlabel('Detector ID')
        ax.set_ylabel('Element ID')
        ax.set_title('μ+ Softmax Output')
        ax.legend()
        plt.colorbar(im, ax=ax, label='Probability')
        
        # Bottom-left: Mu- softmax
        ax = axes[1, 0]
        im = ax.imshow(step_data['mu_minus_softmax'].T, aspect='auto', origin='lower',
                      cmap='viridis', vmin=0, vmax=0.5,
                      extent=[0, 62, 0, 201])
        ax.scatter(np.arange(62)[mum_nonzero], mu_minus_pred[mum_nonzero], 
                  c='red', s=20, marker='x', label='Predicted')
        ax.set_xlabel('Detector ID')
        ax.set_ylabel('Element ID')
        ax.set_title('μ− Softmax Output')
        ax.legend()
        plt.colorbar(im, ax=ax, label='Probability')
        
        # Bottom-right: Track comparison and metrics
        ax = axes[1, 1]
        ax.axis('off')
        
        # Text info
        text_y = 0.9
        line_spacing = 0.08
        
        ax.text(0.05, text_y, f'Step: {frame+1}/{len(steps_data)}', 
               fontsize=14, weight='bold', transform=ax.transAxes)
        text_y -= line_spacing
        
        if confidence is not None:
            ax.text(0.05, text_y, f'Confidence: {confidence:.4f}', 
                   fontsize=12, transform=ax.transAxes)
            text_y -= line_spacing
        
        ax.text(0.05, text_y, f'Status: {status}', 
               fontsize=12, color='green' if active else 'red',
               weight='bold', transform=ax.transAxes)
        text_y -= 1.5 * line_spacing
        
        # Track accuracy
        mup_correct = np.sum((mu_plus_pred == y_mup_true) & (y_mup_true > 0))
        mup_total = np.sum(y_mup_true > 0)
        mum_correct = np.sum((mu_minus_pred == y_mum_true) & (y_mum_true > 0))
        mum_total = np.sum(y_mum_true > 0)
        
        ax.text(0.05, text_y, 'Accuracy (this step):', 
               fontsize=12, weight='bold', transform=ax.transAxes)
        text_y -= line_spacing
        
        if mup_total > 0:
            mup_acc = mup_correct / mup_total
            ax.text(0.05, text_y, f'  μ+: {mup_correct}/{mup_total} = {mup_acc:.1%}', 
                   fontsize=11, transform=ax.transAxes)
            text_y -= line_spacing
        
        if mum_total > 0:
            mum_acc = mum_correct / mum_total
            ax.text(0.05, text_y, f'  μ−: {mum_correct}/{mum_total} = {mum_acc:.1%}', 
                   fontsize=11, transform=ax.transAxes)
            text_y -= line_spacing
        
        text_y -= line_spacing
        
        # Show hit counts
        hits_remaining = int(np.sum(X_current > 0))
        ax.text(0.05, text_y, f'Hits remaining: {hits_remaining}', 
               fontsize=11, transform=ax.transAxes)
        
        plt.tight_layout()
        return fig,
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(steps_data),
        interval=1000,  # 1 second per frame
        blit=False, repeat=True
    )
    
    # Save
    if output_file.endswith('.gif'):
        anim.save(output_file, writer='pillow', fps=1, dpi=100)
        print(f"Saved animation as GIF: {output_file}")
    else:
        # Save as mp4
        anim.save(output_file, writer='ffmpeg', fps=1, dpi=100, bitrate=2000)
        print(f"Saved animation as video: {output_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create animation of multi-track finding process'
    )
    parser.add_argument('root_file', type=str, help='Input ROOT file')
    parser.add_argument('model_path', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--event', type=int, default=0, 
                       help='Event index to visualize')
    parser.add_argument('--max_steps', type=int, default=5,
                       help='Maximum number of iterations')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for stopping')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename (.mp4 or .gif). Default: auto-generated')
    parser.add_argument('--format', type=str, choices=['mp4', 'gif'], default='mp4',
                       help='Output format (default: mp4)')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(
        args.model_path,
        compile=False,
        custom_objects={'AxialAttention': AxialAttention}
    )
    
    # Load event
    print(f"Loading event {args.event} from {args.root_file}...")
    X_event, y_mup_true, y_mum_true = load_single_event(args.root_file, args.event)
    if X_event is None:
        print("Error loading data!")
        return
    
    # Run step-by-step
    print(f"Running multi-track finder for {args.max_steps} steps...")
    steps_data = run_multitrack_step_by_step(
        model, X_event, 
        max_steps=args.max_steps,
        confidence_threshold=args.confidence_threshold
    )
    
    # Generate output filename
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.model_path))[0]
        output_file = f"multitrack_event{args.event}_{base}.{args.format}"
    else:
        output_file = args.output
    
    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots', 'animations')
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, output_file)
    
    # Create animation
    print(f"Creating animation...")
    create_animation(steps_data, y_mup_true, y_mum_true, output_path, args.event)
    
    print(f"\n✅ Animation saved to: {output_path}")
    print(f"   Total steps: {len(steps_data)}")
    print(f"   Active steps: {sum(1 for s in steps_data if s['active'])}")


if __name__ == '__main__':
    main()
