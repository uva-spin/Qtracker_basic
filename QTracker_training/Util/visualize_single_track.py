"""
Animate the single-track finding process as a video.

Shows the progression:
1. Input (noisy hit matrix)
2. After denoising
3. Segmentation (mu+ and mu- predictions)
4. Final result with ground truth comparison
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
from models.data_loader import load_data_denoise


def load_single_event(root_file: str, event_idx: int):
    """Load a single event's data."""
    X, X_clean, y_mup, y_mum = load_data_denoise(root_file)
    if X is None:
        return None, None, None, None
    return (
        X[event_idx:event_idx+1], 
        X_clean[event_idx:event_idx+1],
        y_mup[event_idx], 
        y_mum[event_idx]
    )


def run_single_track_inference(model, X_input, X_clean_true):
    """
    Run single-track model and capture intermediate outputs.
    
    Returns dict with:
        - X_input: noisy input (1, 62, 201, 1)
        - X_clean_true: true clean input
        - X_denoised: denoised output from model
        - mu_plus_softmax: softmax for mu+ (62, 201)
        - mu_minus_softmax: softmax for mu- (62, 201)
        - mu_plus_pred: predicted mu+ track (62,)
        - mu_minus_pred: predicted mu- track (62,)
        - has_confidence: bool
        - confidence: float or None
    """
    outputs = model.predict(tf.cast(X_input, tf.float32), verbose=0)
    
    # Detect model outputs
    has_confidence = False
    if isinstance(outputs, (list, tuple)):
        if len(outputs) >= 3:
            has_confidence = True
            denoise_out, segment_out, confidence = outputs[0], outputs[1], outputs[2]
            confidence_val = float(confidence[0, 0] if confidence.ndim > 1 else confidence[0])
        else:
            denoise_out, segment_out = outputs[0], outputs[1]
            confidence_val = None
    else:
        denoise_out = outputs
        segment_out = None
        confidence_val = None
    
    # Extract predictions
    if segment_out is not None:
        mp_softmax = segment_out[0, 0, :, :]  # (62, 201)
        mm_softmax = segment_out[0, 1, :, :]  # (62, 201)
        
        mu_plus_pred = np.argmax(mp_softmax, axis=-1).astype(np.int32)
        mu_minus_pred = np.argmax(mm_softmax, axis=-1).astype(np.int32)
    else:
        mp_softmax = np.zeros((62, 201))
        mm_softmax = np.zeros((62, 201))
        mu_plus_pred = np.zeros(62, dtype=np.int32)
        mu_minus_pred = np.zeros(62, dtype=np.int32)
    
    return {
        'X_input': X_input[0, :, :, 0],  # (62, 201)
        'X_clean_true': X_clean_true[0, :, :, 0],
        'X_denoised': denoise_out[0, :, :, 0],
        'mu_plus_softmax': mp_softmax,
        'mu_minus_softmax': mm_softmax,
        'mu_plus_pred': mu_plus_pred,
        'mu_minus_pred': mu_minus_pred,
        'has_confidence': has_confidence,
        'confidence': confidence_val,
    }


def create_animation(inference_data, y_mup_true, y_mum_true, output_file, event_idx=0):
    """
    Create animated visualization showing the single-track finding pipeline.
    
    Frames:
    0: Input (noisy)
    1: Denoised (model output)
    2: Segmentation (mu+ softmax)
    3: Segmentation (mu- softmax)
    4: Final predictions vs ground truth
    """
    fig = plt.figure(figsize=(18, 10))
    
    # Custom colormap for hit matrix
    cmap_hits = LinearSegmentedColormap.from_list('hits', ['white', 'blue', 'red'], N=256)
    
    # Prepare data
    X_input = inference_data['X_input']
    X_clean_true = inference_data['X_clean_true']
    X_denoised = inference_data['X_denoised']
    mp_softmax = inference_data['mu_plus_softmax']
    mm_softmax = inference_data['mu_minus_softmax']
    mu_plus_pred = inference_data['mu_plus_pred']
    mu_minus_pred = inference_data['mu_minus_pred']
    confidence = inference_data['confidence']
    
    def update(frame):
        fig.clear()
        
        if frame == 0:
            # Frame 0: Input (noisy)
            ax = fig.add_subplot(1, 1, 1)
            im = ax.imshow(X_input.T, aspect='auto', origin='lower', 
                          cmap=cmap_hits, vmin=0, vmax=1,
                          extent=[0, 62, 0, 201])
            ax.set_xlabel('Detector ID', fontsize=12)
            ax.set_ylabel('Element ID', fontsize=12)
            ax.set_title(f'Step 1/5: Noisy Input - Event {event_idx}', fontsize=16, weight='bold')
            plt.colorbar(im, ax=ax, label='Hit Presence')
            
            # Add stats
            n_hits = int(np.sum(X_input > 0))
            ax.text(0.02, 0.98, f'Total hits: {n_hits}', 
                   transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        elif frame == 1:
            # Frame 1: Ground truth clean vs Denoised comparison
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            ax1 = fig.add_subplot(gs[0, :])
            im1 = ax1.imshow(X_clean_true.T, aspect='auto', origin='lower', 
                            cmap=cmap_hits, vmin=0, vmax=1,
                            extent=[0, 62, 0, 201])
            ax1.set_xlabel('Detector ID', fontsize=11)
            ax1.set_ylabel('Element ID', fontsize=11)
            ax1.set_title('Ground Truth (Clean)', fontsize=13)
            plt.colorbar(im1, ax=ax1, label='Hit')
            
            ax2 = fig.add_subplot(gs[1, :])
            im2 = ax2.imshow(X_denoised.T, aspect='auto', origin='lower', 
                            cmap=cmap_hits, vmin=0, vmax=1,
                            extent=[0, 62, 0, 201])
            ax2.set_xlabel('Detector ID', fontsize=11)
            ax2.set_ylabel('Element ID', fontsize=11)
            ax2.set_title('Step 2/5: Denoised Output (Model)', fontsize=13, weight='bold')
            plt.colorbar(im2, ax=ax2, label='Hit Probability')
            
            fig.suptitle(f'Denoising Stage - Event {event_idx}', fontsize=16, weight='bold')
        
        elif frame == 2:
            # Frame 2: mu+ segmentation
            gs = fig.add_gridspec(2, 1, hspace=0.3)
            
            ax1 = fig.add_subplot(gs[0])
            im1 = ax1.imshow(X_denoised.T, aspect='auto', origin='lower', 
                            cmap=cmap_hits, vmin=0, vmax=1,
                            extent=[0, 62, 0, 201])
            ax1.set_xlabel('Detector ID', fontsize=11)
            ax1.set_ylabel('Element ID', fontsize=11)
            ax1.set_title('Denoised Input', fontsize=13)
            plt.colorbar(im1, ax=ax1, label='Hit Probability')
            
            ax2 = fig.add_subplot(gs[1])
            im2 = ax2.imshow(mp_softmax.T, aspect='auto', origin='lower',
                            cmap='viridis', vmin=0, vmax=0.5,
                            extent=[0, 62, 0, 201])
            
            # Overlay predictions
            mup_nonzero = mu_plus_pred > 0
            ax2.scatter(np.arange(62)[mup_nonzero], mu_plus_pred[mup_nonzero], 
                       c='red', s=30, marker='x', label='Predicted', linewidths=2)
            
            # Overlay ground truth
            mup_true_nonzero = y_mup_true > 0
            ax2.scatter(np.arange(62)[mup_true_nonzero], y_mup_true[mup_true_nonzero], 
                       c='lime', s=40, marker='o', label='Ground Truth', 
                       edgecolors='black', linewidths=1, alpha=0.6)
            
            ax2.set_xlabel('Detector ID', fontsize=11)
            ax2.set_ylabel('Element ID', fontsize=11)
            ax2.set_title('Step 3/5: μ+ Segmentation Softmax', fontsize=13, weight='bold')
            ax2.legend(loc='upper right')
            plt.colorbar(im2, ax=ax2, label='Probability')
            
            fig.suptitle(f'Segmentation (μ+) - Event {event_idx}', fontsize=16, weight='bold')
        
        elif frame == 3:
            # Frame 3: mu- segmentation
            gs = fig.add_gridspec(2, 1, hspace=0.3)
            
            ax1 = fig.add_subplot(gs[0])
            im1 = ax1.imshow(X_denoised.T, aspect='auto', origin='lower', 
                            cmap=cmap_hits, vmin=0, vmax=1,
                            extent=[0, 62, 0, 201])
            ax1.set_xlabel('Detector ID', fontsize=11)
            ax1.set_ylabel('Element ID', fontsize=11)
            ax1.set_title('Denoised Input', fontsize=13)
            plt.colorbar(im1, ax=ax1, label='Hit Probability')
            
            ax2 = fig.add_subplot(gs[1])
            im2 = ax2.imshow(mm_softmax.T, aspect='auto', origin='lower',
                            cmap='viridis', vmin=0, vmax=0.5,
                            extent=[0, 62, 0, 201])
            
            # Overlay predictions
            mum_nonzero = mu_minus_pred > 0
            ax2.scatter(np.arange(62)[mum_nonzero], mu_minus_pred[mum_nonzero], 
                       c='red', s=30, marker='x', label='Predicted', linewidths=2)
            
            # Overlay ground truth
            mum_true_nonzero = y_mum_true > 0
            ax2.scatter(np.arange(62)[mum_true_nonzero], y_mum_true[mum_true_nonzero], 
                       c='yellow', s=40, marker='s', label='Ground Truth', 
                       edgecolors='black', linewidths=1, alpha=0.6)
            
            ax2.set_xlabel('Detector ID', fontsize=11)
            ax2.set_ylabel('Element ID', fontsize=11)
            ax2.set_title('Step 4/5: μ− Segmentation Softmax', fontsize=13, weight='bold')
            ax2.legend(loc='upper right')
            plt.colorbar(im2, ax=ax2, label='Probability')
            
            fig.suptitle(f'Segmentation (μ−) - Event {event_idx}', fontsize=16, weight='bold')
        
        elif frame == 4:
            # Frame 4: Final comparison
            gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
            
            # Denoised with predictions
            ax1 = fig.add_subplot(gs[0, :])
            im1 = ax1.imshow(X_denoised.T, aspect='auto', origin='lower', 
                            cmap=cmap_hits, vmin=0, vmax=1,
                            extent=[0, 62, 0, 201])
            
            # Overlay predictions
            mup_nonzero = mu_plus_pred > 0
            ax1.scatter(np.arange(62)[mup_nonzero], mu_plus_pred[mup_nonzero], 
                       c='lime', s=40, marker='o', label='μ+ pred', 
                       edgecolors='black', linewidths=0.5)
            
            mum_nonzero = mu_minus_pred > 0
            ax1.scatter(np.arange(62)[mum_nonzero], mu_minus_pred[mum_nonzero], 
                       c='yellow', s=40, marker='s', label='μ− pred', 
                       edgecolors='black', linewidths=0.5)
            
            ax1.set_xlabel('Detector ID', fontsize=11)
            ax1.set_ylabel('Element ID', fontsize=11)
            ax1.set_title('Final Predictions', fontsize=13, weight='bold')
            ax1.legend(loc='upper right')
            plt.colorbar(im1, ax=ax1, label='Hit Probability')
            
            # Metrics panel
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.axis('off')
            
            # Calculate accuracy
            mup_correct = np.sum((mu_plus_pred == y_mup_true) & (y_mup_true > 0))
            mup_total = np.sum(y_mup_true > 0)
            mum_correct = np.sum((mu_minus_pred == y_mum_true) & (y_mum_true > 0))
            mum_total = np.sum(y_mum_true > 0)
            
            text_y = 0.95
            line_spacing = 0.12
            
            ax2.text(0.1, text_y, 'Accuracy Metrics', 
                    fontsize=14, weight='bold', transform=ax2.transAxes)
            text_y -= line_spacing * 1.5
            
            if mup_total > 0:
                mup_acc = mup_correct / mup_total
                ax2.text(0.1, text_y, f'μ+ Exact Match:', 
                        fontsize=12, transform=ax2.transAxes)
                ax2.text(0.6, text_y, f'{mup_correct}/{mup_total} = {mup_acc:.1%}', 
                        fontsize=12, weight='bold', transform=ax2.transAxes)
                text_y -= line_spacing
            
            if mum_total > 0:
                mum_acc = mum_correct / mum_total
                ax2.text(0.1, text_y, f'μ− Exact Match:', 
                        fontsize=12, transform=ax2.transAxes)
                ax2.text(0.6, text_y, f'{mum_correct}/{mum_total} = {mum_acc:.1%}', 
                        fontsize=12, weight='bold', transform=ax2.transAxes)
                text_y -= line_spacing * 1.5
            
            # Residuals
            mup_res = np.abs(y_mup_true - mu_plus_pred)
            mum_res = np.abs(y_mum_true - mu_minus_pred)
            
            if mup_total > 0:
                mup_within2 = np.sum(mup_res[y_mup_true > 0] <= 2) / mup_total
                ax2.text(0.1, text_y, f'μ+ Within-2:', 
                        fontsize=12, transform=ax2.transAxes)
                ax2.text(0.6, text_y, f'{mup_within2:.1%}', 
                        fontsize=12, transform=ax2.transAxes)
                text_y -= line_spacing
            
            if mum_total > 0:
                mum_within2 = np.sum(mum_res[y_mum_true > 0] <= 2) / mum_total
                ax2.text(0.1, text_y, f'μ− Within-2:', 
                        fontsize=12, transform=ax2.transAxes)
                ax2.text(0.6, text_y, f'{mum_within2:.1%}', 
                        fontsize=12, transform=ax2.transAxes)
                text_y -= line_spacing * 1.5
            
            if confidence is not None:
                ax2.text(0.1, text_y, f'Confidence Score:', 
                        fontsize=12, transform=ax2.transAxes)
                ax2.text(0.6, text_y, f'{confidence:.4f}', 
                        fontsize=12, weight='bold', transform=ax2.transAxes)
            
            # Residual visualization
            ax3 = fig.add_subplot(gs[1, 1])
            
            detectors = np.arange(1, 63)
            ax3.plot(detectors, mup_res, 'o-', label='μ+ |residual|', color='blue', alpha=0.7)
            ax3.plot(detectors, mum_res, 's-', label='μ− |residual|', color='red', alpha=0.7)
            ax3.axhline(0, color='black', linestyle='--', linewidth=1)
            ax3.axhline(2, color='gray', linestyle=':', linewidth=1, label='±2 threshold')
            ax3.set_xlabel('Detector ID', fontsize=11)
            ax3.set_ylabel('Absolute Residual', fontsize=11)
            ax3.set_title('Residuals per Detector', fontsize=12)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            fig.suptitle(f'Step 5/5: Final Results - Event {event_idx}', fontsize=16, weight='bold')
        
        return fig,
    
    # Create animation
    n_frames = 5
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=2000,  # 2 seconds per frame
        blit=False, repeat=True
    )
    
    # Save
    if output_file.endswith('.gif'):
        anim.save(output_file, writer='pillow', fps=0.5, dpi=100)
        print(f"Saved animation as GIF: {output_file}")
    else:
        # Save as mp4
        anim.save(output_file, writer='ffmpeg', fps=0.5, dpi=100, bitrate=2000)
        print(f"Saved animation as video: {output_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create animation of single-track finding pipeline'
    )
    parser.add_argument('root_file', type=str, help='Input ROOT file')
    parser.add_argument('model_path', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--event', type=int, default=0, 
                       help='Event index to visualize')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename (.mp4 or .gif). Default: auto-generated')
    parser.add_argument('--format', type=str, choices=['mp4', 'gif'], default='mp4',
                       help='Output format (default: mp4)')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    # Import AxialAttention before loading to ensure it's registered
    from models.layers import AxialAttention
    
    # Try loading with custom_objects
    try:
        model = tf.keras.models.load_model(
            args.model_path,
            compile=False,
            custom_objects={'AxialAttention': AxialAttention}
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load with alternative custom_objects...")
        # Register under both possible module paths
        import sys
        import os
        # Add models directory to path if not already there
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
        
        # Import directly from layers module (as used during training)
        from layers import AxialAttention as AxialAttentionDirect
        
        model = tf.keras.models.load_model(
            args.model_path,
            compile=False,
            custom_objects={
                'AxialAttention': AxialAttentionDirect,
                'layers>AxialAttention': AxialAttentionDirect
            }
        )
        print("Model loaded successfully with alternative import!")

    
    # Load event
    print(f"Loading event {args.event} from {args.root_file}...")
    X_event, X_clean, y_mup_true, y_mum_true = load_single_event(args.root_file, args.event)
    if X_event is None:
        print("Error loading data!")
        return
    
    # Run inference
    print(f"Running single-track inference...")
    inference_data = run_single_track_inference(model, X_event, X_clean)
    
    # Generate output filename
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.model_path))[0]
        output_file = f"single_track_event{args.event}_{base}.{args.format}"
    else:
        output_file = args.output
    
    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots', 'animations')
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, output_file)
    
    # Create animation
    print(f"Creating animation...")
    create_animation(inference_data, y_mup_true, y_mum_true, output_path, args.event)
    
    print(f"\n✅ Animation saved to: {output_path}")
    print(f"   Frames: 5 (input → denoise → segment μ+ → segment μ− → results)")


if __name__ == '__main__':
    main()
