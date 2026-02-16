"""
Comprehensive comparison between Custom Autoencoder and Library-based Autoencoder
Generates visualizations and metrics comparison saved to outputs/ directory
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from autoencoder import AutoEncoder
from library_autoencoder import SequentialAutoencoder
import train_autoencoder as ta
import time
import json


def calculate_metrics(original, reconstructed):
    """Calculate various reconstruction metrics"""
    mse = nn.MSELoss()(reconstructed, original).item()
    
    # PSNR (Peak Signal-to-Noise Ratio)
    if mse > 0:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # MAE (Mean Absolute Error)
    mae = torch.mean(torch.abs(original - reconstructed)).item()
    
    return {
        'mse': mse,
        'psnr': psnr,
        'mae': mae
    }


def test_autoencoder(model, test_loader, device, model_name="Model"):
    """Test an autoencoder and return metrics and sample reconstructions"""
    model.eval()
    
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    all_originals = []
    all_reconstructions = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_features, _ in test_loader:
            batch_features = batch_features.view(-1, 784).to(device)
            reconstructions = model(batch_features)
            
            # Calculate batch metrics
            metrics = calculate_metrics(batch_features, reconstructions)
            total_mse += metrics['mse'] * batch_features.size(0)
            total_mae += metrics['mae'] * batch_features.size(0)
            total_samples += batch_features.size(0)
            
            # Store first batch for visualization
            if len(all_originals) == 0:
                all_originals = batch_features[:20].cpu()
                all_reconstructions = reconstructions[:20].cpu()
    
    inference_time = time.time() - start_time
    
    avg_metrics = {
        'model_name': model_name,
        'avg_mse': total_mse / total_samples,
        'avg_mae': total_mae / total_samples,
        'avg_psnr': 20 * np.log10(1.0 / np.sqrt(total_mse / total_samples)),
        'inference_time': inference_time,
        'samples_tested': total_samples
    }
    
    return avg_metrics, all_originals, all_reconstructions


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visualize_single_model(originals, reconstructions, model_name, save_path):
    """Create visualization for a single model"""
    fig = plt.figure(figsize=(20, 4))
    
    n_samples = 10
    
    for i in range(n_samples):
        # Original images (row 1)
        ax = plt.subplot(2, n_samples, i + 1)
        plt.imshow(originals[i].numpy().reshape(28, 28), cmap='gray')
        if i == 0:
            ax.set_ylabel('Original', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Reconstructed images (row 2)
        ax = plt.subplot(2, n_samples, i + n_samples + 1)
        plt.imshow(reconstructions[i].numpy().reshape(28, 28), cmap='gray')
        if i == 0:
            ax.set_ylabel('Reconstructed', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle(f'{model_name} - Original vs Reconstructed', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"{model_name} visualization saved to: {save_path}")
    plt.close()


def visualize_comparison(custom_orig, custom_recon, lib_orig, lib_recon, save_path):
    """Create side-by-side comparison visualization"""
    fig = plt.figure(figsize=(20, 8))
    
    n_samples = 10
    
    for i in range(n_samples):
        # Original images (row 1)
        ax = plt.subplot(4, n_samples, i + 1)
        plt.imshow(custom_orig[i].numpy().reshape(28, 28), cmap='gray')
        if i == 0:
            ax.set_ylabel('Original', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Custom autoencoder reconstructions (row 2)
        ax = plt.subplot(4, n_samples, i + n_samples + 1)
        plt.imshow(custom_recon[i].numpy().reshape(28, 28), cmap='gray')
        if i == 0:
            ax.set_ylabel('Custom AE', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Library autoencoder originals (row 3)
        ax = plt.subplot(4, n_samples, i + 2*n_samples + 1)
        plt.imshow(lib_orig[i].numpy().reshape(28, 28), cmap='gray')
        if i == 0:
            ax.set_ylabel('Original', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Library autoencoder reconstructions (row 4)
        ax = plt.subplot(4, n_samples, i + 3*n_samples + 1)
        plt.imshow(lib_recon[i].numpy().reshape(28, 28), cmap='gray')
        if i == 0:
            ax.set_ylabel('Library AE', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Autoencoder Comparison: Custom vs Library-based', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison visualization saved to: {save_path}")
    plt.close()


def create_metrics_comparison_plot(custom_metrics, lib_metrics, save_path):
    """Create bar chart comparing metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_plot = [
        ('avg_mse', 'Mean Squared Error', 'lower is better'),
        ('avg_mae', 'Mean Absolute Error', 'lower is better'),
        ('avg_psnr', 'PSNR (dB)', 'higher is better')
    ]
    
    models = ['Custom\nAutoencoder', 'Library-based\nAutoencoder']
    colors = ['#3498db', '#e74c3c']
    
    for idx, (metric_key, metric_name, direction) in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = [custom_metrics[metric_key], lib_metrics[metric_key]]
        bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Highlight the better performer
        if direction == 'lower is better':
            better_idx = 0 if values[0] < values[1] else 1
        else:
            better_idx = 0 if values[0] > values[1] else 1
        
        bars[better_idx].set_alpha(1.0)
        bars[better_idx].set_edgecolor('gold')
        bars[better_idx].set_linewidth(3)
        
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name}\n({direction})', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Metrics comparison plot saved to: {save_path}")
    plt.close()


def generate_comparison_report(custom_metrics, lib_metrics, custom_params, lib_params, save_path):
    """Generate a detailed text report"""
    report = []
    report.append("=" * 80)
    report.append("AUTOENCODER COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")
    
    report.append("ARCHITECTURE COMPARISON")
    report.append("-" * 80)
    report.append(f"{'Metric':<40} {'Custom AE':<20} {'Library AE':<20}")
    report.append("-" * 80)
    report.append(f"{'Trainable Parameters':<40} {custom_params:<20,} {lib_params:<20,}")
    report.append("")
    
    report.append("PERFORMANCE METRICS")
    report.append("-" * 80)
    report.append(f"{'Metric':<40} {'Custom AE':<20} {'Library AE':<20}")
    report.append("-" * 80)
    report.append(f"{'Mean Squared Error (MSE)':<40} {custom_metrics['avg_mse']:<20.6f} {lib_metrics['avg_mse']:<20.6f}")
    report.append(f"{'Mean Absolute Error (MAE)':<40} {custom_metrics['avg_mae']:<20.6f} {lib_metrics['avg_mae']:<20.6f}")
    report.append(f"{'PSNR (dB)':<40} {custom_metrics['avg_psnr']:<20.2f} {lib_metrics['avg_psnr']:<20.2f}")
    report.append(f"{'Inference Time (seconds)':<40} {custom_metrics['inference_time']:<20.4f} {lib_metrics['inference_time']:<20.4f}")
    report.append(f"{'Samples Tested':<40} {custom_metrics['samples_tested']:<20,} {lib_metrics['samples_tested']:<20,}")
    report.append("")
    
    report.append("ANALYSIS")
    report.append("-" * 80)
    
    # Determine winner for each metric
    mse_winner = "Custom" if custom_metrics['avg_mse'] < lib_metrics['avg_mse'] else "Library"
    mae_winner = "Custom" if custom_metrics['avg_mae'] < lib_metrics['avg_mae'] else "Library"
    psnr_winner = "Custom" if custom_metrics['avg_psnr'] > lib_metrics['avg_psnr'] else "Library"
    speed_winner = "Custom" if custom_metrics['inference_time'] < lib_metrics['inference_time'] else "Library"
    
    report.append(f"• MSE Winner: {mse_winner} Autoencoder")
    report.append(f"• MAE Winner: {mae_winner} Autoencoder")
    report.append(f"• PSNR Winner: {psnr_winner} Autoencoder (higher is better)")
    report.append(f"• Speed Winner: {speed_winner} Autoencoder")
    report.append("")
    
    mse_diff_pct = abs(custom_metrics['avg_mse'] - lib_metrics['avg_mse']) / max(custom_metrics['avg_mse'], lib_metrics['avg_mse']) * 100
    report.append(f"• MSE Difference: {mse_diff_pct:.2f}%")
    
    psnr_diff = abs(custom_metrics['avg_psnr'] - lib_metrics['avg_psnr'])
    report.append(f"• PSNR Difference: {psnr_diff:.2f} dB")
    
    speed_diff_pct = abs(custom_metrics['inference_time'] - lib_metrics['inference_time']) / max(custom_metrics['inference_time'], lib_metrics['inference_time']) * 100
    report.append(f"• Speed Difference: {speed_diff_pct:.2f}%")
    report.append("")
    
    report.append("IMPLEMENTATION COMPARISON")
    report.append("-" * 80)
    report.append("Custom Autoencoder:")
    report.append("  • Uses explicit nn.Module class definition")
    report.append("  • Manually defines each layer in __init__")
    report.append("  • Explicit forward pass with activation functions")
    report.append("  • More verbose but offers finer control")
    report.append("")
    report.append("Library-based Autoencoder:")
    report.append("  • Uses nn.Sequential containers")
    report.append("  • More compact and readable code")
    report.append("  • Higher-level PyTorch abstractions")
    report.append("  • Easier to modify architecture")
    report.append("")
    
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    # Save report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Comparison report saved to: {save_path}")
    
    # Also print to console
    print("\n" + '\n'.join(report))
    
    return report


def main():
    """Main comparison function"""
    print("=" * 80)
    print("AUTOENCODER COMPARISON: Custom vs Library-based")
    print("=" * 80)
    print()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load test dataset
    test_dataset = ta.test_dataset
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
    # Load Custom Autoencoder
    print("Loading Custom Autoencoder...")
    custom_model = AutoEncoder(input_shape=784).to(device)
    try:
        custom_model.load_state_dict(torch.load("../trained-autoencoder", map_location=device))
        print("✓ Custom model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading custom model: {e}")
        return
    
    # Load Library-based Autoencoder
    print("Loading Library-based Autoencoder...")
    lib_model = SequentialAutoencoder(input_dim=784, encoding_dim=128).to(device)
    try:
        lib_model.load_state_dict(torch.load("../trained-library-autoencoder", map_location=device))
        print("✓ Library model loaded successfully")
    except Exception as e:
        print(f"⚠ Library model not found. Training new model...")
        from library_autoencoder import train_library_autoencoder
        lib_model, _, _, _ = train_library_autoencoder(epochs=100, patience=3)
        lib_model = lib_model.to(device)
        print("✓ Library model trained and ready")
    
    print()
    
    # Count parameters
    custom_params = count_parameters(custom_model)
    lib_params = count_parameters(lib_model)
    print(f"Custom Autoencoder parameters: {custom_params:,}")
    print(f"Library Autoencoder parameters: {lib_params:,}")
    print()
    
    # Test Custom Autoencoder
    print("Testing Custom Autoencoder...")
    custom_metrics, custom_orig, custom_recon = test_autoencoder(
        custom_model, test_loader, device, "Custom Autoencoder"
    )
    print("✓ Custom autoencoder testing complete")
    
    # Test Library Autoencoder
    print("Testing Library-based Autoencoder...")
    lib_metrics, lib_orig, lib_recon = test_autoencoder(
        lib_model, test_loader, device, "Library Autoencoder"
    )
    print("✓ Library autoencoder testing complete")
    print()
    
    # Generate outputs
    print("Generating comparison outputs...")
    
    # Create separate visualizations for each model
    visualize_single_model(
        custom_orig, custom_recon, 
        "Custom Autoencoder",
        "../outputs/custom_autoencoder_output.png"
    )
    
    visualize_single_model(
        lib_orig, lib_recon,
        "Library-based Autoencoder",
        "../outputs/library_autoencoder_output.png"
    )
    
    # Create combined comparison visualization
    visualize_comparison(
        custom_orig, custom_recon, lib_orig, lib_recon,
        "../outputs/autoencoder_comparison.png"
    )
    
    create_metrics_comparison_plot(
        custom_metrics, lib_metrics,
        "../outputs/metrics_comparison.png"
    )
    
    # Generate report
    generate_comparison_report(
        custom_metrics, lib_metrics, custom_params, lib_params,
        "../outputs/comparison_report.txt"
    )
    
    # Save metrics as JSON
    comparison_data = {
        'custom_autoencoder': custom_metrics,
        'library_autoencoder': lib_metrics,
        'architecture': {
            'custom_parameters': custom_params,
            'library_parameters': lib_params
        }
    }
    
    with open("../outputs/metrics.json", 'w') as f:
        json.dump(comparison_data, f, indent=4)
    print(f"Metrics JSON saved to: ../outputs/metrics.json")
    
    print()
    print("=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print("\nGenerated files in outputs/ directory:")
    print("  • custom_autoencoder_output.png - Custom autoencoder results")
    print("  • library_autoencoder_output.png - Library-based autoencoder results")
    print("  • autoencoder_comparison.png - Side-by-side comparison")
    print("  • metrics_comparison.png - Performance metrics bar charts")
    print("  • comparison_report.txt - Detailed text report")
    print("  • metrics.json - Machine-readable metrics data")
    print()


if __name__ == "__main__":
    main()
