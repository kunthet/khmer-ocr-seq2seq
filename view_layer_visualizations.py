#!/usr/bin/env python3
"""
Layer Visualization Browser

A simple script to browse and display the generated layer visualizations.
"""

import os
import sys
from pathlib import Path
import argparse

def list_visualizations(vis_dir):
    """List all available visualizations organized by category."""
    vis_path = Path(vis_dir)
    
    if not vis_path.exists():
        print(f"Visualization directory not found: {vis_dir}")
        return
    
    print("="*80)
    print(f"LAYER VISUALIZATIONS SUMMARY: {vis_dir}")
    print("="*80)
    
    # Get all subdirectories
    categories = [d for d in vis_path.iterdir() if d.is_dir()]
    categories.sort()
    
    total_files = 0
    
    for category in categories:
        png_files = list(category.glob("*.png"))
        total_files += len(png_files)
        
        print(f"\n{category.name.upper().replace('_', ' ')} ({len(png_files)} images):")
        print("-" * 60)
        
        # Separate weight visualizations from distribution plots
        weight_files = [f for f in png_files if '_weights.png' in f.name]
        dist_files = [f for f in png_files if '_distribution.png' in f.name]
        other_files = [f for f in png_files if f not in weight_files and f not in dist_files]
        
        if weight_files:
            print(f"  Weight Visualizations ({len(weight_files)}):")
            for f in sorted(weight_files)[:10]:  # Show first 10
                layer_name = f.name.replace('_weights.png', '').replace('_', '.')
                print(f"    • {layer_name}")
            if len(weight_files) > 10:
                print(f"    ... and {len(weight_files) - 10} more")
        
        if dist_files:
            print(f"  Distribution Plots ({len(dist_files)}):")
            for f in sorted(dist_files)[:5]:  # Show first 5
                layer_name = f.name.replace('_distribution.png', '').replace('_', '.')
                print(f"    • {layer_name}")
            if len(dist_files) > 5:
                print(f"    ... and {len(dist_files) - 5} more")
        
        if other_files:
            print(f"  Other Visualizations ({len(other_files)}):")
            for f in sorted(other_files):
                print(f"    • {f.name}")
    
    print(f"\n{'='*80}")
    print(f"TOTAL: {total_files} visualization files generated")
    print(f"{'='*80}")

def show_layer_info(vis_dir, layer_name):
    """Show information about a specific layer's visualizations."""
    vis_path = Path(vis_dir)
    
    # Find files matching the layer name
    matching_files = []
    for category_dir in vis_path.iterdir():
        if category_dir.is_dir():
            for png_file in category_dir.glob("*.png"):
                if layer_name.lower() in png_file.name.lower():
                    matching_files.append(png_file)
    
    if not matching_files:
        print(f"No visualizations found for layer: {layer_name}")
        return
    
    print(f"\nFound {len(matching_files)} visualizations for '{layer_name}':")
    print("-" * 60)
    
    for f in sorted(matching_files):
        file_size = f.stat().st_size / 1024  # Size in KB
        print(f"  • {f.name} ({file_size:.1f} KB)")
        print(f"    Path: {f}")
    
    print(f"\nTo view these images, use any image viewer or open them directly.")

def create_html_gallery(vis_dir, output_file="layer_gallery.html"):
    """Create an HTML gallery to view all visualizations in a web browser."""
    vis_path = Path(vis_dir)
    
    if not vis_path.exists():
        print(f"Visualization directory not found: {vis_dir}")
        return
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Neural Network Layer Visualizations</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .category {{ margin-bottom: 30px; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .category h2 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
            .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
            .image-item {{ text-align: center; background-color: #fafafa; padding: 10px; border-radius: 5px; }}
            .image-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
            .image-item h4 {{ margin: 10px 0 5px 0; font-size: 14px; color: #555; }}
            .overview {{ background-color: #e8f5e8; }}
            .stats {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>🧠 Neural Network Layer Visualizations</h1>
        <div class="stats">
            <strong>Checkpoint:</strong> {vis_path.name}<br>
            <strong>Generated:</strong> {len(list(vis_path.rglob('*.png')))} visualization images
        </div>
    """
    
    # Get all categories
    categories = [d for d in vis_path.iterdir() if d.is_dir()]
    categories.sort()
    
    for category in categories:
        png_files = list(category.glob("*.png"))
        if not png_files:
            continue
        
        category_class = "overview" if category.name == "overview" else ""
        html_content += f"""
        <div class="category {category_class}">
            <h2>{category.name.replace('_', ' ').title()} ({len(png_files)} images)</h2>
            <div class="image-grid">
        """
        
        for png_file in sorted(png_files):
            rel_path = png_file.relative_to(vis_path)
            image_title = png_file.stem.replace('_', ' ').title()
            html_content += f"""
                <div class="image-item">
                    <h4>{image_title}</h4>
                    <img src="{rel_path}" alt="{image_title}" onclick="window.open('{rel_path}', '_blank')">
                </div>
            """
        
        html_content += """
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML file
    output_path = vis_path / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML gallery created: {output_path}")
    print(f"Open this file in your web browser to view all visualizations.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Browse layer visualizations")
    parser.add_argument("vis_dir", help="Visualization directory")
    parser.add_argument("--layer", help="Show info for specific layer")
    parser.add_argument("--html", action="store_true", help="Create HTML gallery")
    
    args = parser.parse_args()
    
    if args.html:
        create_html_gallery(args.vis_dir)
    elif args.layer:
        show_layer_info(args.vis_dir, args.layer)
    else:
        list_visualizations(args.vis_dir)

if __name__ == "__main__":
    main() 