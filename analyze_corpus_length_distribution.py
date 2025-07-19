#!/usr/bin/env python3
"""
Khmer OCR Training Corpus Length Distribution Analysis

This script analyzes the character length distribution of training corpus data
and categorizes text samples into predefined length ranges.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import seaborn as sns

class CorpusLengthAnalyzer:
    """Analyzes length distribution of corpus text data."""
    
    def __init__(self, corpus_dir="data/processed_round_robin_1_150"):
        """
        Initialize the analyzer.
        
        Args:
            corpus_dir (str): Directory containing corpus files
        """
        self.corpus_dir = Path(corpus_dir)
        
        # Length categories as specified
        self.length_categories = [
            (1, 5, "very_short"),
            (6, 10, "short"), 
            (11, 20, "medium"),
            (21, 50, "long"),
            (51, 100, "very_long"),
            (101, 150, "extra_long"),
            (151, float('inf'), "extreme_long")
        ]
        
        self.data = {
            'train_0': [],
            'train_1': [],
            'combined': []
        }
        
        self.stats = {}
        
    def load_corpus_data(self):
        """Load and process corpus data from training files."""
        print("Loading corpus data...")
        
        # Process train_0.txt
        train_0_file = self.corpus_dir / "train_0.txt"
        if train_0_file.exists():
            print(f"Processing {train_0_file}...")
            with open(train_0_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        length = len(line)
                        self.data['train_0'].append({
                            'text': line,
                            'length': length,
                            'line_num': line_num
                        })
                        
                        if line_num % 10000 == 0:
                            print(f"  Processed {line_num:,} lines from train_0.txt")
        
        # Process train_1.txt
        train_1_file = self.corpus_dir / "train_1.txt"
        if train_1_file.exists():
            print(f"Processing {train_1_file}...")
            with open(train_1_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        length = len(line)
                        self.data['train_1'].append({
                            'text': line,
                            'length': length,
                            'line_num': line_num
                        })
                        
                        if line_num % 10000 == 0:
                            print(f"  Processed {line_num:,} lines from train_1.txt")
        
        # Combine all data
        self.data['combined'] = self.data['train_0'] + self.data['train_1']
        
        print(f"Total samples loaded:")
        print(f"  train_0.txt: {len(self.data['train_0']):,}")
        print(f"  train_1.txt: {len(self.data['train_1']):,}")
        print(f"  Combined: {len(self.data['combined']):,}")
        
    def categorize_by_length(self, dataset_name='combined'):
        """
        Categorize texts by length ranges.
        
        Args:
            dataset_name (str): Which dataset to analyze ('train_0', 'train_1', or 'combined')
        
        Returns:
            dict: Statistics for each length category
        """
        if dataset_name not in self.data:
            raise ValueError(f"Dataset '{dataset_name}' not found")
            
        data = self.data[dataset_name]
        categorized = defaultdict(list)
        
        for item in data:
            length = item['length']
            
            # Find appropriate category
            for min_len, max_len, category_name in self.length_categories:
                if min_len <= length <= max_len:
                    categorized[category_name].append(item)
                    break
        
        # Calculate statistics for each category
        category_stats = {}
        total_samples = len(data)
        
        for min_len, max_len, category_name in self.length_categories:
            samples = categorized[category_name]
            count = len(samples)
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            
            if samples:
                lengths = [s['length'] for s in samples]
                avg_length = np.mean(lengths)
                median_length = np.median(lengths)
                std_length = np.std(lengths)
                min_length = min(lengths)
                max_length = max(lengths)
            else:
                avg_length = median_length = std_length = min_length = max_length = 0
            
            range_str = f"{min_len}-{max_len}" if max_len != float('inf') else f"{min_len}+"
            
            category_stats[category_name] = {
                'range': range_str,
                'count': count,
                'percentage': percentage,
                'avg_length': avg_length,
                'median_length': median_length,
                'std_length': std_length,
                'min_length': min_length,
                'max_length': max_length,
                'samples': samples[:5]  # Store first 5 samples for inspection
            }
        
        return category_stats
        
    def analyze_all_datasets(self):
        """Analyze length distribution for all datasets."""
        print("\nAnalyzing length distributions...")
        
        for dataset_name in ['train_0', 'train_1', 'combined']:
            if self.data[dataset_name]:
                print(f"\nAnalyzing {dataset_name}...")
                self.stats[dataset_name] = self.categorize_by_length(dataset_name)
                
    def print_summary_report(self):
        """Print a comprehensive summary report."""
        print("\n" + "="*80)
        print("KHMER OCR TRAINING CORPUS LENGTH DISTRIBUTION ANALYSIS")
        print("="*80)
        
        for dataset_name, stats in self.stats.items():
            print(f"\n{dataset_name.upper()} DATASET:")
            print("-" * 40)
            
            total_samples = sum(cat['count'] for cat in stats.values())
            print(f"Total Samples: {total_samples:,}")
            
            # Print category breakdown
            print(f"\n{'Category':<15} {'Range':<10} {'Count':<10} {'%':<8} {'Avg Len':<8} {'Med Len':<8}")
            print("-" * 70)
            
            for min_len, max_len, category_name in self.length_categories:
                if category_name in stats:
                    cat_stats = stats[category_name]
                    print(f"{category_name:<15} {cat_stats['range']:<10} {cat_stats['count']:<10,} "
                          f"{cat_stats['percentage']:<7.1f}% {cat_stats['avg_length']:<7.1f} "
                          f"{cat_stats['median_length']:<7.1f}")
            
            # Overall statistics
            if self.data[dataset_name]:
                all_lengths = [item['length'] for item in self.data[dataset_name]]
                print(f"\nOverall Statistics:")
                print(f"  Mean Length: {np.mean(all_lengths):.1f}")
                print(f"  Median Length: {np.median(all_lengths):.1f}")
                print(f"  Std Dev: {np.std(all_lengths):.1f}")
                print(f"  Min Length: {min(all_lengths)}")
                print(f"  Max Length: {max(all_lengths)}")
                print(f"  95th Percentile: {np.percentile(all_lengths, 95):.1f}")
                print(f"  99th Percentile: {np.percentile(all_lengths, 99):.1f}")
    
    def create_visualization(self, output_dir="corpus_analysis_results"):
        """Create visualization plots for length distribution."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\nCreating visualizations in {output_dir}/...")
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Khmer OCR Training Corpus Length Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Category distribution (bar chart)
        ax1 = axes[0, 0]
        if 'combined' in self.stats:
            categories = []
            counts = []
            percentages = []
            
            for min_len, max_len, category_name in self.length_categories:
                if category_name in self.stats['combined']:
                    categories.append(category_name)
                    counts.append(self.stats['combined'][category_name]['count'])
                    percentages.append(self.stats['combined'][category_name]['percentage'])
            
            bars = ax1.bar(categories, counts, alpha=0.7)
            ax1.set_title('Distribution by Length Category')
            ax1.set_xlabel('Length Category')
            ax1.set_ylabel('Number of Samples')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add percentage labels on bars
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{pct:.1f}%', ha='center', va='bottom')
        
        # 2. Length histogram
        ax2 = axes[0, 1]
        if self.data['combined']:
            lengths = [item['length'] for item in self.data['combined']]
            ax2.hist(lengths, bins=50, alpha=0.7, edgecolor='black')
            ax2.set_title('Length Distribution Histogram')
            ax2.set_xlabel('Text Length (characters)')
            ax2.set_ylabel('Frequency')
            ax2.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
            ax2.axvline(np.median(lengths), color='green', linestyle='--', label=f'Median: {np.median(lengths):.1f}')
            ax2.legend()
        
        # 3. Cumulative distribution
        ax3 = axes[1, 0]
        if self.data['combined']:
            lengths = sorted([item['length'] for item in self.data['combined']])
            cumulative_pct = np.arange(1, len(lengths) + 1) / len(lengths) * 100
            ax3.plot(lengths, cumulative_pct)
            ax3.set_title('Cumulative Length Distribution')
            ax3.set_xlabel('Text Length (characters)')
            ax3.set_ylabel('Cumulative Percentage')
            ax3.grid(True, alpha=0.3)
            
            # Add percentile markers
            for percentile in [50, 90, 95, 99]:
                value = np.percentile(lengths, percentile)
                ax3.axvline(value, color='red', alpha=0.5, linestyle='--')
                ax3.text(value, percentile, f'P{percentile}', rotation=90, ha='right')
        
        # 4. Comparison between train_0 and train_1
        ax4 = axes[1, 1]
        if 'train_0' in self.stats and 'train_1' in self.stats:
            categories = []
            train_0_counts = []
            train_1_counts = []
            
            for min_len, max_len, category_name in self.length_categories:
                if category_name in self.stats['train_0'] and category_name in self.stats['train_1']:
                    categories.append(category_name)
                    train_0_counts.append(self.stats['train_0'][category_name]['count'])
                    train_1_counts.append(self.stats['train_1'][category_name]['count'])
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax4.bar(x - width/2, train_0_counts, width, label='train_0', alpha=0.7)
            ax4.bar(x + width/2, train_1_counts, width, label='train_1', alpha=0.7)
            
            ax4.set_title('train_0 vs train_1 Distribution')
            ax4.set_xlabel('Length Category')
            ax4.set_ylabel('Number of Samples')
            ax4.set_xticks(x)
            ax4.set_xticklabels(categories, rotation=45)
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'corpus_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved as {output_dir}/corpus_length_distribution.png")
    
    def save_detailed_results(self, output_dir="corpus_analysis_results"):
        """Save detailed analysis results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving detailed results to {output_dir}/...")
        
        # Save summary statistics as JSON
        summary_data = {}
        for dataset_name, stats in self.stats.items():
            summary_data[dataset_name] = {}
            for category, cat_stats in stats.items():
                # Remove sample texts for JSON serialization
                summary_stats = {k: v for k, v in cat_stats.items() if k != 'samples'}
                summary_data[dataset_name][category] = summary_stats
        
        with open(output_dir / 'length_distribution_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # Save detailed samples for each category
        for dataset_name, stats in self.stats.items():
            with open(output_dir / f'{dataset_name}_sample_texts.txt', 'w', encoding='utf-8') as f:
                f.write(f"SAMPLE TEXTS FROM {dataset_name.upper()}\n")
                f.write("="*60 + "\n\n")
                
                for category, cat_stats in stats.items():
                    f.write(f"{category.upper()} ({cat_stats['range']}):\n")
                    f.write(f"Count: {cat_stats['count']:,} ({cat_stats['percentage']:.1f}%)\n")
                    f.write(f"Sample texts (first 5):\n")
                    
                    for i, sample in enumerate(cat_stats['samples'], 1):
                        f.write(f"  {i}. [{sample['length']} chars] {sample['text'][:100]}{'...' if len(sample['text']) > 100 else ''}\n")
                    
                    f.write("\n" + "-"*40 + "\n\n")
        
        print(f"Results saved:")
        print(f"  - Summary: {output_dir}/length_distribution_summary.json")
        print(f"  - Sample texts: {output_dir}/*_sample_texts.txt")
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting Khmer OCR Training Corpus Length Distribution Analysis...")
        
        # Load data
        self.load_corpus_data()
        
        if not any(self.data.values()):
            print("No data found! Please check that corpus files exist in the specified directory.")
            return
        
        # Analyze distributions
        self.analyze_all_datasets()
        
        # Generate reports
        self.print_summary_report()
        
        # Create visualizations
        self.create_visualization()
        
        # Save detailed results
        self.save_detailed_results()
        
        print(f"\nAnalysis complete! Check 'corpus_analysis_results/' for detailed outputs.")


def main():
    """Main function to run the corpus length distribution analysis."""
    analyzer = CorpusLengthAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main() 