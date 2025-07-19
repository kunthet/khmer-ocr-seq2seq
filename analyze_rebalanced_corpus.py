#!/usr/bin/env python3
"""
Quick analysis of the rebalanced corpus length distribution.
"""

import sys
from pathlib import Path
from collections import defaultdict

def analyze_rebalanced_corpus(corpus_dir="data/processed_rebalanced"):
    """Analyze the length distribution of the rebalanced corpus."""
    
    corpus_dir = Path(corpus_dir)
    
    # Length categories matching the rebalancing script (reduced to 150)
    length_categories = [
        (1, 5, "class_001_005"),
        (6, 10, "class_006_010"),
        (11, 15, "class_011_015"),
        (16, 20, "class_016_020"),
        (21, 25, "class_021_025"),
        (26, 30, "class_026_030"),
        (31, 35, "class_031_035"),
        (36, 40, "class_036_040"),
        (41, 45, "class_041_045"),
        (46, 50, "class_046_050"),
        (51, 55, "class_051_055"),
        (56, 60, "class_056_060"),
        (61, 65, "class_061_065"),
        (66, 70, "class_066_070"),
        (71, 75, "class_071_075"),
        (76, 80, "class_076_080"),
        (81, 85, "class_081_085"),
        (86, 90, "class_086_090"),
        (91, 95, "class_091_095"),
        (96, 100, "class_096_100"),
        (101, 105, "class_101_105"),
        (106, 110, "class_106_110"),
        (111, 115, "class_111_115"),
        (116, 120, "class_116_120"),
        (121, 125, "class_121_125"),
        (126, 130, "class_126_130"),
        (131, 135, "class_131_135"),
        (136, 140, "class_136_140"),
        (141, 145, "class_141_145"),
        (146, 150, "class_146_150"),
    ]
    
    print("REBALANCED CORPUS LENGTH DISTRIBUTION ANALYSIS (1-150 RANGE)")
    print("======================================================================")
    
    for file_name in ["train_0.txt", "train_1.txt"]:
        file_path = corpus_dir / file_name
        if not file_path.exists():
            print(f"File {file_name} not found!")
            continue
            
        print(f"\n{file_name.upper()}:")
        print("-" * 50)
        
        # Count samples by length
        category_counts = defaultdict(int)
        total_samples = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    length = len(line)
                    total_samples += 1
                    
                    # Categorize by length
                    for min_len, max_len, category_name in length_categories:
                        if min_len <= length <= max_len:
                            category_counts[category_name] += 1
                            break
        
        print(f"Total Samples: {total_samples:,}")
        print(f"\n{'Category':<15} {'Range':<10} {'Count':<10} {'%':<8}")
        print("-" * 50)
        
        # Only show classes that have data
        for min_len, max_len, category_name in length_categories:
            count = category_counts[category_name]
            if count > 0:
                percentage = (count / total_samples * 100) if total_samples > 0 else 0
                range_str = f"{min_len}-{max_len}"
                print(f"{category_name:<15} {range_str:<10} {count:<10,} {percentage:<7.1f}%")
    
    # Combined analysis
    print(f"\nCOMBINED REBALANCED ANALYSIS:")
    print("-" * 50)
    
    combined_category_counts = defaultdict(int)
    combined_total = 0
    
    for file_name in ["train_0.txt", "train_1.txt"]:
        file_path = corpus_dir / file_name
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        length = len(line)
                        combined_total += 1
                        
                        for min_len, max_len, category_name in length_categories:
                            if min_len <= length <= max_len:
                                combined_category_counts[category_name] += 1
                                break
    
    print(f"Total Combined Samples: {combined_total:,}")
    print(f"\n{'Category':<15} {'Range':<10} {'Count':<10} {'%':<8}")
    print("-" * 50)
    
    # Only show classes that have data
    classes_with_data = 0
    for min_len, max_len, category_name in length_categories:
        count = combined_category_counts[category_name]
        if count > 0:
            classes_with_data += 1
            percentage = (count / combined_total * 100) if combined_total > 0 else 0
            range_str = f"{min_len}-{max_len}"
            print(f"{category_name:<15} {range_str:<10} {count:<10,} {percentage:<7.1f}%")
    
    print("SUMMARY:")
    print(f"  Classes with data: {classes_with_data}/{len(length_categories)}")
    coverage = (classes_with_data / len(length_categories) * 100) if len(length_categories) > 0 else 0
    print(f"  Coverage: {coverage:.1f}% of 1-150 range")
    if classes_with_data > 0:
        print(f"  Avg samples per active class: {combined_total // classes_with_data:,}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        corpus_dir = sys.argv[1]
    else:
        corpus_dir = "data/processed_rebalanced"
    
    analyze_rebalanced_corpus(corpus_dir) 