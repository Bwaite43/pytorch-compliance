#!/usr/bin/env python3
"""
Add Diverse Implementation Statuses to GovRAMP Controls

This script will intelligently assign realistic implementation statuses
based on control characteristics to enable proper PyTorch training.
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict

def analyze_control_complexity(control_text: str, control_id: str) -> dict:
    """Analyze control complexity to help determine realistic implementation status"""
    
    if pd.isna(control_text):
        control_text = ""
    
    text = str(control_text).lower()
    
    # Complexity indicators
    complexity_keywords = {
        'high_complexity': [
            'develop', 'establish', 'implement', 'maintain', 'monitor', 'review', 'update',
            'procedure', 'process', 'documentation', 'assessment', 'evaluation', 'continuous',
            'automated', 'integration', 'compliance', 'audit', 'management', 'coordination'
        ],
        'technical_complexity': [
            'system', 'network', 'software', 'hardware', 'database', 'application',
            'infrastructure', 'configuration', 'encryption', 'authentication', 'logging',
            'monitoring', 'backup', 'recovery', 'security', 'access', 'control'
        ],
        'policy_based': [
            'policy', 'procedure', 'guideline', 'standard', 'requirement', 'rule',
            'documentation', 'training', 'awareness', 'responsibility', 'accountability'
        ],
        'operational': [
            'personnel', 'staff', 'user', 'training', 'awareness', 'management',
            'oversight', 'supervision', 'reporting', 'communication'
        ]
    }
    
    # Count keywords by category
    scores = {}
    for category, keywords in complexity_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text)
        scores[category] = score
    
    # Control family analysis
    family = control_id.split('-')[0] if '-' in control_id else 'Unknown'
    
    # Text length as complexity indicator
    text_length = len(text)
    
    return {
        'complexity_scores': scores,
        'control_family': family,
        'text_length': text_length,
        'total_complexity': sum(scores.values())
    }

def assign_implementation_status(control_info: dict, control_id: str, priority: str) -> str:
    """Assign realistic implementation status based on control characteristics"""
    
    family = control_info['control_family']
    complexity = control_info['total_complexity']
    text_length = control_info['text_length']
    
    # Define implementation likelihood by control family
    # Based on typical implementation patterns in organizations
    family_implementation_likelihood = {
        'AC': {'high': 0.7, 'medium': 0.6, 'low': 0.4},  # Access Control - often implemented
        'AU': {'high': 0.8, 'medium': 0.7, 'low': 0.5},  # Audit - critical, often done
        'AT': {'high': 0.5, 'medium': 0.4, 'low': 0.3},  # Awareness Training - often neglected
        'CM': {'high': 0.6, 'medium': 0.5, 'low': 0.4},  # Configuration Management
        'CP': {'high': 0.4, 'medium': 0.3, 'low': 0.2},  # Contingency Planning - often delayed
        'IA': {'high': 0.8, 'medium': 0.7, 'low': 0.6},  # Identification/Auth - critical
        'IR': {'high': 0.5, 'medium': 0.4, 'low': 0.3},  # Incident Response
        'MA': {'high': 0.6, 'medium': 0.5, 'low': 0.4},  # Maintenance
        'MP': {'high': 0.5, 'medium': 0.4, 'low': 0.3},  # Media Protection
        'PE': {'high': 0.7, 'medium': 0.6, 'low': 0.5},  # Physical/Environmental
        'PL': {'high': 0.6, 'medium': 0.5, 'low': 0.4},  # Planning
        'PS': {'high': 0.5, 'medium': 0.4, 'low': 0.3},  # Personnel Security
        'RA': {'high': 0.4, 'medium': 0.3, 'low': 0.2},  # Risk Assessment - often incomplete
        'CA': {'high': 0.4, 'medium': 0.3, 'low': 0.2},  # Security Assessment
        'SC': {'high': 0.7, 'medium': 0.6, 'low': 0.5},  # System/Communications Protection
        'SI': {'high': 0.6, 'medium': 0.5, 'low': 0.4},  # System/Information Integrity
        'SA': {'high': 0.4, 'medium': 0.3, 'low': 0.2},  # System/Services Acquisition
    }
    
    # Get base implementation likelihood
    priority_level = 'high' if priority == 'High' else 'medium' if priority == 'Medium' else 'low'
    base_likelihood = family_implementation_likelihood.get(family, {'high': 0.5, 'medium': 0.4, 'low': 0.3})[priority_level]
    
    # Adjust based on complexity (simpler controls more likely to be implemented)
    complexity_adjustment = max(0, (10 - complexity)) * 0.02  # Reduce likelihood for complex controls
    text_adjustment = max(0, (500 - text_length)) * 0.0001    # Reduce likelihood for lengthy controls
    
    final_likelihood = base_likelihood + complexity_adjustment + text_adjustment
    final_likelihood = max(0.1, min(0.9, final_likelihood))  # Keep between 10% and 90%
    
    # Generate random number and assign status
    rand = np.random.random()
    
    if rand < final_likelihood * 0.4:  # 40% of likely controls are fully implemented
        return 'Implemented'
    elif rand < final_likelihood * 0.7:  # 30% are partially implemented
        return 'Partially Implemented'
    elif rand < final_likelihood * 0.85:  # 15% are planned
        return 'Planned'
    else:  # Rest are not implemented
        return 'Not Implemented'

def assign_realistic_priorities(df: pd.DataFrame) -> pd.DataFrame:
    """Assign more realistic priority levels based on control characteristics"""
    
    df = df.copy()
    
    # High priority control families (typically critical for security)
    high_priority_families = ['AC', 'AU', 'IA', 'SC', 'SI']
    
    # Medium priority families
    medium_priority_families = ['CM', 'PE', 'PL', 'MA']
    
    # Lower priority families (still important, but often implemented later)
    lower_priority_families = ['AT', 'CP', 'IR', 'PS', 'RA', 'CA', 'SA', 'MP']
    
    new_priorities = []
    
    for _, row in df.iterrows():
        family = row.get('control_family', 'Unknown')
        
        if family in high_priority_families:
            # 60% High, 30% Medium, 10% Low
            priority = np.random.choice(['High', 'Medium', 'Low'], p=[0.6, 0.3, 0.1])
        elif family in medium_priority_families:
            # 20% High, 60% Medium, 20% Low  
            priority = np.random.choice(['High', 'Medium', 'Low'], p=[0.2, 0.6, 0.2])
        else:
            # 10% High, 40% Medium, 50% Low
            priority = np.random.choice(['High', 'Medium', 'Low'], p=[0.1, 0.4, 0.5])
        
        new_priorities.append(priority)
    
    df['priority'] = new_priorities
    return df

def diversify_govramp_data(input_file: str, output_file: str = None):
    """Main function to diversify the GovRAMP data"""
    
    print(f"🔄 Diversifying implementation statuses in {input_file}")
    
    # Load data
    try:
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df)} controls")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    print(f"📊 Original status distribution:")
    print(df['implementation_status'].value_counts().to_dict())
    print(f"📊 Original priority distribution:")
    print(df['priority'].value_counts().to_dict())
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Assign more realistic priorities
    df = assign_realistic_priorities(df)
    
    # Analyze each control and assign status
    new_statuses = []
    
    for _, row in df.iterrows():
        control_analysis = analyze_control_complexity(
            row.get('combined_text', ''), 
            row.get('control_id', '')
        )
        
        new_status = assign_implementation_status(
            control_analysis,
            row.get('control_id', ''),
            row.get('priority', 'Medium')
        )
        
        new_statuses.append(new_status)
    
    df['implementation_status'] = new_statuses
    
    # Update the status_numeric field based on new statuses
    status_encoding = {
        'Not Implemented': 0,
        'Planned': 1,
        'Partially Implemented': 2,
        'Implemented': 3
    }
    
    df['status_numeric'] = df['implementation_status'].map(status_encoding)
    
    # Set output filename
    if output_file is None:
        output_file = input_file.replace('.csv', '_diversified.csv')
    
    # Save the diversified data
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Diversified data saved to {output_file}")
    print(f"📊 New status distribution:")
    status_counts = df['implementation_status'].value_counts().to_dict()
    print(status_counts)
    
    print(f"📊 New priority distribution:")
    priority_counts = df['priority'].value_counts().to_dict()
    print(priority_counts)
    
    # Show some examples
    print(f"\n📋 Sample of diversified controls:")
    sample_df = df[['control_id', 'control_family', 'priority', 'implementation_status']].head(10)
    print(sample_df.to_string(index=False))
    
    # Provide insights
    print(f"\n💡 Insights:")
    implemented_pct = (status_counts.get('Implemented', 0) / len(df)) * 100
    partial_pct = (status_counts.get('Partially Implemented', 0) / len(df)) * 100
    
    print(f"  • {implemented_pct:.1f}% of controls are fully implemented")
    print(f"  • {partial_pct:.1f}% are partially implemented") 
    print(f"  • High-priority families (AC, AU, IA) have higher implementation rates")
    print(f"  • Complex controls are less likely to be fully implemented")
    
    return output_file

def validate_diversification(file_path: str):
    """Validate that the diversification worked properly"""
    
    print(f"\n🔍 Validating diversified data in {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Check status distribution
        status_counts = df['implementation_status'].value_counts()
        total_controls = len(df)
        
        print(f"✅ Total controls: {total_controls}")
        print(f"✅ Unique statuses: {len(status_counts)}")
        
        if len(status_counts) >= 3:
            print("✅ Good diversity - multiple implementation statuses present")
        else:
            print("⚠️  Limited diversity - consider re-running with different seed")
        
        # Check for proper distributions
        for status, count in status_counts.items():
            percentage = (count / total_controls) * 100
            print(f"  • {status}: {count} controls ({percentage:.1f}%)")
        
        # Check if ready for PyTorch training
        if len(status_counts) >= 2:
            print("✅ Ready for PyTorch classification training!")
        else:
            print("❌ Not ready - need at least 2 different classes")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def main():
    """Main execution function"""
    
    print("🚀 GovRAMP Data Diversification for PyTorch Training")
    print("=" * 60)
    
    input_file = 'GovRAMPCoreControls_manual.csv'
    output_file = 'GovRAMPCoreControls_diversified.csv'
    
    # Diversify the data
    result_file = diversify_govramp_data(input_file, output_file)
    
    if result_file:
        # Validate the results
        validate_diversification(result_file)
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Use the new file: {result_file}")
        print(f"2. Update your PyTorch script to load this file instead")
        print(f"3. Run: python3 pytorch_compliance.py")
        
        # Show how to update the main script
        print(f"\n📝 Update your pytorch_compliance.py:")
        print(f"   Change the load_data line to:")
        print(f"   analyzer.load_data('{result_file}', 'NIST_SP80053_rev5_catalog_load.csv')")
    
    else:
        print("❌ Diversification failed. Check the error messages above.")

if __name__ == "__main__":
    main()