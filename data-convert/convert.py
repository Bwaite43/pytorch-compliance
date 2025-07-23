import pandas as pd
import numpy as np

def debug_excel_structure(excel_file: str):
    """
    Debug Excel file structure to understand the layout
    """
    print(f"🔍 Debugging Excel file: {excel_file}")
    print("=" * 60)
    
    # Load all sheets
    try:
        excel_data = pd.read_excel(excel_file, sheet_name=None, header=None)
        
        for sheet_name, df in excel_data.items():
            print(f"\n📋 SHEET: {sheet_name}")
            print(f"Shape: {df.shape}")
            print("-" * 40)
            
            # Show first 15 rows to understand structure
            for i in range(min(15, len(df))):
                row_data = []
                for j, cell in enumerate(df.iloc[i]):
                    if pd.notna(cell):
                        cell_str = str(cell)[:50]  # Truncate long text
                        if len(str(cell)) > 50:
                            cell_str += "..."
                        row_data.append(f"Col{j}: {cell_str}")
                    else:
                        row_data.append(f"Col{j}: [EMPTY]")
                
                if any("EMPTY" not in item for item in row_data):  # Only show rows with data
                    print(f"Row {i:2d}: {' | '.join(row_data[:5])}")  # Show first 5 columns
                    
                    # If this looks like a control ID, highlight it
                    first_cell = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""
                    if any(pattern in first_cell.upper() for pattern in ['AC-', 'AU-', 'AT-', 'CM-', 'CP-']):
                        print(f"     ⭐ POTENTIAL CONTROL ROW")
            
            print()
    
    except Exception as e:
        print(f"❌ Error reading Excel: {e}")

def simple_extract_controls(excel_file: str):
    """
    Simple extraction focusing on finding control data
    """
    print(f"🎯 Simple control extraction from {excel_file}")
    
    # Try the controls sheet specifically
    try:
        df = pd.read_excel(excel_file, sheet_name='GovRAMP Mod Core Controls', header=None)
        
        controls_found = []
        
        # Scan all rows for control IDs
        for i in range(len(df)):
            for j in range(min(5, len(df.columns))):  # Check first 5 columns
                cell_value = df.iloc[i, j]
                if pd.notna(cell_value):
                    cell_str = str(cell_value).strip()
                    
                    # Check if this looks like a control ID
                    if any(pattern in cell_str.upper() for pattern in ['AC-', 'AU-', 'AT-', 'CM-', 'CP-', 'IA-', 'IR-', 'MA-', 'MP-', 'PE-', 'PL-', 'PS-', 'RA-', 'CA-', 'SC-', 'SI-', 'SA-']):
                        
                        # Extract this row's data
                        row_data = {}
                        row_data['row_index'] = i
                        row_data['control_id'] = cell_str
                        
                        # Try to get description from nearby columns
                        for k in range(j+1, min(j+5, len(df.columns))):
                            next_cell = df.iloc[i, k]
                            if pd.notna(next_cell) and len(str(next_cell).strip()) > 10:
                                row_data['description'] = str(next_cell).strip()
                                break
                        
                        controls_found.append(row_data)
                        break
        
        print(f"🎯 Found {len(controls_found)} potential controls:")
        for ctrl in controls_found[:10]:  # Show first 10
            desc = ctrl.get('description', 'No description')[:100]
            print(f"  Row {ctrl['row_index']:2d}: {ctrl['control_id']} - {desc}")
        
        if len(controls_found) > 10:
            print(f"  ... and {len(controls_found) - 10} more")
        
        return controls_found
        
    except Exception as e:
        print(f"❌ Error in simple extraction: {e}")
        return []

def create_manual_csv(excel_file: str, controls_data: list):
    """
    Manually create CSV from extracted control data
    """
    if not controls_data:
        print("❌ No control data to convert")
        return
    
    print(f"📝 Creating manual CSV from {len(controls_data)} controls...")
    
    # Load the raw sheet to get full row data
    df_raw = pd.read_excel(excel_file, sheet_name='GovRAMP Mod Core Controls', header=None)
    
    # Create structured data
    structured_data = []
    
    for ctrl in controls_data:
        row_idx = ctrl['row_index']
        row = df_raw.iloc[row_idx]
        
        # Extract data from this row
        control_record = {
            'control_id': ctrl['control_id'],
            'framework': 'GovRAMP'
        }
        
        # Try to extract description and guidance from the row
        text_fields = []
        for col_idx in range(len(row)):
            cell = row.iloc[col_idx]
            if pd.notna(cell) and len(str(cell).strip()) > 20:  # Substantial text
                text_fields.append(str(cell).strip())
        
        # Assign fields based on position and content
        if len(text_fields) >= 1:
            control_record['description'] = text_fields[0]
        if len(text_fields) >= 2:
            control_record['guidance'] = text_fields[1]
        
        # Combine all text for ML
        control_record['combined_text'] = ' '.join(text_fields)
        
        # Add default values
        control_record['priority'] = 'Medium'
        control_record['implementation_status'] = 'Not Implemented'
        
        structured_data.append(control_record)
    
    # Create DataFrame and save
    df_manual = pd.DataFrame(structured_data)
    
    # Add ML features
    df_manual['control_family'] = df_manual['control_id'].apply(
        lambda x: x.split('-')[0] if '-' in x else 'Unknown'
    )
    df_manual['combined_text_length'] = df_manual['combined_text'].str.len()
    
    output_file = excel_file.replace('.xlsx', '_manual.csv')
    df_manual.to_csv(output_file, index=False)
    
    print(f"✅ Manual CSV saved to {output_file}")
    print(f"📊 Shape: {df_manual.shape}")
    print(f"📋 Columns: {list(df_manual.columns)}")
    
    # Show sample
    print(f"\n📋 Sample data:")
    print(df_manual.head(3).to_string())
    
    return output_file

if __name__ == "__main__":
    excel_file = "GovRAMPCoreControls.xlsx"
    
    # Step 1: Debug the structure
    debug_excel_structure(excel_file)
    
    print("\n" + "="*60)
    
    # Step 2: Simple extraction
    controls = simple_extract_controls(excel_file)
    
    print("\n" + "="*60)
    
    # Step 3: Create manual CSV
    if controls:
        csv_file = create_manual_csv(excel_file, controls)
        print(f"\n🎯 READY! Use this file for PyTorch: {csv_file}")