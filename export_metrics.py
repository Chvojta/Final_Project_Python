import openpyxl
import numpy as np
import os
from datetime import datetime
from io import BytesIO

def export_ebitda_bridge(metrics):
    """
    Export the EBITDA bridge data to an in-memory Excel file without saving to disk.
    
    Args:
        metrics (dict): The metrics dictionary containing ebitda_bridge data
    
    Returns:
        BytesIO: In-memory Excel file that can be used with Flask's send_file
    """
    # Check if ebitda_bridge exists in metrics
    if 'ebitda_bridge' not in metrics:
        raise ValueError("The metrics dictionary does not contain EBITDA bridge data")
    
    # Create a new workbook
    wb = openpyxl.Workbook()
    
    # Use the default sheet
    sheet = wb.active
    sheet.title = "EBITDA Bridge"
    
    # Add headers for the bridge data
    sheet['A1'] = "Component"
    sheet['B1'] = "Value"
    sheet['C1'] = "Running Total"
    
    # Add the EBITDA bridge data
    ebitda_bridge = metrics['ebitda_bridge']
    for i, item in enumerate(ebitda_bridge):
        row = i + 2  # Start from row 2
        sheet[f'A{row}'] = item['label']
        sheet[f'B{row}'] = item['value']
        
        if 'running_total' in item:
            sheet[f'C{row}'] = item['running_total']
    

    output = BytesIO()
    wb.save(output)
    output.seek(0) 
    
    return output

