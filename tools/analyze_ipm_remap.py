#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze the IPM remap data from tusimple_ipm_remap.yml
"""
import os
import yaml
import numpy as np

################################################################################
# Yaml content example
# remap_ipm_x: !!opencv-matrix
#    rows: 640
#    cols: 640
#    dt: f
#    data: [ 5.62353492e-01, 2.81194115e+00, 5.05978918e+00,
#     ...
#     ]
################################################################################
def load_and_analyze_remap():
    """
    Load and analyze the IPM remap data
    """
    # Load YAML file
    yaml_path = './data/tusimple_ipm_remap.yml'
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Cannot find {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        remap_data = yaml.safe_load(f)
        
    # Get remap data x
    # 
    remap_ipm_x = remap_data['remap_ipm_x']
    # get remap_ipm_x.rows
    x_rows = remap_ipm_x['rows']
    x_cols = remap_ipm_x['cols']
    x_dt = remap_ipm_x['dt']
    x_data = remap_ipm_x['data']
    
    print(f"remap_ipm_x.rows: {x_rows}")
    print(f"remap_ipm_x.cols: {x_cols}")
    print(f"remap_ipm_x.dt: {x_dt}")
    print(f"len(remap_ipm_x.data): {len(x_data)}")

    # Get remap data y
    remap_ipm_y = remap_data['remap_ipm_y']
    y_rows = remap_ipm_y['rows']
    y_cols = remap_ipm_y['cols']
    y_dt = remap_ipm_y['dt']
    y_data = remap_ipm_y['data']
    
    print(f"remap_ipm_y.rows: {y_rows}")
    print(f"remap_ipm_y.cols: {y_cols}")
    print(f"remap_ipm_y.dt: {y_dt}")
    print(f"len(remap_ipm_y.data): {len(y_data)}")

if __name__ == "__main__":
    try:
        load_and_analyze_remap()
    except Exception as e:
        print(f"Error: {str(e)}") 