import numpy as np
import os

def save_map(map_points, filepath):
    np.save(filepath, map_points)

def save_reports(reports, filepath):
    with open(filepath, 'w') as f:
        for report in reports:
            f.write(f"{report['compliance']},{report['details']}\n")
