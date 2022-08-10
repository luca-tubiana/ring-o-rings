#!/usr/bin/env python
import sys
from pathlib import Path
proj_name = 'ring-o-rings/github-code'
home = Path(__file__).resolve().home()
root_dir = home / ('Projects/' + proj_name)
print(root_dir)
sys.path.insert(0, str(root_dir / 'src'))

from ror.analysis_pipeline import AnalysisPipeline
if __name__ == '__main__':
    input_dir = root_dir / ('data/02-raw/')
    output_dir = root_dir / 'data/03-processed/COM_twn/'
    output_dir.mkdir(parents=True,  exist_ok=True)
    for sim in input_dir.glob('cat_M*Lk*'):
        print(f"Now analyzing {sim}")
        files = list(sim.glob('traj*trj'))
        if len(files)>0:
            # Switch the normals at every red ring, matching a twisted ribbon
            pipeline = AnalysisPipeline(sim, verse=-1, twist_normals=True)
            # avoid re-running previous analyses if results present
            out = output_dir / str(pipeline)
            out_files = list(out.glob('*'))
            if len(out_files) == 3:
                print(f'---Analysis of {sim} already performed, skipping.')
            else:
                pipeline.save_vtf_file(output_dir)
                pipeline.save_local_properties(output_dir)
                pipeline.save_global_properties(output_dir)
        else:
            print(f'---Missing data in {sim}, skipping.')
