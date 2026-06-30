import os
from dataprepro2 import process_batch
import pandas as pd

# Diverse list of 20 known TESS targets (some confirmed planets, some EBs, some quiet)
REAL_TICS = [
    261136679, # Pi Mensae c (Planet)
    25155310,  # WASP-126 b (Planet)
    144065872, # TOI-119 b
    279741379, # TOI-114 b
    281541555, # TOI-134 b
    410153553, # TOI-111
    350622204, # Known EB
    100100827, # EB
    441462736, # Blend / EB
    307210830, # EB
    # Let's add 10 more random TICs from Sector 1
    # Note: These are arbitrary valid TICs, the pipeline will evaluate them.
    100100828,
    100100829,
    141914082,
    141914083,
    141914084,
    150428135,
    150428136,
    150428137,
    16005254,
    16005255
]

if __name__ == '__main__':
    print("================================================================")
    print(" RUNNING REAL DATASET BATCH (20 Targets via MAST)")
    print("================================================================")
    
    # Process batch with network download enabled
    # Sector 1 is a good default for these early TICs
    batch_df = process_batch(
        tic_list=REAL_TICS,
        sector=1,
        n_workers=4, # Parallel
        use_network=True,
        make_plot=False
    )
    
    print("\nBatch Processing Complete!")
    print(f"Output saved to: ./tess_pipeline_output/batch_sector1_results.csv")
    print(batch_df.head())
