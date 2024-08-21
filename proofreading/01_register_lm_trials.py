# proofread_step01.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tifffile as tiff
from proofreading_base import ProofreadingBase
from scripts.utils.image_utils import load_tiff_as_hyperstack
from scripts.sample_db import SampleDB

class ProofreadRegisterLMTrials(ProofreadingBase):
    def __init__(self):
        super().__init__('01_register_lm_trials')

    def load_sample_data(self, sample_id, db_path):
        sample_db = SampleDB()
        sample_db.load(db_path)
        exp = sample_db.get_sample(sample_id)
        print(exp)

        # Load stacks
        processed_folder = os.path.join(exp.paths.trials_path, "processed")
        raw_sum_path =  os.path.join(processed_folder, f"sum_raw_trials_{sample_id}.tif")
        raw_trial = tiff.imread(raw_sum_path)
        aligned_sum_path = os.path.join(processed_folder, f"sum_rigid_corrected_trials_{sample_id}.tif")
        aligned_trial = tiff.imread(aligned_sum_path)

        motion_corrected_trial_path = os.path.join(processed_folder, f"motion_corrected_{sample_id}_trial_001.tif")
        motion_corrected_trial = load_tiff_as_hyperstack(motion_corrected_trial_path, n_slices=exp.params_lm.n_planes,
                                                         doubling=True)

        return {
            'raw': raw_trial,
            'aligned': aligned_trial,
            'motion_corrected': motion_corrected_trial
        }


if __name__ == "__main__":
    proofreader = ProofreadRegisterLMTrials()
    proofreader.main()
