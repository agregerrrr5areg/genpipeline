import cProfile
import pstats
import torch
import numpy as np
import logging
from genpipeline.config import load_config
from genpipeline.optimiser import run_optimisation
from io import StringIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def profile_bo_loop():
    config = load_config("pipeline_config.json")
    config.n_optimisation_iterations = 10 # More iterations for better data
    
    pr = cProfile.Profile()
    pr.enable()
    
    logger.info("Starting profiled BO loop...")
    run_optimisation(config)
    
    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    
    print("\n" + "="*50)
    print("PROFILING RESULTS (Top 30 by Cumulative Time)")
    print("="*50)
    print(s.getvalue())

if __name__ == "__main__":
    profile_bo_loop()
