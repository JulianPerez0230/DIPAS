# -*- coding: utf-8 -*-
# Quick test script to run the first 2 variants in the CFD Dataset Builder
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset_cfd_builder import CFDDatasetBuilder

def run_test():
    builder = CFDDatasetBuilder()
    # Ejecutamos solo 2 variantes (16 simulaciones en total) para validar
    builder.build_dataset(total_variants=2)

if __name__ == "__main__":
    run_test()
