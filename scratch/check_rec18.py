"""Evaluate exact Path C v3 logic on Track 246 and Track 229 at Frame 835."""

import json
import numpy as np
import math

with open("scratch/path_c_forensic_results.json") as f:
    forensics = json.load(f)

rec18 = [r for r in forensics if r["rec_id"] == 18][0]
print("Forensic record 18:", rec18)
