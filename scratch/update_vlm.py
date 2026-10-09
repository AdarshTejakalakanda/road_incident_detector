import re, os

with open('modules/vlm_arbitrator.py', 'r', encoding='utf-8') as f:
    text = f.read()

exports = '''
from modules.evidence_builder import (
