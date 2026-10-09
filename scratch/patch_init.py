with open('modules/__init__.py', 'r', encoding='utf-8') as f:
    code = f.read()

patch_import = '''from .evidence_builder import (
    select_temporal_evidence_frames,
    render_actor_overlays,
    create_stage_t_composite_strip,
)
from .vlm_arbitrator import ('''

patch_all = '''    " select_temporal_evidence_frames\,
 \render_actor_overlays\,
 \create_stage_t_composite_strip\,
 \create_evidence_strip\,'''

code = code.replace('from .vlm_arbitrator import (', patch_import)
code = code.replace(' \create_evidence_strip\,', patch_all)

with open('modules/__init__.py', 'w', encoding='utf-8') as f:
 f.write(code)
print('SUCCESS: modules/__init__.py updated')
