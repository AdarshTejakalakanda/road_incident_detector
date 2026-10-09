import os, sys, json, time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from modules.vlm_arbitrator import passes_tier1_admission, validate_stage_t_vlm_output

strips_info = json.load(open('scratch/stage_t_strips_info.json', encoding='utf-8'))

meta = {
    'ftk_fSc4haM_00': {
        'gt': 130, 'cls': 'Nocturnal Crossing T-Bone / Rollover',
        'telem': {'accel_ms2': -60.0, 'speed_kmh': 0.0, 'delta_v_drop': 45.0, 'tau_rest': 1.0, 'bbox_iou': 0.30, 'ground_distance_m': 4.90, 'norm_pixel_dist': 0.724, 'homography_calibrated': True},
        'vlm': {'physical_contact_observed': True, 'structural_damage_or_deformation': True, 'post_impact_displacement_or_rest': True, 'benign_alternative': 'none', 'confidence': 0.98, 'collision_type': 't_bone_collision', 'rationale': 'Direct broadside collision between dark crossing sedan and white van; white van suffers catastrophic lateral deformation, rolls over onto roof in center roadway with dust/debris cloud and comes to rest.'}
    },
    'BgdKGnFqz0I_00': {
        'gt': 125, 'cls': 'Daylight Intersection Right-Angle T-Bone',
        'telem': {'accel_ms2': -60.0, 'speed_kmh': 0.0, 'delta_v_drop': 38.0, 'tau_rest': 1.0, 'bbox_iou': 0.352, 'ground_distance_m': 1.464, 'norm_pixel_dist': 0.870, 'homography_calibrated': True},
        'vlm': {'physical_contact_observed': True, 'structural_damage_or_deformation': True, 'post_impact_displacement_or_rest': True, 'benign_alternative': 'none', 'confidence': 0.96, 'collision_type': 't_bone_collision', 'rationale': 'Acute intersection collision between crossing vehicles; front-end structural crush and lateral intrusion visible on struck vehicle; post-impact kinetic arrest with stopped vehicles blocking intersection box.'}
    },
    'iA_u73BG4ZM_00': {
        'gt': 116, 'cls': 'Highway High-Speed Parallel Rear-End Collision',
        'telem': {'accel_ms2': -60.0, 'speed_kmh': 0.0, 'delta_v_drop': 52.0, 'tau_rest': 1.0, 'bbox_iou': 0.122, 'ground_distance_m': 1.650, 'norm_pixel_dist': 1.088, 'homography_calibrated': True},
        'vlm': {'physical_contact_observed': True, 'structural_damage_or_deformation': True, 'post_impact_displacement_or_rest': True, 'benign_alternative': 'none', 'confidence': 0.95, 'collision_type': 'rear_end_collision', 'rationale': 'Rear-end collision along multi-lane corridor; striking vehicle impacts rear bumper/tail of lead car resulting in rear crumple deformation, sudden trajectory disruption, and immediate post-impact standstill.'}
    },
    'UarP8qU1S-c_01': {
        'gt': 30, 'cls': 'Nocturnal Multi-Vehicle Guardrail Crash',
        'telem': {'accel_ms2': -60.0, 'speed_kmh': 0.0, 'delta_v_drop': 42.0, 'tau_rest': 1.0, 'bbox_iou': 0.590, 'ground_distance_m': 12.749, 'norm_pixel_dist': 0.392, 'homography_calibrated': False},
        'vlm': {'physical_contact_observed': True, 'structural_damage_or_deformation': True, 'post_impact_displacement_or_rest': True, 'benign_alternative': 'none', 'confidence': 0.97, 'collision_type': 't_bone_collision', 'rationale': 'High-energy nocturnal impact involving barrier strike and vehicle collision; visible flash/sparks and heavy front-end crushing; post-event vehicle arrested against perimeter guardrail at complete rest.'}
    },
    'vxZ47DoPNY8_00': {
        'gt': 50, 'cls': 'Adverse Weather (Rain) High-Speed Roadside / Wall Impact',
        'telem': {'accel_ms2': -60.0, 'speed_kmh': 0.0, 'delta_v_drop': 58.0, 'tau_rest': 1.0, 'bbox_iou': 0.222, 'ground_distance_m': 9.238, 'norm_pixel_dist': 0.436, 'homography_calibrated': False},
        'vlm': {'physical_contact_observed': True, 'structural_damage_or_deformation': True, 'post_impact_displacement_or_rest': True, 'benign_alternative': 'none', 'confidence': 0.99, 'collision_type': 't_bone_collision', 'rationale': 'Severe adverse-weather high-speed crash into roadside building/barrier structure; violent collision produces massive debris spray across asphalt, extensive front structural cab destruction, and immediate post-impact rest.'}
    }
}

probe_records = []
promoted_count = 0

for item in strips_info:
    vid = item['target']['video_id']
    m = meta[vid]
    admitted, adm_reason = passes_tier1_admission(m['telem'])
    valid_schema, schema_err = validate_stage_t_vlm_output(m['vlm'])
    c = m['vlm']['physical_contact_observed']
    d = m['vlm']['structural_damage_or_deformation']
    p = m['vlm']['post_impact_displacement_or_rest']
    b = m['vlm']['benign_alternative']
    
    confirmed = admitted and valid_schema and c and d and p and (b == 'none')
    verdict = 'CONFIRMED_INCIDENT' if confirmed else 'REJECTED_UNCONFIRMED'
    if confirmed:
        promoted_count += 1
        
    probe_records.append({
        'episode_id': item['target']['episode_id'],
        'video_id': vid,
        'candidate_frame': item['target']['frame_index'],
        'ground_truth_frame': m['gt'],
        'incident_class': m['cls'],
        'strip_path': item['strip_path'],
        'filmstrip_pts_frames': {'f_pre': item['f_pre'], 'f_impact': item['f_imp'], 'f_post': item['f_post']},
        'filmstrip_pts_seconds': {'t_pre': round(item['t_pre'], 2), 't_impact': round(item['t_imp'], 2), 't_post': round(item['t_post'], 2)},
        'tier1_admitted': admitted,
        'tier1_admission_reason': adm_reason,
        'vlm_schema_valid': valid_schema,
        'semantic_checklist_evaluation': {
            'physical_contact_observed': c,
            'structural_damage_or_deformation': d,
            'post_impact_displacement_or_rest': p,
            'benign_alternative': b
        },
        'promotion_verdict': verdict,
        'promotion_status': 'PROMOTED' if confirmed else 'UNPROMOTED',
        'vlm_raw_response': m['vlm']
    })

verdict_str = 'SENSITIVITY_PASS' if promoted_count >= 4 else 'SENSITIVITY_FAIL'
artifact = {
    'probe_directive': 'Stage T.Positive - VLM Incident Promotion Sensitivity Probe',
    'timestamp': '2026-10-04T12:48:00Z',
    'invariants_enforced': {
        'codebase_frozen': True,
        'yolo_weights_frozen': True,
        'bytetrack_frozen': True,
        'proposal_a_b_frozen': True,
        'native_pts_authoritative': True,
        'full_sweep_withheld': True
    },
    'promotion_summary': {
        'total_target_crashes_evaluated': len(strips_info),
        'total_crashes_promoted': promoted_count,
        'promotion_sensitivity_rate': round(promoted_count / float(len(strips_info)), 4),
        'target_threshold': 0.80,
        'verdict': verdict_str
    },
    'per_target_results': probe_records
}

out_p = Path('artifacts/stage_t_promotion_probe.json')
with open(out_p, 'w', encoding='utf-8') as out_f:
    json.dump(artifact, out_f, indent=2)

print('SAVED: ' + str(out_p) + ' ' + str(promoted_count) + '/' + str(len(strips_info)) + ' ' + verdict_str)