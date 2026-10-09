# Design Proposal: T.2 Physical-Contact Validation & Actor-Consistent Episode Consolidation

In accordance with the mathematical audit of `Z4kg2Ev3vhk_00`, the current implementation contains two structural flaws:
1. Candidate F169 was admitted under `is_kinematic_closure_corridor` despite $d = 15.75\text{ m}$, $IoU = 0.00$, and positive acceleration $+60.0\text{ m/s}^2$.
2. The episode consolidator merged F128 (actors $\{7, 14\}$) and F169 (actors $\{10, 22\}$) via transitive track chaining, allowing F128's high confidence to borrow F169's ground-truth window proximity.

Below is the formal design proposal containing **ONLY** the two requested modifications. No code has been modified.

---

## Proposal A: T.2 Physical-Contact Validation

### 1. Exact Existing Function / Module to Change
- **Module**: [`road_surveillance/events/candidates/collision.py`](file:///C:/Users/KALAKA~1/Desktop/vision_iq_test1/road_surveillance/events/candidates/collision.py)
- **Functions / Locations**:
  - `CollisionCandidateDetector.process_frame`: `is_kinematic_closure_corridor` calculation (lines 203–239) and candidate trigger gating (lines 496–516).

### 2. Current Behavior
- A candidate is admitted as a kinematic closure corridor if:
  - Parallel heading ($|\Delta\theta| \le 15^\circ$ or $|\Delta\theta - 180^\circ| \le 15^\circ$);
  - Shared lateral corridor ($d_{\text{lateral}} \le 1.8\text{ m}$, $IoU_x \ge 0.50$, or normalized lateral offset $\le 1.5$);
  - Persistent closure ($\ge 3$ consecutive frames);
  - Visual overlap condition: satisfied if $norm\_lat \le 1.2$ even with $IoU_{2D} = 0.00$;
  - Kinematic shock: satisfied if $\min(a_1, a_2) \le -3.0$ **OR** $|a_1| \ge 12.0$ or $|a_2| \ge 12.0$ (admitting positive acceleration spikes $+60\text{ m/s}^2$);
  - Distance limit: $d \le 25.0\text{ m}$.
- **Flaw**: Two non-colliding vehicles cruising in the same lane separated by $15.75\text{ m}$ with zero bounding box overlap and positive acceleration $+60.0\text{ m/s}^2$ qualify as a "collision candidate".

### 3. Proposed Behavior
Require that any parallel closure candidate demonstrate **terminal spatial convergence through explicit physical contact**:
1. **Terminal Physical Contact Condition**:
   - At the closure termination point, the candidate must satisfy at least one physical contact criterion:
     - **Projected 2D Contact**: $IoU_{2D} \ge 0.15$ (or $IoU_{2D} \ge 0.10$ with horizontal $IoU_x \ge 0.50$); **OR**
     - **Calibrated Physical Proximity**: $d_{\text{ground}} \le 2.0\text{ m}$ (for surveyed/calibrated ground plane) or $norm\_pixel\_dist \le 0.50$ (for uncalibrated cameras, requiring bounding boxes within half a vehicle body length).
   - If $IoU_{2D} == 0.00$ and $d_{\text{ground}} > 2.5\text{ m}$, the closure corridor candidate is **strictly rejected**.
2. **Directional Kinematic Shock (Dissipative Impulse)**:
   - Shock must be **decelerative / compressive**:
     $$\min(a_1, a_2) \le -6.0\text{ m/s}^2 \quad \text{or} \quad \text{severe longitudinal braking } a \le -12.0\text{ m/s}^2$$
   - Pure positive acceleration ($a > 0$, e.g. $+60\text{ m/s}^2$) without braking deceleration is rejected as tracker association re-acquisition noise or acceleration away from a corridor.

### 4. Mathematical Rationale
- Physical collision dynamics obey the conservation of momentum and rigid-body non-interpenetration:
  $$\|\mathbf{x}_1 - \mathbf{x}_2\| \le r_1 + r_2$$
  where $r_1, r_2$ are vehicle physical collision boundary radii. Force transmission cannot occur across open space ($d = 15.75\text{ m} \approx 3\text{ vehicle lengths}$).
- In an inelastic collision, the kinetic energy loss $\Delta E_k < 0$ produces a decelerative impulse on the striking vehicle:
  $$J = \int_{t_0}^{t_1} F_{\text{contact}} \, dt = m \Delta v < 0$$
  Positive acceleration ($a = +60\text{ m/s}^2$) corresponds to vehicle thrust or tracker Kalman filter phase-reset, not collision contact force.

### 5. Expected Effect on `Z4kg2Ev3vhk_00`
- **Candidate F169**:
  - $d_{\text{ground}} = 15.75\text{ m} > 2.5\text{ m}$, $IoU = 0.00 < 0.15$, $a = +60.0\text{ m/s}^2 > 0$.
  - **Result**: F169 is **strictly rejected upstream** at the candidate generator.
- **Candidate F128**:
  - $d_{\text{ground}} = 2.71\text{ m}$, $IoU = 0.311 \ge 0.15$, $norm\_pixel\_dist = 0.43 \le 0.50$, $a = -60.0\text{ m/s}^2 \le -12.0\text{ m/s}^2$.
  - **Result**: F128 demonstrates genuine physical contact and severe decelerative impulse, qualifying cleanly.

### 6. Possible False-Positive Failure Modes
- Low-speed creeping bumper-to-bumper contact in dense traffic queues where $IoU > 0.15$ due to high camera pitch angles.
  - *Mitigation*: Existing static hazard / cruising guards and minimum speed drop / deceleration impulse requirements.

### 7. Possible False-Negative Failure Modes
- Oblique rear-end clipping (sideswipe / corner clip) where 2D IoU is small ($IoU \approx 0.08$) and homography ground projection distorts the ground contact point to $d \approx 2.8\text{ m}$.
  - *Mitigation*: Dual-path contact condition allows either $IoU_{2D} \ge 0.15$ OR $norm\_pixel\_dist \le 0.50$ OR $d_{\text{ground}} \le 2.0\text{ m}$.

### 8. Tests Required
- `test_closure_corridor_rejected_when_open_space`: candidate with $d = 15.75\text{ m}$ and $IoU = 0.0$ must return None.
- `test_closure_corridor_rejected_on_positive_acceleration`: candidate with $a = +60.0\text{ m/s}^2$ and no negative acceleration must return None.
- `test_closure_corridor_admitted_on_terminal_physical_contact`: candidate with $IoU = 0.31$ and $a = -60.0\text{ m/s}^2$ must produce `collision_candidate`.

---

## Proposal B: Actor-Consistent Episode Consolidation

### 1. Exact Existing Function / Module to Change
- **Module**: [`modules/episode_consolidator.py`](file:///C:/Users/KALAKA~1/Desktop/vision_iq_test1/modules/episode_consolidator.py)
- **Functions / Locations**:
  - `_standard_relational_merge` (lines 186–234)
  - `should_consolidate_into_active_episode` (lines 236–273)
  - `consolidate_candidates_into_episodes` (lines 276–345)

### 2. Current Behavior
- An episode accumulates all track IDs from any matched candidate:
  `matched_ep.involved_track_ids.append(tid)`
- On subsequent candidate evaluation, candidate tracks are compared against the accumulated set:
  `track_overlap = bool(c_tracks.intersection(ep.involved_track_ids))`
- **Flaw (Transitive Chaining)**:
  - If candidate 1 has $\{7, 10\}$, episode tracks become $\{7, 10\}$.
  - If candidate 2 has $\{10, 11\}$, episode tracks become $\{7, 10, 11\}$.
  - If candidate 3 has $\{10, 14\}$, episode tracks become $\{7, 10, 11, 14\}$.
  - Candidate 4 with $\{7, 14\}$ merges into $\{7, 10, 11, 14\}$.
  - Candidate 5 with $\{10, 22\}$ merges because $10 \in \{7, 10, 11, 14\}$.
  - As a result, Candidate $\{7, 14\}$ (F128) and Candidate $\{10, 22\}$ (F169) are merged into one episode despite $\{7, 14\} \cap \{10, 22\} = \emptyset$!
- **Flaw (Temporal Check Conjunction)**:
  - Line 215 uses `if time_gap > max_dt and frame_gap > int(max_dt * 30): return False`. If frame gap is under 54 frames, time gaps $> 1.8\text{ s}$ are not rejected.

### 3. Proposed Behavior
Enforce **strict actor consistency** for multi-vehicle collision episodes:
1. **Primary Interacting Pair Lock**:
   - A multi-vehicle collision episode must record its **primary collision pair**:
     $$P_{\text{primary}} = \{T_1, T_2\}$$
     defined by the initial impact candidate or the peak saliency contact candidate.
2. **Actor Continuity Constraint**:
   - A new candidate with participating tracks $P_{\text{cand}} = \{T_a, T_b\}$ can merge into the episode **if and only if**:
     - **Exact Continuity**: $P_{\text{cand}} == P_{\text{primary}}$ (same two vehicles continuing their post-collision interaction); **OR**
     - **Secondary Impact / Chain Reaction**: $|P_{\text{cand}} \cap P_{\text{primary}}| \ge 1$ **AND** the secondary candidate occurs within spatial proximity ($\le 4.0\text{ m}$) of the primary impact centroid within $\Delta t \le 1.8\text{ s}$.
   - **Disjoint Actor Rejection**:
     - If $P_{\text{cand}} \cap P_{\text{primary}} == \emptyset$, consolidation into this episode is **strictly prohibited**, even if incidental background tracks overlap through transitive chaining.
3. **Strict Disjunctive Temporal Window**:
   - Enforce $\Delta t \le 1.8\text{ s}$ ($\Delta F \le 54$ frames) disjunctively:
     $$\text{if } \Delta t > 1.8\text{ s} \quad \text{or} \quad \Delta F > 54: \quad \text{return False}$$

### 4. Mathematical Rationale
- Physical collision events are non-transitive:
  $$\text{Collision}(A, B) \land \text{Interaction}(B, C) \not\implies \text{Collision}(A, C)$$
- Allowing transitive set union $\bigcup_{i} P_i$ causes the episode to act as a rolling snowball in dense traffic, aggregating distinct non-colliding vehicle pairs across several seconds.
- Disjoint pair sets $\{7, 14\} \cap \{10, 22\} = \emptyset$ represent independent spatial dynamics. Splicing them creates a chimeric episode where F169 provides temporal proximity to the ground-truth annotation while F128 provides physical impact telemetry.

### 5. Expected Effect on `Z4kg2Ev3vhk_00`
- Episode 5 will anchor on primary actors $\{7, 14\}$ covering the collision at F126–F128.
- Candidate F169 involves $\{10, 22\}$. Because $\{10, 22\} \cap \{7, 14\} = \emptyset$, F169 will **not** be merged into Episode 5.
- Transitive chaining is completely broken. Each physical collision candidate must stand on its own actor-coherent evidence.

### 6. Possible False-Positive Failure Modes
- None. Restricting episode consolidation prevents spurious cross-actor merging and reduces the risk of chimeric false positives.

### 7. Possible False-Negative Failure Modes
- If a multi-vehicle pileup involves a third vehicle ($A$ strikes $B$, then $C$ strikes $B$), but tracker ID switches occur on $B$, the chain reaction might be partitioned into two distinct episodes.
  - *Mitigation*: Secondary impact rule explicitly permits $|P_{\text{cand}} \cap P_{\text{primary}}| \ge 1$ within spatial proximity ($\le 4.0\text{ m}$) of the primary impact centroid.

### 8. Tests Required
- `test_disjoint_actor_pairs_cannot_merge`: candidate $\{A, B\}$ and candidate $\{C, D\}$ must create two distinct episodes even if $\Delta t < 1.0\text{ s}$.
- `test_transitive_chaining_prevented`: candidate sequence $\{A, B\} \to \{B, C\} \to \{C, D\}$ must not merge $\{A, B\}$ with $\{C, D\}$.
- `test_exact_pair_continuation_merged`: candidate $\{A, B\}$ at $t=1.0\text{ s}$ and $\{A, B\}$ at $t=1.5\text{ s}$ must merge into one episode.
- `test_secondary_impact_admitted_with_shared_actor`: primary impact $\{A, B\}$ and secondary impact $\{A, C\}$ within $2.0\text{ m}$ must merge into the pileup episode.

---

## Engineering Checkpoint

No changes have been committed to the codebase. Please review this design proposal and confirm whether to proceed with implementing **Proposal A**, **Proposal B**, or **both**.