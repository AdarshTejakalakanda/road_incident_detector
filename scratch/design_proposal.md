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

### 3. Proposed Beh
<truncated 6879 bytes>
represent independent spatial dynamics. Splicing them creates a chimeric episode where F169 provides temporal proximity to the ground-truth annotation while F128 provides physical impact telemetry.

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