"""Qwen3.8-27B Multimodal Verifier Implementation."""

import os
import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
import httpx
import numpy as np

from .base import IncidentVerifier, CandidateEvidencePackage
from .schema import VerificationVerdict, IncidentVerificationResult
from .evidence_extractor import CandidateEvidenceExtractor

logger = logging.getLogger("road_surveillance.verification.qwen_verifier")


SYSTEM_PROMPT = """You are an expert Traffic Incident Verification Specialist in an Intelligent Transportation System (ITS).
Your role is to strictly verify whether candidate incident alarms flagged by upstream deterministic video analytics represent genuine traffic incidents or false alarms.

You will be provided visual surveillance evidence and/or kinematic telemetry.
You must output a strict JSON object with:
- "verdict": Exactly one of:
    * "CONFIRMED_INCIDENT": Direct physical collision (crumple, body contact, debris), vehicle rollover/upset, or violent roadway barrier impact.
    * "REJECTED_FALSE_ALARM": Non-incident (e.g. normal vehicle following, parallel lane merge, crossing vehicles clearing without impact, or stationary resting wreckage already stopped from an earlier crash).
    * "UNCERTAIN": Camera resolution, severe darkness, glare, or occlusion prevents definitive visual verification.
- "confidence": Float between 0.0 and 1.0.
- "detected_crash_type": Incident category (e.g. "t_bone", "rear_end", "single_vehicle_barrier", "rollover", "sideswipe", "none").
- "rationale": Concise, 1-2 sentence technical explanation referencing visual observations across the keyframes and kinematic telemetry.

Output ONLY valid JSON. Do not wrap in conversational markdown outside the JSON object."""


def _load_dotenv_if_exists():
    for candidate_path in [".env", os.path.join(os.path.dirname(__file__), "../../.env")]:
        abs_path = os.path.abspath(candidate_path)
        if os.path.exists(abs_path):
            with open(abs_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k not in os.environ:
                            os.environ[k] = v
            break


class QwenMultimodalVerifier(IncidentVerifier):
    """Downstream incident verifier powered by Qwen3.8-27B multimodal VLM."""

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 45.0,
        mock_mode: bool = False,
        mock_default_verdict: VerificationVerdict = VerificationVerdict.CONFIRMED_INCIDENT,
        ablation_level: str = "H4",
    ):
        _load_dotenv_if_exists()
        self.ablation_level = ablation_level.upper()
        self.api_base = (
            api_base
            or os.environ.get("QWEN_API_BASE")
            or os.environ.get("OPENAI_API_BASE")
            or "http://localhost:8000/v1"
        )
        self.api_key = api_key or os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.model = model or os.environ.get("QWEN_MODEL") or "Qwen/Qwen2.5-VL-72B-Instruct"
        self.temperature = temperature
        self.timeout = timeout
        self.mock_mode = mock_mode
        self.mock_default_verdict = mock_default_verdict

    def _build_telemetry_text(self, evidence: CandidateEvidencePackage) -> str:
        """Format upstream physical measurements for cross-modal consistency prompt."""
        tel = evidence.telemetry_summary
        if not tel:
            return "No numerical kinematics available for this candidate episode."

        lines = [
            f"Event Type: {evidence.candidate_detector}",
            f"Candidate Frame: {evidence.candidate_frame}",
            f"Involved Tracks: {evidence.involved_track_ids}",
        ]
        if "tca_sec" in tel:
            lines.append(f"Time-to-Collision (TCA): {tel['tca_sec']:.2f} s")
        if "dca_m" in tel:
            lines.append(f"Distance-at-Closest-Approach (DCA): {tel['dca_m']:.2f} m")
        if "speed_kmh" in tel:
            lines.append(f"Estimated Speed: {tel['speed_kmh']:.1f} km/h")
        if "accel_ms2" in tel or "acceleration_mag" in tel:
            a = tel.get("accel_ms2", tel.get("acceleration_mag", 0.0))
            lines.append(f"Signed Longitudinal Acceleration: {a:+.1f} m/s^2")
        if "yaw_rate_deg_s" in tel:
            lines.append(f"Yaw Rotation Rate: {tel['yaw_rate_deg_s']:.1f} deg/s")

        return "\n".join(lines)

    def _parse_vlm_response(self, text: str, evidence: CandidateEvidencePackage) -> IncidentVerificationResult:
        """Parse and sanitize JSON from the VLM response."""
        clean_text = text.strip()
        if clean_text.startswith("```"):
            clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
            clean_text = re.sub(r"\s*```$", "", clean_text)
            clean_text = clean_text.strip()

        try:
            data = json.loads(clean_text)
            verdict_str = str(data.get("verdict", "UNCERTAIN")).strip().upper()
            if verdict_str in [v.value for v in VerificationVerdict]:
                verdict = VerificationVerdict(verdict_str)
            else:
                verdict = VerificationVerdict.UNCERTAIN

            return IncidentVerificationResult(
                event_id=evidence.event_id,
                video_id=evidence.video_id,
                candidate_frame=evidence.candidate_frame,
                verdict=verdict,
                confidence=float(data.get("confidence", 0.7)),
                rationale=str(data.get("rationale", "")),
                detected_crash_type=str(data.get("detected_crash_type", "unknown")),
                involved_tracks=evidence.involved_track_ids,
                raw_response=text,
                telemetry_summary=evidence.telemetry_summary,
            )
        except Exception as e:
            logger.warning(f"Failed to parse VLM response as JSON: {e}. Raw text: {text[:200]}")
            return IncidentVerificationResult(
                event_id=evidence.event_id,
                video_id=evidence.video_id,
                candidate_frame=evidence.candidate_frame,
                verdict=VerificationVerdict.UNCERTAIN,
                confidence=0.0,
                rationale=f"VLM JSON parsing error: {e}",
                detected_crash_type="unknown",
                involved_tracks=evidence.involved_track_ids,
                raw_response=text,
                telemetry_summary=evidence.telemetry_summary,
            )

    def verify(self, evidence: CandidateEvidencePackage) -> IncidentVerificationResult:
        """Verify a single candidate evidence package across controlled H0-H4 ladder."""
        if self.mock_mode:
            return IncidentVerificationResult(
                event_id=evidence.event_id,
                video_id=evidence.video_id,
                candidate_frame=evidence.candidate_frame,
                verdict=self.mock_default_verdict,
                confidence=0.95,
                rationale="Mock verification verdict for testing",
                detected_crash_type="mock_incident",
                involved_tracks=evidence.involved_track_ids,
                raw_response='{"verdict": "CONFIRMED_INCIDENT", "confidence": 0.95}',
                telemetry_summary=evidence.telemetry_summary,
            )

        extractor = CandidateEvidenceExtractor()
        images_to_attach: List[Tuple[str, np.ndarray]] = []
        prompt_intro: str = ""

        # Global overview frame (original full canvas at t0)
        global_t0 = None
        if evidence.full_frames and "t0" in evidence.full_frames and evidence.full_frames["t0"].size > 0:
            global_t0 = evidence.full_frames["t0"]
        elif evidence.keyframes and "trigger" in evidence.keyframes and evidence.keyframes["trigger"].size > 0:
            global_t0 = evidence.keyframes["trigger"]

        if self.ablation_level == "H0":
            # H0: Full frames only (3 frames: t-1, t0, t+1)
            prompt_intro = (
                "Evaluate this traffic surveillance video sequence (3 full-canvas keyframes).\n"
                "Analyze whether a traffic accident, near-miss, or normal traffic flow occurs.\n"
                "Provide your verdict, confidence, crash type, and concise visual rationale."
            )
            frames_src = evidence.full_frames if evidence.full_frames else evidence.keyframes
            for k in (["t-1", "t0", "t+1"] if evidence.full_frames else ["pre", "trigger", "post"]):
                img = frames_src.get(k)
                if img is not None:
                    images_to_attach.append((k, img))

        elif self.ablation_level == "H1":
            # H1: Full + Clean ROI (Global full frame + 3 clean focused 384x384 ROI crops)
            prompt_intro = (
                "Evaluate this traffic incident candidate with two views:\n"
                "- Image 1: Global Scene Overview (Original full canvas)\n"
                "- Images 2-4: High-resolution clean 384x384 focused ROI crops (t-1 approach, t0 conflict, t+1 resolution)\n"
                "Do these frames show a vehicle collision, near-miss, or normal traffic?"
            )
            if global_t0 is not None:
                images_to_attach.append(("global_t0", global_t0))
            crops_src = evidence.roi_crops if evidence.roi_crops else evidence.keyframes
            for k in (["t-1", "t0", "t+1"] if evidence.roi_crops else ["pre", "trigger", "post"]):
                img = crops_src.get(k)
                if img is not None:
                    images_to_attach.append((k, img))

        elif self.ablation_level == "H2":
            # H2: Full + ROI + Set-of-Mark (Global full frame + 3 marked ROI crops with track badges)
            prompt_intro = (
                f"Evaluate this traffic incident candidate with target vehicle tracking badges:\n"
                f"- Image 1: Global Scene Overview (Original full canvas)\n"
                f"- Images 2-4: Focused 384x384 ROI crops with Set-of-Mark badges (Track IDs: {evidence.involved_track_ids})\n"
                "Did a collision or physical impact occur between these highlighted vehicles?"
            )
            if global_t0 is not None:
                images_to_attach.append(("global_t0", global_t0))
            som_src = evidence.som_crops if evidence.som_crops else (evidence.roi_crops if evidence.roi_crops else evidence.keyframes)
            for k in (["t-1", "t0", "t+1"] if (evidence.som_crops or evidence.roi_crops) else ["pre", "trigger", "post"]):
                img = som_src.get(k)
                if img is not None:
                    images_to_attach.append((k, img))

        elif self.ablation_level == "H3":
            # H3: Full + Temporal Clean ROI Sequence (Global full frame + 5 dense clean temporal crops)
            prompt_intro = (
                "Evaluate this dense 5-frame temporal sequence of the incident region:\n"
                "- Image 1: Global Scene Overview (Original full canvas)\n"
                "- Images 2-6: Dense clean temporal ROI sequence:\n"
                "  * Image 2: t-2 (Pre-approach window)\n"
                "  * Image 3: t-1 (Immediate approach)\n"
                "  * Image 4: t0  (Potential impact / conflict)\n"
                "  * Image 5: t+1 (Immediate resolution)\n"
                "  * Image 6: t+2 (Post-incident outcome)\n"
                "Analyze the motion dynamics and determine if an impact or crash occurred."
            )
            if global_t0 is not None:
                images_to_attach.append(("global_t0", global_t0))
            crops_src = evidence.roi_crops if evidence.roi_crops else evidence.keyframes
            for k in (["t-2", "t-1", "t0", "t+1", "t+2"] if evidence.roi_crops else ["pre", "trigger", "post"]):
                img = crops_src.get(k)
                if img is not None:
                    images_to_attach.append((k, img))

        else:  # H4: Full + Temporal ROI Sequence + Set-of-Mark + Upstream Telemetry
            telemetry_text = self._build_telemetry_text(evidence)
            prompt_intro = (
                "CROSS-MODAL CONSISTENCY VERIFICATION:\n\n"
                f"### Upstream Physical Measurements:\n{telemetry_text}\n\n"
                "### Visual Sequence:\n"
                "- Image 1: Global Scene Overview (Original full canvas at conflict point t0)\n"
                "- Images 2-6: High-Resolution 384x384 Focused ROI sequence with Set-of-Mark tracking badges:\n"
                "  * Image 2: t-2 (Pre-approach window)\n"
                "  * Image 3: t-1 (Immediate approach)\n"
                "  * Image 4: t0  (Point of maximum physical conflict)\n"
                "  * Image 5: t+1 (Immediate resolution / deflection)\n"
                "  * Image 6: t+2 (Post-incident outcome)\n\n"
                "Question: Does the visual evidence support the physical event described by the measurements?"
            )
            if global_t0 is not None:
                images_to_attach.append(("global_t0", global_t0))
            som_src = evidence.som_crops if evidence.som_crops else (evidence.roi_crops if evidence.roi_crops else evidence.keyframes)
            for k in (["t-2", "t-1", "t0", "t+1", "t+2"] if (evidence.som_crops or evidence.roi_crops) else ["pre", "trigger", "post"]):
                img = som_src.get(k)
                if img is not None:
                    images_to_attach.append((k, img))

        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt_intro}
        ]

        for lbl, img in images_to_attach:
            if img is not None and isinstance(img, np.ndarray) and img.size > 0:
                try:
                    b64_str = extractor.encode_image_to_base64(img)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_str}",
                        },
                    })
                except Exception as e:
                    logger.warning(f"Error encoding frame '{lbl}' to base64: {e}")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": 512,
        }

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                res_data = response.json()
                raw_content = res_data["choices"][0]["message"]["content"]
                return self._parse_vlm_response(raw_content, evidence)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.warning(f"Could not connect to Qwen endpoint at {url}: {e}")
            return IncidentVerificationResult(
                event_id=evidence.event_id,
                video_id=evidence.video_id,
                candidate_frame=evidence.candidate_frame,
                verdict=VerificationVerdict.UNCERTAIN,
                confidence=0.0,
                rationale=f"Connection error to VLM endpoint: {e}",
                detected_crash_type="unknown",
                involved_tracks=evidence.involved_track_ids,
                raw_response="",
                telemetry_summary=evidence.telemetry_summary,
            )
        except Exception as e:
            logger.warning(f"Error calling Qwen endpoint: {e}")
            return IncidentVerificationResult(
                event_id=evidence.event_id,
                video_id=evidence.video_id,
                candidate_frame=evidence.candidate_frame,
                verdict=VerificationVerdict.UNCERTAIN,
                confidence=0.0,
                rationale=f"VLM call failure: {e}",
                detected_crash_type="unknown",
                involved_tracks=evidence.involved_track_ids,
                raw_response="",
                telemetry_summary=evidence.telemetry_summary,
            )
