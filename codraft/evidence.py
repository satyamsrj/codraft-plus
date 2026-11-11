import re
import uuid
from typing import List, Dict, Optional
from codraft.types import Claim

class EvidenceLedger:
    def __init__(self):
        self.claims: Dict[str, Claim] = {}

    def extract_claims(self, text: str, source_model: str, version: int) -> List[Claim]:
        clean_text = re.sub(r"```[\s\S]*?```", "", text)
        raw_sentences = re.split(r"(?<=[.!?])\s+", clean_text)
        extracted: List[Claim] = []

        for s in raw_sentences:
            s_clean = s.strip()
            if not s_clean or s_clean.startswith("#") or s_clean.endswith("?") or len(s_clean.split()) < 5:
                continue
            
            has_fact_marker = bool(
                re.search(r"\d+|is\s+a|was\s+a|defined\s+as|consists\s+of|located\s+in|founded\s+in|causes|results\s+in", s_clean, re.IGNORECASE)
            )

            is_dup = any(c.statement.lower() == s_clean.lower() for c in self.claims.values())
            if not is_dup and has_fact_marker:
                cid = f"claim_{len(self.claims) + 1}_{uuid.uuid4().hex[:4]}"
                claim = Claim(
                    claim_id=cid,
                    statement=s_clean,
                    source_model=source_model,
                    verification_status="unverified",
                    confidence=0.5,
                    created_at_version=version
                )
                self.claims[cid] = claim
                extracted.append(claim)

        return extracted

    def verify_claim(self, claim_id: str, status: str, confidence: float, notes: Optional[str] = None):
        if claim_id in self.claims:
            claim = self.claims[claim_id]
            claim.verification_status = status  # type: ignore
            claim.confidence = confidence
            if notes:
                claim.evidence_notes = notes

    def compute_verification_pressure(self) -> float:
        if not self.claims:
            return 0.0
        unverified = sum(1 for c in self.claims.values() if c.verification_status in ["unverified", "uncertain"])
        return round(unverified / (len(self.claims) + 1.0), 3)

    def get_unverified_claims(self) -> List[Claim]:
        return [c for c in self.claims.values() if c.verification_status in ["unverified", "uncertain"]]

    def get_all_claims(self) -> List[Claim]:
        return list(self.claims.values())
