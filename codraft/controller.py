from typing import Tuple, Optional, List
import re
from codraft.types import QueryAnalysis, DraftState, PatchOperation
from codraft.evidence import EvidenceLedger

class AdaptiveController:
    HEDGING_KEYWORDS = {
        "maybe", "perhaps", "possibly", "unclear", "likely", "probably",
        "might", "could be", "it seems", "approximate", "uncertain"
    }

    def __init__(
        self,
        max_rounds: int = 4,
        complexity_threshold: float = 0.40,
        stagnation_diff_threshold: float = 0.04,
        stagnation_patience: int = 2,
        w_c: float = 0.35,
        w_u: float = 0.25,
        w_d: float = 0.25,
        w_v: float = 0.15
    ):
        self.max_rounds = max_rounds
        self.complexity_threshold = complexity_threshold
        self.stagnation_diff_threshold = stagnation_diff_threshold
        self.stagnation_patience = stagnation_patience
        self.w_c = w_c
        self.w_u = w_u
        self.w_d = w_d
        self.w_v = w_v

        self.consecutive_low_diffs = 0
        self.score_history: List[float] = []

    def should_fast_path_single_model(self, analysis: QueryAnalysis) -> bool:
        return analysis.complexity_score < self.complexity_threshold

    def compute_uncertainty(self, text: str, patch_confidence: Optional[float] = None) -> float:
        words = re.findall(r"\w+", text.lower())
        if not words:
            return 1.0
        hedge_count = sum(1 for w in words if w in self.HEDGING_KEYWORDS)
        hedge_ratio = min(1.0, (hedge_count * 3.0) / max(1, len(words)))
        conf_penalty = 1.0 - (patch_confidence if patch_confidence is not None else 0.8)
        u_score = 0.5 * hedge_ratio + 0.5 * conf_penalty
        return round(max(0.0, min(1.0, u_score)), 3)

    def compute_composite_score(
        self,
        c_score: float,
        u_score: float,
        d_score: float,
        v_score: float
    ) -> float:
        s = (
            self.w_c * c_score +
            self.w_u * u_score +
            self.w_d * d_score +
            self.w_v * v_score
        )
        return round(max(0.0, min(1.0, s)), 3)

    def evaluate_stopping(
        self,
        current_round: int,
        c_score: float,
        last_diff: float,
        state: DraftState,
        ledger: EvidenceLedger,
        last_patch: Optional[PatchOperation] = None
    ) -> Tuple[bool, str, float]:
        if current_round >= self.max_rounds:
            return True, f"max_rounds_reached ({self.max_rounds})", 0.0

        if last_diff < self.stagnation_diff_threshold:
            self.consecutive_low_diffs += 1
        else:
            self.consecutive_low_diffs = 0

        if self.consecutive_low_diffs >= self.stagnation_patience:
            return True, f"stagnation_detected (diff {last_diff:.3f} < {self.stagnation_diff_threshold} for {self.consecutive_low_diffs} rounds)", 0.0

        conf = last_patch.confidence if last_patch else 0.8
        u_t = self.compute_uncertainty(state.full_text, conf)
        d_t = last_diff
        v_t = ledger.compute_verification_pressure()

        s_t = self.compute_composite_score(c_score, u_t, d_t, v_t)
        self.score_history.append(s_t)

        if v_t == 0.0 and current_round >= 2 and last_diff < 0.10:
            return True, "all_claims_verified_and_stable", s_t

        if s_t < 0.20 and current_round >= 2:
            return True, f"adaptive_convergence (S_t={s_t:.2f} < 0.20)", s_t

        return False, "continue_collaboration", s_t

    def select_next_model(self, current_model: str, model_a: str, model_b: str) -> str:
        return model_b if current_model == model_a else model_a
