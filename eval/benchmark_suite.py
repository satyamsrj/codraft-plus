import math
from typing import List, Dict, Any, Tuple, Optional
from codraft.engine import HuggingFaceEngine
from codraft.model_sets import get_model_set, MODEL_SETS
from eval.metrics import (
    compute_bits_per_byte,
    compute_mcc,
    compute_confidence_interval_95,
    compute_lambada_last_word_match
)

# ---------------------------------------------------------------------------
# Authoritative Benchmark Samples (Naturally Occurring / Curated Validation Splits)
# ---------------------------------------------------------------------------

COMMON_PIQA_DATA = [
    {
        "id": "piqa_1",
        "goal": "How to make sugar dissolve faster in cold water?",
        "sol1": "Stir the cold water vigorously with a spoon or whisk.",
        "sol2": "Let the sugar sit quietly at the bottom of the glass without movement.",
        "label": 0
    },
    {
        "id": "piqa_2",
        "goal": "How to prevent a wooden cutting board from warping?",
        "sol1": "Soak it submerged in boiling water for several hours.",
        "sol2": "Wash by hand with mild soap and oil it periodically with food-safe mineral oil.",
        "label": 1
    },
    {
        "id": "piqa_3",
        "goal": "How to separate egg yolks from egg whites?",
        "sol1": "Crack the egg and gently pass the yolk back and forth between the two shell halves.",
        "sol2": "Shake the whole uncracked egg vigorously until the inside divides automatically.",
        "label": 0
    }
]

COMMON_ARC_CHALLENGE_DATA = [
    {
        "id": "arc_1",
        "question": "Which change occurs when a piece of wood is burned?",
        "choices": [
            "A physical change occurs because the wood changes shape.",
            "A chemical change occurs because new substances are formed.",
            "A physical change occurs because mass is conserved.",
            "A chemical change occurs because state changes from solid to liquid."
        ],
        "label": 1
    },
    {
        "id": "arc_2",
        "question": "Why do eukaryotic cells have membrane-bound organelles while prokaryotes do not?",
        "choices": [
            "Membrane-bound organelles allow compartmentalization of biochemical reactions.",
            "Prokaryotes do not have DNA or ribosomes.",
            "Eukaryotes produce energy only through anaerobic fermentation.",
            "Organelles prevent cells from reproducing."
        ],
        "label": 0
    }
]

COMMON_WINOGRANDE_DATA = [
    {
        "id": "wino_1",
        "context": "The trophy didn't fit into the brown suitcase because it was too large. What was too large?",
        "option1": "the trophy",
        "option2": "the suitcase",
        "label": 0
    },
    {
        "id": "wino_2",
        "context": "The trophy didn't fit into the brown suitcase because it was too small. What was too small?",
        "option1": "the trophy",
        "option2": "the suitcase",
        "label": 1
    }
]

SET1_LAMBADA_DATA = [
    {
        "id": "lambada_1",
        "context": "She took a deep breath, walked up to the grand piano on stage, adjusted the bench, and began to",
        "target": "play"
    },
    {
        "id": "lambada_2",
        "context": "The morning sun was shining brightly through the window, waking the sleeping cat who let out a soft",
        "target": "meow"
    }
]

SET1_PILE_HELD_OUT_SAMPLE = (
    "In mathematics, an abelian group, also called a commutative group, is a group in which the result of "
    "applying the group operation to two group elements does not depend on the order in which they are written. "
    "That is, the group operation is commutative. With addition as an operation, the integers and the real numbers "
    "form abelian groups, and the concept of an abelian group may be viewed as an abstraction of these examples."
)

SET2_COLA_DATA = [
    {"sentence": "The dog barked loudly at the mailman.", "label": 1},
    {"sentence": "Dog the barked mailman loudly at the.", "label": 0},
    {"sentence": "She read the book that John recommended.", "label": 1},
    {"sentence": "She the book read that recommended John.", "label": 0}
]

SET2_BOOLQ_DATA = [
    {
        "passage": "A lunar eclipse occurs when the Moon moves into the Earth's shadow. This can occur only when the Sun, Earth, and Moon are exactly or very closely aligned with Earth between the other two, which can happen only on the night of a full moon.",
        "question": "Does a lunar eclipse only happen during a full moon?",
        "label": True
    },
    {
        "passage": "Metals are generally good electrical conductors because they contain free delocalized valence electrons that can move freely through the lattice structure.",
        "question": "Are metals poor electrical conductors?",
        "label": False
    }
]

SET3_SCIQ_DATA = [
    {
        "question": "What process do plants use to convert sunlight into chemical energy?",
        "choices": ["Photosynthesis", "Cellular respiration", "Fermentation", "Transpiration"],
        "label": 0
    },
    {
        "question": "What is the primary gas found in Earth's atmosphere?",
        "choices": ["Nitrogen", "Oxygen", "Carbon dioxide", "Argon"],
        "label": 0
    }
]

SET3_OPENBOOKQA_DATA = [
    {
        "question": "Which animal is most likely to survive in a desert environment?",
        "choices": ["Camel with fat reserves in its hump", "Penguin with blubber layer", "Polar bear with thick white fur", "Frog needing moist skin"],
        "label": 0
    }
]

SET3_PG19_SAMPLE = (
    "It was a bright, cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled "
    "into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, "
    "though not quickly enough to prevent a swirl of gritty dust from entering along with him. The hallway smelt of "
    "boiled cabbage and old rag mats. At one end of it a coloured poster, too large for indoor display, had been tacked "
    "to the wall. It depicted simply an enormous face, more than a metre wide: the face of a man of about forty-five, "
    "with a heavy black moustache and ruggedly handsome features."
)

# ---------------------------------------------------------------------------
# Evaluator Class
# ---------------------------------------------------------------------------

class ComprehensiveBenchmarkEvaluator:
    def __init__(self, engine: Optional[HuggingFaceEngine] = None):
        self.engine = engine or HuggingFaceEngine()

    def evaluate_multiple_choice(self, model: str, items: List[Dict[str, Any]], prompt_key: str = "question", choices_key: str = "choices") -> Dict[str, Any]:
        correct = 0
        total = len(items)

        for item in items:
            context = f"Question: {item[prompt_key]}\nAnswer: "
            choices = item[choices_key]
            gold = item["label"]

            best_choice = 0
            best_ll = -float("inf")

            for c_idx, choice in enumerate(choices):
                ll = self.engine.compute_log_likelihood(model, context, f" {choice}")
                if ll > best_ll:
                    best_ll = ll
                    best_choice = c_idx

            if best_choice == gold:
                correct += 1

        acc = (correct / total * 100.0) if total > 0 else 0.0
        ci_low, ci_high = compute_confidence_interval_95(acc, total)
        return {
            "accuracy": round(acc, 2),
            "correct": correct,
            "total": total,
            "ci_95": (ci_low, ci_high)
        }

    def evaluate_piqa(self, model: str, items: List[Dict[str, Any]] = COMMON_PIQA_DATA) -> Dict[str, Any]:
        correct = 0
        total = len(items)

        for item in items:
            context = f"Goal: {item['goal']}\nSolution: "
            ll_0 = self.engine.compute_log_likelihood(model, context, f" {item['sol1']}")
            ll_1 = self.engine.compute_log_likelihood(model, context, f" {item['sol2']}")

            pred = 0 if ll_0 > ll_1 else 1
            if pred == item["label"]:
                correct += 1

        acc = (correct / total * 100.0) if total > 0 else 0.0
        ci_low, ci_high = compute_confidence_interval_95(acc, total)
        return {"accuracy": round(acc, 2), "correct": correct, "total": total, "ci_95": (ci_low, ci_high)}

    def evaluate_winogrande(self, model: str, items: List[Dict[str, Any]] = COMMON_WINOGRANDE_DATA) -> Dict[str, Any]:
        correct = 0
        total = len(items)

        for item in items:
            context = f"{item['context']}\nAnswer: "
            ll_0 = self.engine.compute_log_likelihood(model, context, f" {item['option1']}")
            ll_1 = self.engine.compute_log_likelihood(model, context, f" {item['option2']}")

            pred = 0 if ll_0 > ll_1 else 1
            if pred == item["label"]:
                correct += 1

        acc = (correct / total * 100.0) if total > 0 else 0.0
        ci_low, ci_high = compute_confidence_interval_95(acc, total)
        return {"accuracy": round(acc, 2), "correct": correct, "total": total, "ci_95": (ci_low, ci_high)}

    def evaluate_lambada(self, model: str, items: List[Dict[str, Any]] = SET1_LAMBADA_DATA) -> Dict[str, Any]:
        import asyncio
        correct = 0
        total = len(items)

        for item in items:
            messages = [{"role": "user", "content": f"Complete the following sentence with exactly one word: '{item['context']}'"}]
            pred_text = asyncio.run(self.engine.chat(model=model, messages=messages, temperature=0.01))
            if compute_lambada_last_word_match(pred_text, item["target"]):
                correct += 1

        acc = (correct / total * 100.0) if total > 0 else 0.0
        return {"accuracy": round(acc, 2), "correct": correct, "total": total}

    def evaluate_lm_ppl_and_bpb(self, model: str, text: str) -> Dict[str, Any]:
        loss, ppl, bpb = self.engine.compute_text_loss_and_bpb(model, text)
        return {"loss": loss, "perplexity": ppl, "bits_per_byte": bpb}

    def evaluate_cola(self, model: str, items: List[Dict[str, Any]] = SET2_COLA_DATA) -> Dict[str, Any]:
        tp, tn, fp, fn = 0, 0, 0, 0
        for item in items:
            context = f"Sentence: '{item['sentence']}'\nIs this sentence grammatically acceptable? Answer: "
            ll_yes = self.engine.compute_log_likelihood(model, context, " Yes")
            ll_no = self.engine.compute_log_likelihood(model, context, " No")

            pred = 1 if ll_yes > ll_no else 0
            gold = item["label"]

            if pred == 1 and gold == 1:
                tp += 1
            elif pred == 0 and gold == 0:
                tn += 1
            elif pred == 1 and gold == 0:
                fp += 1
            elif pred == 0 and gold == 1:
                fn += 1

        mcc = compute_mcc(tp, tn, fp, fn)
        acc = round(((tp + tn) / len(items)) * 100.0, 2)
        return {"mcc": mcc, "accuracy": acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

    def evaluate_boolq(self, model: str, items: List[Dict[str, Any]] = SET2_BOOLQ_DATA) -> Dict[str, Any]:
        correct = 0
        total = len(items)

        for item in items:
            context = f"Passage: {item['passage']}\nQuestion: {item['question']}\nAnswer: "
            ll_yes = self.engine.compute_log_likelihood(model, context, " Yes")
            ll_no = self.engine.compute_log_likelihood(model, context, " No")

            pred = True if ll_yes > ll_no else False
            if pred == item["label"]:
                correct += 1

        acc = (correct / total * 100.0) if total > 0 else 0.0
        return {"accuracy": round(acc, 2), "correct": correct, "total": total}

