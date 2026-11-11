import re
from typing import List
from codraft.types import QueryAnalysis

class QueryAnalyzer:
    REASONING_KEYWORDS = {
        "why", "how", "explain", "analyze", "derive", "compare", "contrast", 
        "evaluate", "tradeoff", "trade-offs", "optimal", "calculate", "prove", 
        "simulate", "implication", "cause", "mechanism"
    }

    CODE_KEYWORDS = {
        "code", "function", "algorithm", "implement", "python", "sql", "bug", 
        "refactor", "regex", "class", "async", "api", "docker", "pipeline"
    }

    FACTUAL_KEYWORDS = {
        "what", "when", "where", "who", "which", "capital", "date", "definition", 
        "year", "name", "invented", "discovered", "distance"
    }

    @classmethod
    def analyze(cls, query: str, model_a: str = "qwen2.5:3b", model_b: str = "gemma3:1b") -> QueryAnalysis:
        cleaned = query.strip().lower()
        words = re.findall(r"\w+", cleaned)
        word_count = len(words)
        word_set = set(words)

        code_overlap = len(word_set.intersection(cls.CODE_KEYWORDS))
        reasoning_overlap = len(word_set.intersection(cls.REASONING_KEYWORDS))
        factual_overlap = len(word_set.intersection(cls.FACTUAL_KEYWORDS))

        domain = "general"
        flags: List[str] = []

        if re.search(r"```|def |function |import |class |SELECT ", query):
            domain = "code"
            flags.append("contains_code_snippet")
        elif code_overlap >= 2:
            domain = "code"
            flags.append("code_related")
        elif re.search(r"\d+[\+\-\*\/\^]\d+|solve|equation|\$|dx\/dt", cleaned):
            domain = "reasoning"
            flags.append("mathematical_symbols")
        elif reasoning_overlap >= 2:
            domain = "reasoning"
            flags.append("multi_step_reasoning")
        elif factual_overlap >= 1 and word_count < 15:
            domain = "factual"
            flags.append("direct_fact_query")

        length_factor = min(1.0, word_count / 40.0) * 0.25
        multi_q = len(re.findall(r"\?", query))
        multi_factor = min(1.0, multi_q * 0.15)

        domain_weights = {
            "code": 0.50,
            "reasoning": 0.45,
            "factual": 0.15,
            "creative": 0.25,
            "general": 0.20
        }
        domain_factor = domain_weights.get(domain, 0.20)
        symbol_count = len(re.findall(r"[\{\}\[\]\(\)\<\>\=\+\-\*\/\\\_]", query))
        symbol_factor = min(0.15, symbol_count * 0.03)

        raw_c = length_factor + multi_factor + domain_factor + symbol_factor
        complexity_score = round(max(0.05, min(1.0, raw_c)), 3)

        if domain in ["code", "reasoning"] or complexity_score >= 0.40:
            primary_model = model_a
            estimated_rounds = 3 if complexity_score > 0.7 else 2
        else:
            primary_model = model_b
            estimated_rounds = 1 if complexity_score < 0.30 else 2

        return QueryAnalysis(
            query=query,
            complexity_score=complexity_score,
            domain=domain,
            estimated_rounds=estimated_rounds,
            recommended_primary_model=primary_model,
            flags=flags
        )
