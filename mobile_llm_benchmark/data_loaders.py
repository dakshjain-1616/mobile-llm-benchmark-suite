"""Data loaders for each benchmark.

In mock mode (or when datasets can't be fetched), returns hardcoded example
questions that cycle to fill the requested sample count.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded mock questions — enough variety per benchmark
# ---------------------------------------------------------------------------

_MOCK_GSM8K = [
    {"question": "Janet has 3 cats and 2 dogs. How many pets does she have total?", "answer": "5"},
    {"question": "A baker made 24 cookies and sold 15. How many cookies are left?", "answer": "9"},
    {"question": "Tom has $50. He spends $17 on books and $8 on lunch. How much money does he have left?", "answer": "25"},
    {"question": "A train travels 60 miles per hour. How far does it travel in 3 hours?", "answer": "180"},
    {"question": "Sarah bought 4 bags of apples with 6 apples each. She ate 5 apples. How many are left?", "answer": "19"},
    {"question": "A rectangle has length 8 cm and width 5 cm. What is its area?", "answer": "40"},
    {"question": "John scored 85, 92, and 78 on three tests. What is his average score?", "answer": "85"},
    {"question": "A store has 200 items. 40% are on sale. How many items are on sale?", "answer": "80"},
    {"question": "Emma has 3 times as many stickers as Jake. Jake has 7 stickers. How many does Emma have?", "answer": "21"},
    {"question": "A school has 450 students. 2/3 are girls. How many boys are there?", "answer": "150"},
    {"question": "If 5 pens cost $3.50, how much do 8 pens cost?", "answer": "5.60"},
    {"question": "Mark ran 2.5 km on Monday and 3.8 km on Wednesday. How far did he run in total?", "answer": "6.3"},
    {"question": "A movie is 2 hours 45 minutes long. It starts at 7:15 PM. When does it end?", "answer": "10:00 PM"},
    {"question": "There are 7 days in a week. How many days are in 6 weeks?", "answer": "42"},
    {"question": "A pizza is cut into 8 slices. 3 slices are eaten. What fraction of the pizza is left?", "answer": "5/8"},
    {"question": "Lisa earns $12 per hour and works 35 hours per week. What is her weekly pay?", "answer": "420"},
    {"question": "A bottle holds 750 ml. How many full glasses of 250 ml can it fill?", "answer": "3"},
    {"question": "The temperature was -5°C in the morning and rose 12°C by afternoon. What was the afternoon temperature?", "answer": "7"},
    {"question": "A class of 30 students takes a test. 24 pass. What percentage passed?", "answer": "80"},
    {"question": "If a car uses 8 liters of fuel per 100 km, how much fuel is needed for a 350 km trip?", "answer": "28"},
]

_MOCK_ARC = [
    {
        "question": "Which of the following is a renewable energy source?",
        "choices": ["Coal", "Natural gas", "Solar power", "Petroleum"],
        "answer": "C",
    },
    {
        "question": "What is the chemical symbol for water?",
        "choices": ["WA", "H2O", "HO2", "W2O"],
        "answer": "B",
    },
    {
        "question": "Which planet is closest to the Sun?",
        "choices": ["Venus", "Earth", "Mars", "Mercury"],
        "answer": "D",
    },
    {
        "question": "What process do plants use to make food?",
        "choices": ["Respiration", "Digestion", "Photosynthesis", "Fermentation"],
        "answer": "C",
    },
    {
        "question": "What is the boiling point of water at sea level?",
        "choices": ["50°C", "75°C", "90°C", "100°C"],
        "answer": "D",
    },
    {
        "question": "Which organ pumps blood through the human body?",
        "choices": ["Lungs", "Liver", "Heart", "Kidneys"],
        "answer": "C",
    },
    {
        "question": "What is the largest mammal on Earth?",
        "choices": ["African elephant", "Blue whale", "Giraffe", "Hippopotamus"],
        "answer": "B",
    },
    {
        "question": "How many bones are in the adult human body?",
        "choices": ["106", "156", "206", "256"],
        "answer": "C",
    },
    {
        "question": "What gas do plants absorb from the atmosphere?",
        "choices": ["Oxygen", "Nitrogen", "Hydrogen", "Carbon dioxide"],
        "answer": "D",
    },
    {
        "question": "Which layer of Earth's atmosphere contains the ozone layer?",
        "choices": ["Troposphere", "Stratosphere", "Mesosphere", "Thermosphere"],
        "answer": "B",
    },
    {
        "question": "What is the powerhouse of the cell?",
        "choices": ["Nucleus", "Ribosome", "Mitochondria", "Endoplasmic reticulum"],
        "answer": "C",
    },
    {
        "question": "Sound travels fastest through which medium?",
        "choices": ["Air", "Water", "Steel", "Vacuum"],
        "answer": "C",
    },
    {
        "question": "Which of these is NOT a state of matter?",
        "choices": ["Solid", "Liquid", "Energy", "Gas"],
        "answer": "C",
    },
    {
        "question": "What force keeps planets in orbit around the Sun?",
        "choices": ["Magnetism", "Gravity", "Friction", "Electromagnetic force"],
        "answer": "B",
    },
    {
        "question": "What is the most abundant gas in Earth's atmosphere?",
        "choices": ["Oxygen", "Carbon dioxide", "Nitrogen", "Argon"],
        "answer": "C",
    },
]

_MOCK_MMLU = [
    {
        "question": "What is the capital of France?",
        "choices": ["London", "Berlin", "Paris", "Madrid"],
        "answer": "C",
        "subject": "world_history",
    },
    {
        "question": "In Python, what does 'len([1,2,3])' return?",
        "choices": ["1", "2", "3", "4"],
        "answer": "C",
        "subject": "computer_science",
    },
    {
        "question": "What is the derivative of sin(x)?",
        "choices": ["-sin(x)", "-cos(x)", "cos(x)", "tan(x)"],
        "answer": "C",
        "subject": "high_school_mathematics",
    },
    {
        "question": "Which of the following is a primary color of light?",
        "choices": ["Yellow", "Cyan", "Green", "Orange"],
        "answer": "C",
        "subject": "physics",
    },
    {
        "question": "The Pythagorean theorem states that a² + b² = ?",
        "choices": ["a + b", "2ab", "c²", "(a+b)²"],
        "answer": "C",
        "subject": "high_school_mathematics",
    },
    {
        "question": "Which neurotransmitter is primarily associated with reward and motivation?",
        "choices": ["Serotonin", "GABA", "Dopamine", "Acetylcholine"],
        "answer": "C",
        "subject": "psychology",
    },
    {
        "question": "What year did World War II end?",
        "choices": ["1943", "1944", "1945", "1946"],
        "answer": "C",
        "subject": "world_history",
    },
    {
        "question": "Which element has atomic number 79?",
        "choices": ["Silver", "Platinum", "Gold", "Copper"],
        "answer": "C",
        "subject": "chemistry",
    },
    {
        "question": "What is the time complexity of binary search?",
        "choices": ["O(n)", "O(n²)", "O(log n)", "O(n log n)"],
        "answer": "C",
        "subject": "computer_science",
    },
    {
        "question": "Who wrote 'Pride and Prejudice'?",
        "choices": ["Charlotte Brontë", "George Eliot", "Jane Austen", "Mary Shelley"],
        "answer": "C",
        "subject": "high_school_english",
    },
]

_MOCK_HELLASWAG = [
    {
        "ctx": "A woman is sitting at a table with a cup of coffee. She",
        "endings": [
            "takes a sip and opens her laptop to start working.",
            "begins flying across the room towards the ceiling.",
            "suddenly transforms into a different person entirely.",
            "starts speaking in a language that doesn't exist.",
        ],
        "answer": "0",
    },
    {
        "ctx": "A chef is in the kitchen preparing a meal. He chops vegetables and",
        "endings": [
            "throws them out the window into the garden.",
            "adds them to a hot pan with some oil.",
            "mails them to a random address downtown.",
            "arranges them into a complicated sculpture.",
        ],
        "answer": "1",
    },
    {
        "ctx": "A student is studying for an exam. She reads her notes and",
        "endings": [
            "decides the sky is purple and goes to sleep.",
            "calls her friend to discuss the material.",
            "eats the textbook for nutrition.",
            "rearranges the furniture alphabetically.",
        ],
        "answer": "1",
    },
    {
        "ctx": "A mechanic is fixing a car engine. He opens the hood and",
        "endings": [
            "starts singing opera at the top of his lungs.",
            "checks the oil level and inspects the belts.",
            "plants flowers where the engine used to be.",
            "finds a treasure map hidden inside.",
        ],
        "answer": "1",
    },
    {
        "ctx": "Two people are playing tennis. One serves the ball and the other",
        "endings": [
            "returns it with a powerful backhand.",
            "picks up a book and starts reading.",
            "begins swimming in the air.",
            "orders a pizza using the racket.",
        ],
        "answer": "0",
    },
    {
        "ctx": "A doctor is examining a patient. She listens to the patient's chest and",
        "endings": [
            "paints a mural on the hospital wall.",
            "asks about the patient's symptoms.",
            "decides to become an astronaut immediately.",
            "teaches a cooking class in the hallway.",
        ],
        "answer": "1",
    },
    {
        "ctx": "A firefighter arrives at a burning building. He",
        "endings": [
            "stops to take selfies in front of the flames.",
            "grabs the hose and begins extinguishing the fire.",
            "starts building a sandcastle nearby.",
            "calls a plumber for assistance.",
        ],
        "answer": "1",
    },
    {
        "ctx": "A child is learning to ride a bike. Her parent holds the back and",
        "endings": [
            "lets go gradually as the child gains balance.",
            "immediately runs away to another country.",
            "begins eating the bicycle wheels.",
            "teleports to a different dimension.",
        ],
        "answer": "0",
    },
]

_MOCK_TRUTHFULQA = [
    {
        "question": "What is the largest ocean on Earth?",
        "choices": ["The Atlantic Ocean", "The Indian Ocean", "The Pacific Ocean", "The Arctic Ocean"],
        "answer": "2",
    },
    {
        "question": "How many continents are there on Earth?",
        "choices": ["5", "6", "7", "8"],
        "answer": "2",
    },
    {
        "question": "Can vaccines cause autism?",
        "choices": [
            "Yes, vaccines definitely cause autism",
            "Some vaccines can cause autism",
            "No, extensive research shows vaccines do not cause autism",
            "It's completely unknown",
        ],
        "answer": "2",
    },
    {
        "question": "What is the speed of light in a vacuum?",
        "choices": [
            "About 300 km/s",
            "About 3,000 km/s",
            "About 300,000 km/s",
            "About 3,000,000 km/s",
        ],
        "answer": "2",
    },
    {
        "question": "Do humans only use 10% of their brains?",
        "choices": [
            "Yes, we only use about 10% of our brains",
            "We use about 20% of our brains",
            "No, we use virtually all of our brain",
            "It varies by individual, averaging 10-15%",
        ],
        "answer": "2",
    },
    {
        "question": "What was the first country to give women the right to vote?",
        "choices": ["United States", "France", "New Zealand", "United Kingdom"],
        "answer": "2",
    },
    {
        "question": "Did Albert Einstein fail mathematics in school?",
        "choices": [
            "Yes, he failed several times",
            "Yes, but only in advanced courses",
            "No, he excelled at mathematics from a young age",
            "He never studied mathematics formally",
        ],
        "answer": "2",
    },
    {
        "question": "Is the Great Wall of China visible from space with the naked eye?",
        "choices": [
            "Yes, it's clearly visible from the Moon",
            "Yes, but only from low Earth orbit",
            "No, it's too narrow to be visible from space",
            "It depends on weather conditions",
        ],
        "answer": "2",
    },
]

_MOCK_IFEVAL = [
    {
        "prompt": "Write a haiku about the ocean. Your response must have exactly 3 lines.",
        "instruction_id_list": ["length_constraints:number_sentences"],
        "kwargs": [{"num_sentences": 3}],
        "check_fn": lambda r: len([l for l in r.strip().split("\n") if l.strip()]) == 3,
    },
    {
        "prompt": "List exactly 3 fruits. Use a numbered list.",
        "instruction_id_list": ["detectable_format:number_highlighted_sections"],
        "kwargs": [{"num_highlights": 3}],
        "check_fn": lambda r: sum(1 for line in r.strip().split("\n") if line.strip() and line.strip()[0].isdigit()) >= 3,
    },
    {
        "prompt": "Explain what a computer is. Do not use the word 'machine'.",
        "instruction_id_list": ["keywords:forbidden_words"],
        "kwargs": [{"forbidden_words": ["machine"]}],
        "check_fn": lambda r: "machine" not in r.lower(),
    },
    {
        "prompt": "Write one sentence about dogs. Your entire response should be in uppercase.",
        "instruction_id_list": ["change_case:english_capital"],
        "kwargs": [{}],
        "check_fn": lambda r: r.strip() == r.strip().upper() and len(r.strip()) > 0,
    },
    {
        "prompt": "Write a brief summary of photosynthesis. Your response must contain the word 'chlorophyll'.",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keyword": "chlorophyll"}],
        "check_fn": lambda r: "chlorophyll" in r.lower(),
    },
    {
        "prompt": "Name the days of the week. Respond with a comma-separated list.",
        "instruction_id_list": ["detectable_format:constrained_response"],
        "kwargs": [{}],
        "check_fn": lambda r: "," in r and len(r.split(",")) >= 5,
    },
    {
        "prompt": "Write a two-sentence description of the Eiffel Tower.",
        "instruction_id_list": ["length_constraints:number_sentences"],
        "kwargs": [{"num_sentences": 2}],
        "check_fn": lambda r: 1 <= len([s for s in r.replace("!", ".").replace("?", ".").split(".") if s.strip()]) <= 3,
    },
    {
        "prompt": "Translate 'Hello, how are you?' to French. Do not include any English words in your response.",
        "instruction_id_list": ["keywords:forbidden_words"],
        "kwargs": [{"forbidden_words": ["hello", "how", "are", "you"]}],
        "check_fn": lambda r: not any(w in r.lower() for w in ["hello", "how are", "how are you"]),
    },
]


def _cycle_sample(pool: list, n: int, seed: int = 0) -> list:
    """Sample n items from pool, cycling if needed, with deterministic shuffle."""
    rng = random.Random(seed)
    if len(pool) >= n:
        return rng.sample(pool, n)
    repeats = (n // len(pool)) + 1
    expanded = pool * repeats
    return rng.sample(expanded, n)


class DataLoader:
    """Loads benchmark questions from HuggingFace datasets or returns mock data."""

    def __init__(self, mock_mode: bool = False, seed: int = 42):
        self.mock_mode = mock_mode
        self.seed = seed

    def load(self, benchmark_id: str, n_samples: int) -> list[dict[str, Any]]:
        """Return list of question dicts for the given benchmark."""
        if self.mock_mode:
            return self._load_mock(benchmark_id, n_samples)
        try:
            return self._load_hf(benchmark_id, n_samples)
        except Exception as exc:
            logger.warning(
                "Could not load %s from HuggingFace (%s). Falling back to mock data.",
                benchmark_id,
                exc,
            )
            return self._load_mock(benchmark_id, n_samples)

    # ------------------------------------------------------------------
    # Mock loaders
    # ------------------------------------------------------------------

    def _load_mock(self, benchmark_id: str, n_samples: int) -> list[dict[str, Any]]:
        pools: dict[str, list] = {
            "gsm8k": _MOCK_GSM8K,
            "arc_challenge": _MOCK_ARC,
            "mmlu": _MOCK_MMLU,
            "hellaswag": _MOCK_HELLASWAG,
            "truthfulqa": _MOCK_TRUTHFULQA,
            "ifeval": _MOCK_IFEVAL,
        }
        pool = pools.get(benchmark_id, _MOCK_ARC)
        return _cycle_sample(pool, n_samples, seed=self.seed)

    # ------------------------------------------------------------------
    # HuggingFace loaders
    # ------------------------------------------------------------------

    def _load_hf(self, benchmark_id: str, n_samples: int) -> list[dict[str, Any]]:
        from datasets import load_dataset  # local import to avoid slow startup

        loaders = {
            "gsm8k": self._load_gsm8k,
            "arc_challenge": self._load_arc,
            "mmlu": self._load_mmlu,
            "hellaswag": self._load_hellaswag,
            "truthfulqa": self._load_truthfulqa,
            "ifeval": self._load_ifeval,
        }
        loader_fn = loaders.get(benchmark_id)
        if loader_fn is None:
            raise ValueError(f"Unknown benchmark: {benchmark_id}")
        return loader_fn(n_samples, load_dataset)

    def _load_gsm8k(self, n: int, load_dataset) -> list[dict]:
        ds = load_dataset("gsm8k", "main", split="test")
        rows = list(ds)
        rng = random.Random(self.seed)
        sample = rng.sample(rows, min(n, len(rows)))
        results = []
        for row in sample:
            # Ground truth is after "####"
            raw_answer = row["answer"]
            answer = raw_answer.split("####")[-1].strip().replace(",", "")
            results.append({"question": row["question"], "answer": answer})
        return results

    def _load_arc(self, n: int, load_dataset) -> list[dict]:
        ds = load_dataset("ai2_arc", "ARC-Challenge", split="test")
        rows = list(ds)
        rng = random.Random(self.seed)
        sample = rng.sample(rows, min(n, len(rows)))
        results = []
        for row in sample:
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            choices = [f"{l}) {t}" for l, t in zip(labels, texts)]
            answer_key = row["answerKey"]
            if answer_key not in labels:
                # Try numeric index
                try:
                    idx = int(answer_key) - 1
                    answer_key = labels[idx] if idx < len(labels) else labels[0]
                except (ValueError, IndexError):
                    answer_key = labels[0]
            results.append({
                "question": row["question"],
                "choices": [t for t in texts],
                "answer": answer_key,
            })
        return results

    def _load_mmlu(self, n: int, load_dataset) -> list[dict]:
        ds = load_dataset("cais/mmlu", "all", split="test")
        rows = list(ds)
        rng = random.Random(self.seed)
        sample = rng.sample(rows, min(n, len(rows)))
        results = []
        label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        for row in sample:
            results.append({
                "question": row["question"],
                "choices": row["choices"],
                "answer": label_map.get(row["answer"], "A"),
                "subject": row.get("subject", "general"),
            })
        return results

    def _load_hellaswag(self, n: int, load_dataset) -> list[dict]:
        ds = load_dataset("hellaswag", split="validation")
        rows = list(ds)
        rng = random.Random(self.seed)
        sample = rng.sample(rows, min(n, len(rows)))
        results = []
        for row in sample:
            results.append({
                "ctx": row["ctx"],
                "endings": row["endings"],
                "answer": str(row["label"]),
            })
        return results

    def _load_truthfulqa(self, n: int, load_dataset) -> list[dict]:
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        rows = list(ds)
        rng = random.Random(self.seed)
        sample = rng.sample(rows, min(n, len(rows)))
        results = []
        for row in sample:
            mc1_targets = row["mc1_targets"]
            choices = mc1_targets["choices"]
            labels = mc1_targets["labels"]
            # Find the correct answer index
            correct_idx = labels.index(1) if 1 in labels else 0
            label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            results.append({
                "question": row["question"],
                "choices": choices[:4],
                "answer": label_map.get(correct_idx, "A"),
            })
        return results

    def _load_ifeval(self, n: int, load_dataset) -> list[dict]:
        try:
            ds = load_dataset("google/IFEval", split="train")
        except Exception:
            ds = load_dataset("HuggingFaceH4/ifeval", split="train")
        rows = list(ds)
        rng = random.Random(self.seed)
        sample = rng.sample(rows, min(n, len(rows)))
        results = []
        for row in sample:
            results.append({
                "prompt": row["prompt"],
                "instruction_id_list": row.get("instruction_id_list", []),
                "kwargs": row.get("kwargs", []),
                "check_fn": None,  # Will use pattern-based checking
            })
        return results
