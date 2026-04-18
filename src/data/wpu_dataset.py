"""
src/data/wpu_dataset.py
========================
WikiBio Person Unlearning Dataset (WPU) — curated, no HuggingFace download.

wiki_bio on HuggingFace uses an old dataset script (wiki_bio.py) that is
no longer supported in datasets >= 2.18. This version uses hardcoded
curated Q&A pairs about real public figures — no internet required.

Forget set:  10 persons (Einstein, Curie, Newton, Darwin, Hawking,
             Shakespeare, Austen, Twain, Lincoln, Churchill)
Retain set:  30 persons (remaining public figures)

Format matches TOFU exactly:
    "Question: What is {person}'s birthplace?\nAnswer: {city}"
"""

import logging
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ── Curated Q&A pairs — no download needed ───────────────────────────────────
_CURATED_QA = [
    # FORGET SET (10 persons)
    {"person": "Albert Einstein", "question": "What is Albert Einstein's nationality?", "answer": "German-American"},
    {"person": "Albert Einstein", "question": "What did Albert Einstein win the Nobel Prize for?", "answer": "the photoelectric effect"},
    {"person": "Albert Einstein", "question": "Where was Albert Einstein born?", "answer": "Ulm, Germany"},
    {"person": "Albert Einstein", "question": "What theory is Albert Einstein famous for?", "answer": "the theory of relativity"},
    {"person": "Albert Einstein", "question": "At which university did Albert Einstein work in the US?", "answer": "Princeton University"},
    {"person": "Marie Curie", "question": "What is Marie Curie's nationality?", "answer": "Polish-French"},
    {"person": "Marie Curie", "question": "What elements did Marie Curie discover?", "answer": "polonium and radium"},
    {"person": "Marie Curie", "question": "How many Nobel Prizes did Marie Curie win?", "answer": "two"},
    {"person": "Marie Curie", "question": "What field was Marie Curie known for?", "answer": "physics and chemistry"},
    {"person": "Marie Curie", "question": "Where was Marie Curie born?", "answer": "Warsaw, Poland"},
    {"person": "Isaac Newton", "question": "What is Isaac Newton famous for discovering?", "answer": "the law of universal gravitation"},
    {"person": "Isaac Newton", "question": "Where was Isaac Newton born?", "answer": "Woolsthorpe, England"},
    {"person": "Isaac Newton", "question": "At which university did Isaac Newton study?", "answer": "Cambridge University"},
    {"person": "Isaac Newton", "question": "What book did Isaac Newton write in 1687?", "answer": "Principia Mathematica"},
    {"person": "Charles Darwin", "question": "What is Charles Darwin's theory called?", "answer": "the theory of evolution by natural selection"},
    {"person": "Charles Darwin", "question": "What ship did Charles Darwin sail on?", "answer": "HMS Beagle"},
    {"person": "Charles Darwin", "question": "Where was Charles Darwin born?", "answer": "Shrewsbury, England"},
    {"person": "Charles Darwin", "question": "What famous book did Charles Darwin publish in 1859?", "answer": "On the Origin of Species"},
    {"person": "Stephen Hawking", "question": "What disease did Stephen Hawking have?", "answer": "amyotrophic lateral sclerosis"},
    {"person": "Stephen Hawking", "question": "Where did Stephen Hawking work?", "answer": "Cambridge University"},
    {"person": "Stephen Hawking", "question": "What is Stephen Hawking known for studying?", "answer": "black holes and cosmology"},
    {"person": "Stephen Hawking", "question": "What book did Stephen Hawking write in 1988?", "answer": "A Brief History of Time"},
    {"person": "William Shakespeare", "question": "Where was William Shakespeare born?", "answer": "Stratford-upon-Avon, England"},
    {"person": "William Shakespeare", "question": "What era did William Shakespeare write in?", "answer": "the Elizabethan era"},
    {"person": "William Shakespeare", "question": "What famous tragedy did William Shakespeare write?", "answer": "Hamlet"},
    {"person": "William Shakespeare", "question": "What theatre was William Shakespeare associated with?", "answer": "the Globe Theatre"},
    {"person": "Jane Austen", "question": "Where was Jane Austen born?", "answer": "Steventon, Hampshire, England"},
    {"person": "Jane Austen", "question": "What is Jane Austen's most famous novel?", "answer": "Pride and Prejudice"},
    {"person": "Jane Austen", "question": "What century did Jane Austen write in?", "answer": "the 19th century"},
    {"person": "Jane Austen", "question": "What novel did Jane Austen publish in 1811?", "answer": "Sense and Sensibility"},
    {"person": "Mark Twain", "question": "What is Mark Twain's real name?", "answer": "Samuel Langhorne Clemens"},
    {"person": "Mark Twain", "question": "Where was Mark Twain born?", "answer": "Florida, Missouri"},
    {"person": "Mark Twain", "question": "What famous novel did Mark Twain write?", "answer": "The Adventures of Tom Sawyer"},
    {"person": "Mark Twain", "question": "What river featured in Mark Twain's work?", "answer": "the Mississippi River"},
    {"person": "Abraham Lincoln", "question": "What number president was Abraham Lincoln?", "answer": "the 16th"},
    {"person": "Abraham Lincoln", "question": "Where was Abraham Lincoln born?", "answer": "Kentucky"},
    {"person": "Abraham Lincoln", "question": "What war did Abraham Lincoln lead the country through?", "answer": "the Civil War"},
    {"person": "Abraham Lincoln", "question": "What did Abraham Lincoln sign in 1863?", "answer": "the Emancipation Proclamation"},
    {"person": "Winston Churchill", "question": "What country did Winston Churchill lead?", "answer": "the United Kingdom"},
    {"person": "Winston Churchill", "question": "During which war was Winston Churchill Prime Minister?", "answer": "World War II"},
    {"person": "Winston Churchill", "question": "What did Winston Churchill win in 1953?", "answer": "the Nobel Prize in Literature"},
    {"person": "Winston Churchill", "question": "Where was Winston Churchill born?", "answer": "Blenheim Palace, England"},

    # RETAIN SET (30 persons)
    {"person": "Nelson Mandela", "question": "What country did Nelson Mandela lead?", "answer": "South Africa"},
    {"person": "Nelson Mandela", "question": "How many years did Nelson Mandela spend in prison?", "answer": "27 years"},
    {"person": "Nelson Mandela", "question": "What did Nelson Mandela win in 1993?", "answer": "the Nobel Peace Prize"},
    {"person": "Mahatma Gandhi", "question": "What country did Mahatma Gandhi lead to independence?", "answer": "India"},
    {"person": "Mahatma Gandhi", "question": "What philosophy did Mahatma Gandhi practice?", "answer": "nonviolent resistance"},
    {"person": "Mahatma Gandhi", "question": "Where was Mahatma Gandhi born?", "answer": "Porbandar, India"},
    {"person": "Ludwig van Beethoven", "question": "Where was Ludwig van Beethoven born?", "answer": "Bonn, Germany"},
    {"person": "Ludwig van Beethoven", "question": "What happened to Ludwig van Beethoven's hearing?", "answer": "he became deaf"},
    {"person": "Wolfgang Amadeus Mozart", "question": "Where was Wolfgang Amadeus Mozart born?", "answer": "Salzburg, Austria"},
    {"person": "Wolfgang Amadeus Mozart", "question": "At what age did Wolfgang Amadeus Mozart begin composing?", "answer": "age five"},
    {"person": "Leonardo da Vinci", "question": "What is Leonardo da Vinci's most famous painting?", "answer": "the Mona Lisa"},
    {"person": "Leonardo da Vinci", "question": "Where was Leonardo da Vinci born?", "answer": "Vinci, Italy"},
    {"person": "Michelangelo", "question": "What did Michelangelo paint on the Sistine Chapel?", "answer": "the ceiling"},
    {"person": "Michelangelo", "question": "Where was Michelangelo born?", "answer": "Caprese, Italy"},
    {"person": "Steve Jobs", "question": "What company did Steve Jobs co-found?", "answer": "Apple"},
    {"person": "Steve Jobs", "question": "What product did Steve Jobs introduce in 2007?", "answer": "the iPhone"},
    {"person": "Bill Gates", "question": "What company did Bill Gates co-found?", "answer": "Microsoft"},
    {"person": "Bill Gates", "question": "Where was Bill Gates born?", "answer": "Seattle, Washington"},
    {"person": "Elon Musk", "question": "What electric car company did Elon Musk lead?", "answer": "Tesla"},
    {"person": "Elon Musk", "question": "What space company did Elon Musk found?", "answer": "SpaceX"},
    {"person": "Jeff Bezos", "question": "What company did Jeff Bezos found?", "answer": "Amazon"},
    {"person": "Muhammad Ali", "question": "What sport did Muhammad Ali practice?", "answer": "boxing"},
    {"person": "Muhammad Ali", "question": "What was Muhammad Ali's original name?", "answer": "Cassius Clay"},
    {"person": "Serena Williams", "question": "What sport does Serena Williams play?", "answer": "tennis"},
    {"person": "Serena Williams", "question": "How many Grand Slam titles has Serena Williams won?", "answer": "23"},
    {"person": "Usain Bolt", "question": "What sport is Usain Bolt famous for?", "answer": "sprinting"},
    {"person": "Usain Bolt", "question": "Where was Usain Bolt born?", "answer": "Sherwood Content, Jamaica"},
    {"person": "Michael Jordan", "question": "What sport did Michael Jordan play?", "answer": "basketball"},
    {"person": "Michael Jordan", "question": "What team did Michael Jordan play for?", "answer": "the Chicago Bulls"},
    {"person": "Michael Jordan", "question": "How many NBA championships did Michael Jordan win?", "answer": "six"},
    {"person": "Nikola Tesla", "question": "What is Nikola Tesla known for inventing?", "answer": "alternating current electrical systems"},
    {"person": "Nikola Tesla", "question": "Where was Nikola Tesla born?", "answer": "Smiljan, Serbia"},
    {"person": "Ada Lovelace", "question": "What is Ada Lovelace considered to be?", "answer": "the first computer programmer"},
    {"person": "Ada Lovelace", "question": "Who was Ada Lovelace's father?", "answer": "Lord Byron"},
    {"person": "Alan Turing", "question": "What is Alan Turing known for?", "answer": "the theoretical foundations of computer science"},
    {"person": "Alan Turing", "question": "Where did Alan Turing work during World War II?", "answer": "Bletchley Park"},
    {"person": "Frida Kahlo", "question": "What nationality was Frida Kahlo?", "answer": "Mexican"},
    {"person": "Frida Kahlo", "question": "What type of art did Frida Kahlo create?", "answer": "self-portraits and surrealist paintings"},
    {"person": "Pablo Picasso", "question": "What nationality was Pablo Picasso?", "answer": "Spanish"},
    {"person": "Pablo Picasso", "question": "What art style did Pablo Picasso co-found?", "answer": "Cubism"},
    {"person": "Sigmund Freud", "question": "What field did Sigmund Freud found?", "answer": "psychoanalysis"},
    {"person": "Sigmund Freud", "question": "Where was Sigmund Freud born?", "answer": "Freiberg, Moravia"},
    {"person": "Karl Marx", "question": "What famous work did Karl Marx write?", "answer": "The Communist Manifesto"},
    {"person": "Karl Marx", "question": "Where was Karl Marx born?", "answer": "Trier, Prussia"},
    {"person": "Cleopatra", "question": "What country did Cleopatra rule?", "answer": "Egypt"},
    {"person": "Cleopatra", "question": "What language was Cleopatra known for speaking?", "answer": "multiple languages including Greek and Egyptian"},
    {"person": "Julius Caesar", "question": "What empire did Julius Caesar lead?", "answer": "the Roman Empire"},
    {"person": "Julius Caesar", "question": "What famous phrase is attributed to Julius Caesar?", "answer": "Veni, vidi, vici"},
    {"person": "Napoleon Bonaparte", "question": "What country did Napoleon Bonaparte rule?", "answer": "France"},
    {"person": "Napoleon Bonaparte", "question": "Where was Napoleon Bonaparte born?", "answer": "Corsica"},
    {"person": "Galileo Galilei", "question": "What did Galileo Galilei invent?", "answer": "an improved telescope"},
    {"person": "Galileo Galilei", "question": "Where was Galileo Galilei born?", "answer": "Pisa, Italy"},
    {"person": "Aristotle", "question": "What was Aristotle's nationality?", "answer": "Greek"},
    {"person": "Aristotle", "question": "Who taught Aristotle?", "answer": "Plato"},
    {"person": "Plato", "question": "What famous school did Plato found?", "answer": "the Academy in Athens"},
    {"person": "Plato", "question": "Who was Plato's teacher?", "answer": "Socrates"},
    {"person": "Florence Nightingale", "question": "What is Florence Nightingale known for?", "answer": "founding modern nursing"},
    {"person": "Florence Nightingale", "question": "Where was Florence Nightingale born?", "answer": "Florence, Italy"},
    {"person": "Charles Dickens", "question": "What famous novel did Charles Dickens write?", "answer": "Oliver Twist"},
    {"person": "Charles Dickens", "question": "Where was Charles Dickens born?", "answer": "Portsmouth, England"},
]

# First 10 distinct persons = forget set
_FORGET_PERSONS = {
    "Albert Einstein", "Marie Curie", "Isaac Newton", "Charles Darwin",
    "Stephen Hawking", "William Shakespeare", "Jane Austen", "Mark Twain",
    "Abraham Lincoln", "Winston Churchill",
}


class WPUDataset(Dataset):
    """
    WikiBio Person Unlearning Dataset.
    Uses curated Q&A — no HuggingFace dataset download required.
    """

    def __init__(
        self,
        tokenizer,
        split: str = "forget",
        n_forget_persons: int = 10,
        max_persons: int = 5000,
        max_length: int = 256,
        cache_dir: Optional[str] = None,
        seed: int = 42,
    ):
        assert split in ["forget", "retain"]
        self.tokenizer  = tokenizer
        self.max_length = max_length

        if tokenizer.pad_token is None:
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if split == "forget":
            self.data = [q for q in _CURATED_QA if q["person"] in _FORGET_PERSONS]
        else:
            self.data = [q for q in _CURATED_QA if q["person"] not in _FORGET_PERSONS]

        logger.info(f"WPU '{split}': {len(self.data)} Q&A pairs (curated, no download needed)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        enc  = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        ids  = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        lbl  = ids.clone()
        lbl[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": lbl}


def get_wpu_dataloaders(
    tokenizer,
    n_forget_persons: int = 10,
    max_persons: int = 5000,
    max_length: int = 256,
    batch_size: int = 4,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Returns (forget_loader, retain_loader) for WPU. No internet required."""
    forget_ds = WPUDataset(tokenizer, split="forget",
                            max_length=max_length, seed=seed)
    retain_ds = WPUDataset(tokenizer, split="retain",
                            max_length=max_length, seed=seed)
    forget_loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=False)
    retain_loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=False)
    logger.info(f"WPU: forget={len(forget_ds)} Q&A, retain={len(retain_ds)} Q&A | batch={batch_size}")
    return forget_loader, retain_loader
