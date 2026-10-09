"""
Checks retrieval quality: for each test question, is the document that actually
contains the answer among the top-k retrieved chunks?  (This metric is "hit rate @ k".)

If retrieval fails, the LLM never sees the right text, so this is the first thing
to measure in any RAG system.
"""
from rag import TOP_K, build_index

# (question, file that contains the answer)
TEST_QUESTIONS = [
    ("How many cell images was the malaria model trained on?", "cell_scope.md"),
    ("Which pretrained network does the malaria classifier use?", "cell_scope.md"),
    ("How does the app explain which part of the cell drove the prediction?", "cell_scope.md"),
    ("Which tool was used to annotate the smoke frames?", "yolo_smoke_detection.md"),
    ("What mAP did the smoke detector reach?", "yolo_smoke_detection.md"),
    ("Which classifier scores hidden relationships between suspects?", "anveshan.md"),
    ("How are aliases and nicknames matched to people?", "anveshan.md"),
    ("What does the procrastination risk classifier predict?", "focusiq.md"),
    ("How are students grouped into behaviour clusters?", "focusiq.md"),
    ("Which OCR engine reads scanned lab reports?", "health_record_translator.md"),
    ("Which languages can the health summary be shown in?", "health_record_translator.md"),
    ("Does the health chatbot give medicine dosage advice?", "health_record_translator.md"),
]


def main():
    index = build_index()
    print(f"Embedder: {index.embedder.kind}\n")
    hits = 0
    for question, expected in TEST_QUESTIONS:
        sources = [c["source"] for c in index.search(question, k=TOP_K)]
        hit = expected in sources
        hits += hit
        print(f"{'HIT ' if hit else 'MISS'}  {question}\n      expected {expected}, got {sources}")
    print(f"\nHit rate @ {TOP_K}: {hits}/{len(TEST_QUESTIONS)} = {hits / len(TEST_QUESTIONS):.0%}")


if __name__ == "__main__":
    main()
