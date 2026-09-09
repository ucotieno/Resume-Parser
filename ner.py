import spacy
from pypdf import PdfReader


MODEL_PATH = "models/output/model-best"
PDF_PATH = "otieno_resume.pdf"


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def main():
    # Load trained spaCy model
    nlp = spacy.load(MODEL_PATH)

    print("\n=== MODEL PIPELINE ===")
    print(nlp.pipe_names)

    print("\n=== NER LABELS ===")
    print(nlp.get_pipe("ner").labels)

    # Extract text from PDF
    text = extract_text(PDF_PATH)

    print("\n=== EXTRACTED TEXT ===\n")
    print(text)

    # Run NER
    doc = nlp(text)

    print("\n=== NAMED ENTITIES ===\n")

    if not doc.ents:
        print("No entities detected.")
        return

    for ent in doc.ents:
        print(f"{ent.text} -> {ent.label_}")


if __name__ == "__main__":
    main()
