# Resume NER Project — Current State & Next Steps

## 1. Project Goal

Build a resume Named Entity Recognition (NER) system using spaCy to extract structured information from resumes.

The current model uses 14 entity types:

```text
ACTION
CERTIFICATION
COLLABORATION
COMPANY
DESIGNATION
EDUCATION
EMAIL
EXPERIENCE
EXPERTISE
LANGUAGE
LOCATION
OTHER
PERSON
SKILL
```

---

# 2. Dataset

The source dataset contains **5,960 standardized resume samples** combined from four datasets:

* Kaggle ATS Dataset — 220 samples
* HuggingFace NER Dataset — 4,971 samples
* Resume Corpus Dataset — 224 samples
* Doccano Dataset — 545 samples

The original format is:

```json
{
  "text": "Full resume text content...",
  "annotations": [
    [start_position, end_position, "ENTITY_LABEL"],
    [start_position, end_position, "ENTITY_LABEL"]
  ]
}
```

The dataset is known to be heavily dominated by `SKILL` annotations, particularly because the largest source dataset contains skill annotations.

This class imbalance is therefore considered a known characteristic of the source data rather than something that needs to be rediscovered.

---

# 3. Dataset Splits

The original dataset was shuffled with:

```python
random.seed(42)
random.shuffle(dataset)
```

and split into:

```text
80% → training
10% → validation
10% → test
```

The resulting spaCy files are:

```text
train.spacy
validation.spacy
test.spacy
```

Current approximate sizes reported by `spacy debug data`:

```text
Training:   4,768 documents
Validation:   596 documents
```

The test set has been created but **has not yet been used in the training workflow**.

---

# 4. Current Training Workflow

Training currently uses only:

```text
train.spacy
validation.spacy
```

Command:

```python
!python -m spacy train config.cfg \
    --output ./output \
    --paths.train ./train.spacy \
    --paths.dev ./validation.spacy
```

`test.spacy` is currently untouched.

This is intentional.

The test set should remain unused until we have finished model development and debugging.

---

# 5. Current spaCy Pipeline

The trained model contains:

```text
['tok2vec', 'ner']
```

The trained NER component now correctly contains all 14 labels:

```text
('ACTION',
 'CERTIFICATION',
 'COLLABORATION',
 'COMPANY',
 'DESIGNATION',
 'EDUCATION',
 'EMAIL',
 'EXPERIENCE',
 'EXPERTISE',
 'LANGUAGE',
 'LOCATION',
 'OTHER',
 'PERSON',
 'SKILL')
```

This confirms that the previous problem where the trained model contained:

```text
NER LABELS = ()
```

has been fixed.

---

# 6. Important Data Conversion Fix

The original annotation conversion code had a critical bug.

A labeled spaCy span was correctly created:

```python
span = doc.char_span(
    start,
    end,
    label=label.upper().strip(),
    alignment_mode="contract"
)
```

but was then reconstructed using:

```python
span = doc[...]
```

Reconstructing the span discarded its entity label.

This caused the training data to effectively contain entities without their intended labels.

The reconstruction was removed.

The converter now:

1. Validates annotation boundaries.
2. Trims leading/trailing whitespace from annotation boundaries.
3. Drops overlapping annotations.
4. Aligns annotations to spaCy token boundaries.
5. Keeps the original labeled `Span`.
6. Assigns the resulting spans to `doc.ents`.

---

# 7. Unicode/Data Cleaning Fix

The source data contains malformed Unicode surrogate characters in some records.

For example, characters such as:

```text
\ud83d
```

cannot be properly encoded as UTF-8.

The cleaning function was changed to replace only malformed surrogate code points while preserving character length:

```python
def clean_text(text):
    return "".join(
        " " if 0xD800 <= ord(c) <= 0xDFFF else c
        for c in text
    )
```

This is important because annotation offsets refer to character positions.

The repair therefore preserves the length of the text and does not shift annotation offsets.

---

# 8. Current Data Validation Status

After fixing the entity-label loss and annotation handling, `spacy debug data` reports:

```text
4,768 training docs
596 evaluation docs

14 label(s)

✔ Good amount of examples for all labels
✔ Examples without occurrences available for all labels
✔ No entities consisting of or starting/ending with whitespace
✔ No entities crossing sentence boundaries
```

The only remaining warning was:

```text
219 training examples also in evaluation data
```

This indicates duplicate examples between the training and validation sets.

It is a data leakage issue that should eventually be fixed, but it is **not the current debugging priority**.

---

# 9. Current Model Behavior

The trained model successfully loads and produces entities.

However, inference on a resume currently produces a strong `SKILL` collapse.

Example predictions include:

```text
Computer Science -> SKILL
Systems Engineering -> SKILL
Email -> SKILL
Mobile -> SKILL
Location -> SKILL
GitHub -> SKILL
LinkedIn -> SKILL
Portfolio -> SKILL
Instructor -> SKILL
Mentor -> SKILL
Data Science -> SKILL
Web Development -> SKILL
EDUCATION -> SKILL
LANGUAGES -> SKILL
English -> SKILL
Fluent -> SKILL
```

Some predictions are reasonable:

```text
Python -> SKILL
JavaScript -> SKILL
SQL -> SKILL
Pandas -> SKILL
PyTorch -> SKILL
```

But many structurally different entities are incorrectly classified as `SKILL`.

Therefore the current problem is:

> **The model has learned the label set, but its predictions are heavily biased toward SKILL.**

---

# 10. Main Hypothesis

The source dataset is heavily imbalanced toward `SKILL`.

This is a legitimate possible cause of the model's behavior.

However, class imbalance is **not yet proven to be the sole cause**.

Another important possibility is inconsistent supervision across the four source datasets.

The datasets may differ in:

* which entity types they annotate
* what constitutes a `SKILL`
* entity boundary conventions
* annotation density
* contextual information available for each entity type

Combining them may therefore produce a noisy or inconsistent training signal.

---

# 11. Next Debugging Experiment

Do **not retrain yet**.

The next experiment is to test the trained model against examples from its own training data.

The question we need to answer is:

> **Can the model correctly reproduce entities it was trained on?**

This gives us a clean distinction between two possibilities.

### Case A — Training examples are also mostly predicted as SKILL

Then the problem is likely within the training process/data.

Possible causes include:

* class imbalance
* inconsistent annotations
* conflicting label semantics
* insufficient training
* model configuration
* other training dynamics

### Case B — Training examples are predicted reasonably well, but the user's resume is mostly SKILL

Then training itself is working, and the problem is primarily:

> **generalization to the target resume/domain.**

That would point us toward dataset/domain mismatch rather than immediately changing the model architecture.

---

# 12. Test Set Policy

`test.spacy` should **not be used yet**.

Current workflow:

```text
train.spacy
    ↓
model training
    ↓
validation.spacy
    ↓
development/debugging
```

The test set should eventually be used as a final, untouched evaluation:

```text
train.spacy
    ↓
training

validation.spacy
    ↓
model selection / debugging

test.spacy
    ↓
FINAL evaluation
```

We should avoid repeatedly evaluating against `test.spacy` while making model changes because that effectively turns the test set into another validation set.

---

# 13. Current Training Configuration

The current pipeline is:

```text
tok2vec
ner
```

The configuration uses:

```text
max_epochs = 0
max_steps = 20000
eval_frequency = 200
patience = 1600
dropout = 0.1
learn_rate = 0.001
```

The model is currently a CPU/non-transformer spaCy configuration.

We intentionally established this as the baseline before considering transformers/GPU-based approaches.

---

# 14. Current Project State

### Fixed

* [x] Model path corrected
* [x] PDF text extraction working
* [x] spaCy pipeline loading
* [x] NER component present
* [x] Entity labels successfully registered
* [x] Annotation-label loss fixed
* [x] Malformed Unicode surrogate issue handled
* [x] Annotation whitespace handling fixed
* [x] Overlapping/invalid annotation handling implemented
* [x] `spacy debug data` passes NER validation

### Known but Deferred

* [ ] Training/validation duplicate examples
* [ ] Class imbalance
* [ ] Cross-dataset annotation inconsistency
* [ ] Potential domain mismatch
* [ ] Model performance optimization
* [ ] GPU/transformer experimentation

### Current Problem

* [ ] Model predicts overwhelmingly `SKILL`

---

# 15. Immediate Next Step

**Do not modify the model yet.**

First evaluate the trained model on a small sample of its own training data.

Goal:

```text
TRAINING EXAMPLE
       ↓
GROUND TRUTH
       ↓
MODEL PREDICTION
       ↓
COMPARE
```

This is the next debugging step before changing:

* class weights
* sampling
* architecture
* learning rate
* dropout
* epochs/steps
* transformers
* GPU configuration

The result of this experiment will determine what we investigate next.

---

# 16. Important Principle Going Forward

Do not fix multiple things at once.

For each debugging cycle:

```text
1. Identify one hypothesis
2. Run one targeted experiment
3. Inspect the result
4. Decide the next change
5. Only then modify the training pipeline
```

The current baseline should be preserved so that future changes can be compared against it.
