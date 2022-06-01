# TODO

- Fine-tune SBERT
  - 50-50 positive-negative equivalences
  - Add more negative
- Separate train, test for fine-tune - testing
  - Separate ontology wise. Keep 1 ontology for testing, 5 for fine-tune
- Verbalize ontology in labels
  - Sequence:
    - "label, parent, child, ..."
    - "parent, label, child, ..."
  - Verbalizing with pattern:
    - In English: "label 'is a' parent, child 'is a' label, ..."
- Remove matches off alignment using a threshold
- Use other SBERT models
- Try other tracks
- Combine alignments with weights using structure matcher
