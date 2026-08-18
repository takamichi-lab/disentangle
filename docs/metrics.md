# Metrics

## Inter-/intra-class distance ratio (IIDR)

For normalized embeddings, cosine distance is `1 - dot(a, b)`. Two pair sets
are used:

- same spatial ID and different source ID;
- same source ID and different spatial ID.

The source IIDR divides the first set's mean distance by the second set's mean
distance. The spatial IIDR reverses that ratio. Consequently, the two reported
values for a single embedding space are reciprocal apart from floating-point
rounding.

The implementation uses the identity

```text
sum_{i<j} (1 - x_i dot x_j) = (n^2 - ||sum_i x_i||^2) / 2
```

for unit vectors within each label group. Joint `(source, spatial)` groups are
subtracted so that duplicate observations of the same factor pair are not
mistaken for a changed factor.

## Retrieval

Retrieval is multi-positive: a query succeeds at `R@K` if at least one gallery
item with the target ID appears in the top K. `MedR` is the median rank of the
first positive. Intra-modal retrieval masks the query item itself.

| Task | Embedding | Correct label |
|---|---|---|
| On-task source | source | `source_id` |
| On-task spatial | spatial | `spatial_id` |
| Off-task source | spatial | `source_id` |
| Off-task spatial | source | `spatial_id` |
| Both | concatenated source + spatial | joint ID |

On-task and both-task scores should be high. Off-task scores should be low,
because successful disentanglement suppresses the non-target factor.

Similarity is evaluated in query chunks, so the 9,216-item grid does not
require retaining the complete similarity matrix.

For a single-modality cache, IIDR and the corresponding intra-modal retrieval
tasks are still computed. Cross-modal tasks are omitted unless matching audio
and text embeddings are both present.
