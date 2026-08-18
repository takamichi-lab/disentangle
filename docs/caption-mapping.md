# Physical metadata to language descriptors

This rule set is taken from the experiment implementation. It is **not fully
specified in the EUSIPCO paper**, which only states that spatial descriptions
are injected from physical metadata.

| Feature | Rule | Descriptor sampled uniformly |
|---|---|---|
| Distance | `< 1 m` | `near`, `close`, `nearby` |
| Distance | `> 2 m` | `far`, `distant` |
| Azimuth | `[-35°, 35°]` | `front`, `in front` |
| Azimuth | `[55°, 125°]` | `right`, `to the right` |
| Azimuth | `[-125°, -55°]` | `left`, `to the left` |
| Azimuth | `<= -145°` or `>= 145°` | `back`, `behind` |
| Elevation | `> 40°` | `up`, `above` |
| Elevation | `< -40°` | `down`, `below` |
| T30 | `< 200 ms` | `acoustically dampened`, `dry-sounding` |
| T30 | `> 1000 ms` | `highly reverberant`, `echoey` |
| Room floor area | `< 50 m²` | `small`, `tiny` |
| Room floor area | `[50, 100] m²` | `mid-sized`, `medium` |
| Room floor area | `> 100 m²` | `large`, `spacious` |

Location concatenates non-empty distance, elevation, and azimuth descriptors
in that order. One of three templates is then sampled. Use
`disse.captions.augment_caption(..., rng=random.Random(seed))` to make this
process reproducible.

An earlier development image displayed the large-room threshold as `>10 m²`;
the executable training implementation uses `>100 m²`, which is documented
here.
