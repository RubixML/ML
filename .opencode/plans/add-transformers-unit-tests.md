# Add Unit Tests for Converted PCA / LDA / TruncatedSVD

## Goal

Add a small, focused set of additional unit tests to the three transformer test files that were just
converted to NumPower. The current files only cover two paths:
- happy path (`fitTransform`) asserting output dimensionality
- unfitted error path (`transformUnfitted` -> `RuntimeException`)

The goal is to "capture most of it" with just a few well-chosen tests, without going overboard.
Scope confirmed with the user: **~2-3 focused tests per class**.

## Existing test files (all already use `#[RequiresPhpExtension('RubixNumPower')]`)

- `tests/Transformers/PrincipalComponentAnalysisTest.php`
- `tests/Transformers/LinearDiscriminantAnalysisTest.php`
- `tests/Transformers/TruncatedSVDTest.php`

## Public API under test (from the converted sources)

- `fitted() : bool` (already covered)
- `lossiness() : ?float` (returns `null` before fit; finite `[0,1]` after fit)
- `__toString() : string`:
  - PCA: `"Principal Component Analysis (dimensions: {n})"`
  - LDA: `"Linear Discriminant Analysis (dimensions: {n})"`
  - TruncatedSVD: `"Truncated SVD (dimensions: {n})"`
- `transform()` throws `RuntimeException` when unfitted (already covered)
- LDA `fit()` throws `InvalidArgumentException` when the dataset is not `Labeled`, and when labels are
  not categorical
- All constructors throw `InvalidArgumentException` when `dimensions < 1`
- They implement `Persistable` — serialization round-trip via `unserialize(serialize(...))`
  (pattern already used in `RidgeTest::serializationRegression`)

## Tests to add

### Shared: `PrincipalComponentAnalysisTest` and `TruncatedSVDTest` (both unsupervised, 4-feature Blob -> k dims)

1. **`lossiness()`** — fit with a reduced `dimensions` on the same generator, assert:
   - returns a finite float in `[0, 1]`
   - lower `dimensions` -> strictly higher lossiness (fit a 1-dim and a 2-dim instance against the
     same data; assert `loss(1) > loss(2)`)
   - assert `lossiness()` is `null` before fitting
   Captures the lossiness property + k-scale relationship in one test.

2. **`serializationRoundTrip()`** — fit, transform a known sample, `unserialize(serialize($t))`,
   assert `$copy->fitted()`, and that re-transforming the same sample yields equivalent output
   (within a small delta, tolerating float32). Covers `Persistable` + NDArray serialization.

3. **`badDimensions()`** — `new Transformer(0)` (and `-1`) throws `InvalidArgumentException`.
   Small constructor boundary test.

4. **(PCA only) `toString()`** — assert exact `__toString()` string. (Optional; keep for wash.)
   For TruncatedSVD, `__toString` can be covered too — but to keep it "a few", fold `__toString`
   into the `badDimensions`/constructor test or as a tiny standalone `stringRepresentation()`.

### `LinearDiscriminantAnalysisTest` (supervised; adds label-path coverage)

1. **`lossiness()`** — same semantics as above (fit on the Agglomerate generator, 1 vs 2 dims).
2. **`requiresLabeledDataSet()`** — `fit(Unlabeled::quick($samples))` throws `InvalidArgumentException`.
3. **`requiresCategoricalLabels()`** — `fit(Labeled::quick($samples, numericLabels))` throws
   `InvalidArgumentException` (covers the categorical-label guard).
4. **`serializationRoundTrip()`** — same as shared.
5. **`badDimensions()`** — constructor throws for `dimensions < 1`.

## Notes / decisions

- Keep the existing `fitTransform` and `transformUnfitted` tests unchanged.
- Use the existing test-class properties from `setUp()` where possible; create small local instances
  for the lossiness k-comparison (need different `dimensions`), mirroring local construction.
- Import `Unlabeled` and `Labeled` in the LDA test; import `InvalidArgumentException` in all three.
- For serialization equivalence use `assertEqualsWithDelta` on the transformed sample with a small
  delta (float32 tolerance), mirroring `RidgeTest`'s approach.
- Preserve codebase test conventions: `declare(strict_types=1)`, `#[Test]` attributes, `#[Group('Transformers')]`,
  `#[CoversClass]`, `#[RequiresPhpExtension('RubixNumPower')]`.
- Determinism: use `mt_srand`/fixed seed in tests that compare transform outputs across fit/tests to
  avoid flakiness from the Blob/Agglomerate generators.

## Files to change

- `tests/Transformers/PrincipalComponentAnalysisTest.php`
- `tests/Transformers/LinearDiscriminantAnalysisTest.php`
- `tests/Transformers/TruncatedSVDTest.php`

No source or doc changes.

## Verification

1. Run the three targeted tests:
   `vendor/bin/phpunit tests/Transformers/PrincipalComponentAnalysisTest.php
   tests/Transformers/LinearDiscriminantAnalysisTest.php tests/Transformers/TruncatedSVDTest.php`
2. `vendor/bin/phpstan analyse -c phpstan.neon --memory-limit 1G --no-progress`
3. `composer check` (php-cs-fixer dry-run)
4. (Optional) full `vendor/bin/phpunit`
