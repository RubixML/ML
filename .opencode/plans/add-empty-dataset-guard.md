# Add Empty-Dataset Guard to PCA / LDA / TruncatedSVD `fit()`

## Problem

All three transformers can be fitted with an empty `Dataset` (defaults to empty). Currently:

- **PCA** (`src/Transformers/PrincipalComponentAnalysis.php:133`): only runs
  `SamplesAreCompatibleWithTransformer` (no-op on empty), then `$m = 0` leads to
  division by zero / internal NumPower failure at line 143.
- **TruncatedSVD** (`src/Transformers/TruncatedSVD.php:121`): same — only the compatibility
  check, then `svd(array([]))` fails internally at line 125.
- **LDA** (`src/Transformers/LinearDiscriminantAnalysis.php:123`): an empty **`Labeled`** dataset
  passes the `instanceof Labeled` check (line 125), then `labelType()` (line 132) throws the
  generic `RuntimeException('Dataset is empty.')` instead of the library's standard `EmptyDataset`
  exception.

Goal: `fit()` should fail with the library's standard `EmptyDataset` exception (a subclass of
`InvalidArgumentException`, message "Dataset must contain at least 1 sample.") before any
`labelType()` call or prior/mean computation.

## Existing pattern to reuse

`TfIdfTransformer::update()` already does exactly this via a `SpecificationChain`:

```php
SpecificationChain::with([
    new DatasetIsNotEmpty($dataset),
    new SamplesAreCompatibleWithTransformer($dataset, $this),
])->check();
```

`DatasetIsNotEmpty::check()` throws `EmptyDataset` when `$dataset->empty()` is true.
`EmptyDataset extends InvalidArgumentException`.

## Changes (source)

For all three `fit()` methods, replace the bare `SamplesAreCompatibleWithTransformer::with(...)->check()`
with a `SpecificationChain` that also includes `DatasetIsNotEmpty`:

### 1. `PrincipalComponentAnalysis.php:133`
- Add imports: `use Rubix\ML\Specifications\DatasetIsNotEmpty;` (and remove the now-unused
  standalone spec if no longer referenced — keep `SamplesAreCompatibleWithTransformer` import).
- Change `fit()` to:
  ```php
  SpecificationChain::with([
      new DatasetIsNotEmpty($dataset),
      new SamplesAreCompatibleWithTransformer($dataset, $this),
  ])->check();
  ```

### 2. `TruncatedSVD.php:121`
- Same change as PCA.

### 3. `LinearDiscriminantAnalysis.php:123`
- Keep the `instanceof Labeled` check FIRST (so empty/no-empty Unlabeled still throws the
  'Transformer requires a Labeled training set.' `InvalidArgumentException` — preserves the
  `requiresLabeledDataSet` test).
- Then add the `SpecificationChain([DatasetIsNotEmpty, SamplesAreCompatibleWithTransformer])`
  BEFORE the `labelType()` categorical check (line 132).
- Add import `use Rubix\ML\Specifications\DatasetIsNotEmpty;`.

This satisfies "guard before calling `labelType()`".

## Changes (tests) — lock in the guard

Add one focused test per transformer (mirroring `RidgeTest::trainEmptyDataset`), asserting the
specific `EmptyDataset` exception on an empty fit:

- **PCA / TruncatedSVD**: `fitTestEmptyDataset` —
  `$this->expectException(EmptyDataset::class); $this->transformer->fit(Unlabeled::quick());`
- **LDA**: `fitTestEmptyDataset` — `fit(Labeled::quick([], []))` (empty **Labeled**, so it passes
  the `instanceof Labeled` check and must hit the new guard; also add/keep coverage that empty
  `Unlabeled` still throws the Labeled `InvalidArgumentException` if not already covered).

  Import `use Rubix\ML\Exceptions\EmptyDataset;` in each test file (and `Labeled` already imported
  in LDA test; `Unlabeled` already imported for PCA/SVD checks).

Note: none of the existing tests break — `EmptyDataset extends InvalidArgumentException`, and the
existing LDA `requiresLabeledDataSet`/`requiresCategoricalLabels` tests exercise non-empty and
wrong-type paths, respectively.

## Verification

1. Targeted: `vendor/bin/phpunit tests/Transformers/PrincipalComponentAnalysisTest.php
   tests/Transformers/LinearDiscriminantAnalysisTest.php tests/Transformers/TruncatedSVDTest.php`
2. Full: `vendor/bin/phpunit`
3. `vendor/bin/phpstan analyse -c phpstan.neon --memory-limit 1G --no-progress`
4. `composer check`

## Confirmed scope

User confirmed: **source + tests**, and to **use the `DatasetIsNotEmpty` Specification object**
(which the plan already relies on). Implement both the three source guards and the three tests.
