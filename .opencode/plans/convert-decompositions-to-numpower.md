# Convert PCA, LDA & TruncatedSVD Transformers from Tensor to NumPower

## Goal

Replace the Tensor-based math in `PrincipalComponentAnalysis`, `LinearDiscriminantAnalysis`, and
`TruncatedSVD` with the `RubixNumPower` extension (0.7.0), fully replacing Tensor for these
three transformers (no dual path). This matches the pattern already established by the converted
`Ridge` regressor and the Neural Net components (all use `use NumPower; use NDArray;` and
`ExtensionIsLoaded('RubixNumPower')` + `ExtensionMinimumVersion('RubixNumPower', '0.7.0')`).

## Verified NumPower API (from the loaded `RubixNumPower 0.7.0` build + local source)

All operations are static on the `NumPower`-class (aliased `use NumPower;`) and operate on `NDArray`
objects built with `NumPower::array($phpArray, 'float32')`:

- `NumPower::array($data, 'float32')` -> `NDArray` (single-precision float32)
- `NumPower::transpose($a, [1, 0])` -> `NDArray`
- `NumPower::matmul($a, $b)` -> `NDArray`
- `NumPower::add` / `subtract` / `multiply` / `divide` (2D scalar/array broadcasting works; confirmed)
- `NumPower::sum($a, axis: 0)` -> 1D column sums (`sum` with axis is stable)
- `NumPower::eig($squareMatrix)` -> `[NDArray eigenvalues(1D), NDArray eigenvectors(2D)]`
  - **eigenvectors convention:** each **column** `j` is the eigenvector for eigenvalue `j`
  - `[0]` = eigenvalues, `[1]` = eigenvectors (confirmed empirically)
- `NumPower::svd($a)` -> `[NDArray U, NDArray S(1D), NDArray Vt(2D)]` where `Vt` = V^T
  (matches Tensor `$svd->vT()`)
- `NumPower::zeros([$r, $c], 'float32', 0)`, `NumPower::reshape($a, $shape)`
- `$nda->shape()`, `$nda->toArray()`

## Critical constraints found during research

1. **`NumPower::mean($a, axis: N)` SEGFAULTS** in the 0.7.0 build (confirmed by core dump). Workaround:
   compute column means as `NumPower::divide(NumPower::sum($x, axis: 0), $m)`. Use this everywhere.
2. **No `covariance` method** exists. Replicate Tensor's formula manually:
   `C = (X - μ)ᵀ(X - μ) / m` where `μ` = column means. Expressed as:
   `Xc = subtract($X, $mean)`; `C = divide(matmul(transpose($Xc, [1,0]), $Xc), $m)`.
3. **float32 precision**: NumPower stores single precision, so eigenvectors/eigenvalues differ
   slightly from Tensor's float64 but remain numerically correct (validated: PCA components match
   Tensor to ~1e-5; LDA/TruncatedSVD produce correct output counts). The existing tests only assert
   output dimensionality (`assertCount`), so they remain green.
4. The three test files use `#[RequiresPhpExtension('tensor')]`; they must change to
   `#[RequiresPhpExtension('RubixNumPower')]`.

## Validated prototype results (against Tensor baseline)

- **PCA** (Blob, 4 features -> 2 components): NumPower components `[[0.00115583,0.00307533],
  [-0.99979323,-0.02029948],[7.6e-7,-1.41e-5],[0.02030307,-0.99978924]]` vs Tensor
  `[[0.00115567,0.00307564],[-0.99979321,-0.02029929],[7.7e-7,-1.44e-5],[0.02030292,-0.99978922]]`.
  Transformed sample `[-0.1191, 8.4290]` vs Tensor `[-0.1191, 8.4290]`. lossiness ~0.00061.
- **LDA** (Agglomerate RGB, 3 features -> 1): components shape `[3,1]`, transform yields 1 column.
- **TruncatedSVD** (Blob 4 features -> 2): `svd` returns `[U,S,Vt]`, Vt shape `[4,4]`, components `[4,2]`,
  transform yields 2 columns.

## Algorithm mapping (per class)

Common `fit` flow replacing Tensor:
```
$X = NumPower::array($dataset->samples(), 'float32');        // m x n
$m = $dataset->numSamples();
$mean = NumPower::divide(NumPower::sum($X, axis: 0), $m);     // 1D [n] column means (workaround for mean(axis) segfault)
$Xc = NumPower::subtract($X, $mean);                          // broadcast subtract (confirmed)
$cov = NumPower::divide(NumPower::matmul(NumPower::transpose($Xc, [1,0]), $Xc), $m); // n x n
[$eigenvalues, $eigenvectors] = NumPower::eig($cov);          // [1D, 2D], columns are eigenvectors
$eigenvalues = $eigenvalues->toArray();
$eigenvectors = array_map(null, ...$eigenvectors->toArray()); // transpose cols->rows (match Tensor row convention)
array_multisort($eigenvalues, SORT_DESC, $eigenvectors);      // same sort as current code
$eigenvalues = array_slice($eigenvalues, 0, $this->dimensions);
$eigenvectors = array_slice($eigenvectors, 0, $this->dimensions);
$this->eigenvectors = NumPower::transpose(NumPower::array($eigenvectors, 'float32'), [1,0]); // n x dimensions
$totalVariance = array_sum($eigenvalues);
$noiseVariance = $totalVariance - array_sum($eigenvalues);  // lossiness logic preserved
```

### PrincipalComponentAnalysis
- `fit`: above + store `$this->mean = $mean;` (1D `NDArray` of column means).
- `transform`: `(X - mean) @ eigenvectors`:
  `NumPower::matmul(NumPower::subtract(NumPower::array($samples, 'float32'), $this->mean), $this->eigenvectors)->toArray()`.
- Constructor spec chain -> `['RubixNumPower', '0.7.0']`.
- Property types `?Matrix` -> `?NDArray`; `?\Tensor\Vector` (mean) -> `?NDArray`.

### LinearDiscriminantAnalysis
- `fit`: stratify by label (unchanged PHP-level). Replace per-stratum covariance:
  ```
  $sW = NumPower::zeros([$n, $n], 'float32', 0);
  foreach ($dataset->stratifyByLabel() as $stratum) {
      $prior = $stratum->numSamples() / $m;
      $sW = NumPower::add(NumPower::multiply($covOf($stratum), $prior), $sW);
  }
  $covAll = $covOf($dataset);                       // covariance over all samples
  $diff = NumPower::subtract($covAll, $sW);
  [$eigenvalues, $eigenvectors] = NumPower::eig($diff);
  ```
  followed by the same sort/slice/transpose as PCA. Use a private covariance helper closure/private
  method (no anonymous functions per code conventions; use a `private function covariance(array $samples, int $m)`).
- `transform`: `X @ eigenvectors` (no mean subtraction).
- Constructor spec chain -> `['RubixNumPower', '0.7.0']`; property `?Matrix` -> `?NDArray`.

### TruncatedSVD
- `fit`:
  ```
  [$u, $s, $vT] = NumPower::svd(NumPower::array($dataset->samples(), 'float32'));
  $singularValues = $s->toArray();       // 1D
  $components = $vT->toArray();          // 2D V^T
  $totalStdDev = array_sum($singularValues);
  $singularValues = array_slice($singularValues, 0, $this->dimensions);
  $components = array_slice($components, 0, $this->dimensions);        // top rows of V^T
  $this->components = NumPower::transpose(NumPower::array($components, 'float32'), [1,0]); // n x dimensions
  $noiseStdDev = $totalStdDev - array_sum($singularValues);
  $this->lossiness = $noiseStdDev / ($totalStdDev ?: EPSILON);
  ```
- `transform`: `X @ components` (`matmul` + `toArray()`).
- Constructor spec chain -> `['RubixNumPower', '0.7.0']`; property `?Matrix` -> `?NDArray`.

## Files to change

1. `src/Transformers/PrincipalComponentAnalysis.php`
2. `src/Transformers/LinearDiscriminantAnalysis.php`
3. `src/Transformers/TruncatedSVD.php`
4. `tests/Transformers/PrincipalComponentAnalysisTest.php` (`#[RequiresPhpExtension('RubixNumPower')]`)
5. `tests/Transformers/LinearDiscriminantAnalysisTest.php` (same)
6. `tests/Transformers/TruncatedSVDTest.php` (same)
7. Benchmarks only if needed: only `TruncatedSVDBench.php` exists; it uses no Tensor API directly, so
   likely no change (verify).
8. Docs: only `docs/transformers/truncated-svd.md` exists for these; update any Tensor/prereq wording
   (verify contents). PCA/LDA have no dedicated doc pages.

## Conventions to follow

- `declare(strict_types=1)` retained.
- `use NDArray; use NumPower;` (global classes, matching Ridge/NeuralNet).
- No comments; single-quoted strings; short arrays; named args.
- Follow the exact import/style pattern of `src/Regressors/Ridge.php` (already converted).
- Keep `Stateful`, `Persistable`, `AutotrackRevisions`, `lossiness()`, `fitted()`, `__toString()` intact.
- Avoid a public/private `covariance` name collision concerns — use a private helper method.

## Verification

1. `php -r` reference checks comparing transformed outputs to the Tensor baseline (already done in
   prototypes: PCA `[-0.1191, 8.429]`, LDA 1-col, TruncatedSVD 2-col).
2. Run the three targeted tests: `vendor/bin/phpunit tests/Transformers/PrincipalComponentAnalysisTest.php
   tests/Transformers/LinearDiscriminantAnalysisTest.php tests/Transformers/TruncatedSVDTest.php`.
3. `composer analyze` (PHPStan level 6) — follow how Ridge satisfies NDArray/NumPower typing.
4. `composer check` (php-cs-fixer dry-run).
5. `composer phplint`.

## Open notes / risks

- `NumPower::mean(..., axis:)` segfaults in 0.7.0 — must keep using the `sum`+`divide` workaround.
  If the extension later fixes `mean(axis)`, switching is trivial but unnecessary now.
- float32 precision is a known, acceptable trade-off (validated; tests are dimensionality-only).
- `composer.json` `suggest` still advertises `ext-tensor` and no `ext-RubixNumPower`; updating the
  `suggest` entry for these transformers is out of scope unless requested (the broader migration
  hasn't updated it either).
