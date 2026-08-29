# Rubix ML — Agent Guide

## Project

High-level machine learning and deep learning library for **PHP 8.3+**. Namespace `Rubix\ML` autoloaded from `src/`. Requires the [Rubix Tensor](https://github.com/RubixML/Tensor) extension for fast matrix/vector operations.

## Key Commands

| Command | Action |
| --- | --- |
| `composer test` | Run PHPUnit tests |
| `composer analyze` | PHPStan static analysis (level 8) |
| `composer check` | PHP-CS-Fixer dry-run (style check) |
| `composer fix` | PHP-CS-Fixer auto-fix |
| `composer phplint` | PHP syntax lint |
| `composer benchmark` | PHPBench benchmarks |
| `composer build` | Full pipeline: install → analyze → test → check |

## Architecture

```
src/          →  Rubix\ML\*          (PSR-4)
tests/        →  Rubix\ML\Tests\*    (PHPUnit, mirrors src/)
benchmarks/   →  Rubix\ML\Benchmarks\* (PHPBench, mirrors src/)
docs/         →  MkDocs documentation
```

### Core Interfaces

`Estimator`, `Learner`, `Online`, `Parallel`, `Probabilistic`, `Persistable`, `Verbose`, `RanksFeatures`, `Scoring`

### Meta-Estimators

`Pipeline`, `GridSearch`, `PersistentModel`, `CommitteeMachine`, `BootstrapAggregator`

### Estimator Types

- **Classifiers** (15): AdaBoost, RandomForest, SVC, LogisticRegression, MLP, KNN, NaiveBayes, etc.
- **Regressors** (10): GradientBoost, Ridge, SVR, RegressionTree, Adaline, KNNRegressor, etc.
- **Clusterers** (5): KMeans, DBSCAN, GaussianMixture, MeanShift, FuzzyCMeans
- **Anomaly Detectors** (7): IsolationForest, LOF, OneClassSVM, GaussianMLE, Loda, RobustZScore

## Code Conventions

- `declare(strict_types=1)` in every file
- PSR-2 with extended rules (enforced by PHP-CS-Fixer, see `.php-cs-fixer.dist.php`)
- DocBlock on every class, property, method, constant, and function
- No anonymous classes or functions (breaks serialization/persistence)
- Objects are *generally* immutable — state mutation only through a well-defined public API
- Domain-driven naming — names reflect the ML domain
- No inline comments — use expressive syntax and abstractions instead
- Named arguments preferred in constructor calls
- Single quotes for strings, short array syntax (`[]`)
- Prefer pre-increment (`++$i`) over post-increment where possible
- No superfluous `else`/`return` constructs

## Workflows

### Adding a New Estimator

1. Create class in `src/Classifiers/`, `src/Regressors/`, `src/AnomalyDetectors/`, or `src/Clusterers/`
2. Implement the appropriate interface(s) — `Estimator` + `Learner` (or `Online`) at minimum
3. Create PHPUnit test in `tests/` with `#[CoversClass]` attribute
4. For learners: end-to-end test — generate synthetic data, train, validate against minimum score; seed the RNG for determinism
5. Create benchmark in `benchmarks/`
6. Run `composer analyze && composer test && composer check`
7. Add documentation page under `docs/`

### Adding a New Transformer

1. Create class in `src/Transformers/` implementing the `Transformer` interface
2. Create PHPUnit test in `tests/Transformers/`
3. Add documentation page under `docs/transformers/`

### Adding a New Neural Net Component

- Layers → `src/NeuralNet/Layers/`
- Activation functions → `src/NeuralNet/ActivationFunctions/`
- Cost functions → `src/NeuralNet/CostFunctions/`
- Optimizers → `src/NeuralNet/Optimizers/`
- Initializers → `src/NeuralNet/Initializers/`

### Building Documentation

```sh
pip install mike mkdocs mkdocs-material mkdocs-git-revision-date-localized-plugin
mike deploy 'VERSION'
mike serve
```

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on push/PR: phplint → phpstan → phpunit → php-cs-fixer check.
