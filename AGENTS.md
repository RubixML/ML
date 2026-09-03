# Rubix ML

High-level machine learning and deep learning library with 40 learning algorithms including ETL and cross-validation for the PHP language.

## Documentation

The project documentation is located in the `/docs` folder in the project root.

## Requirements

- [PHP](https://php.net/manual/en/install.php) 8.3 or above.
- [Tensor extension](https://github.com/RubixML/Tensor) for fast Matrix/Vector computing.
- [Swoole extension](https://openswoole.com/) for multiprocessing support.
- [GD extension](https://php.net/manual/en/book.image.php) for image support.
- [Mbstring extension](https://www.php.net/manual/en/book.mbstring.php) for fast multibyte string manipulation.
- [SVM extension](https://php.net/manual/en/book.svm.php) for Support Vector Machine engine (libsvm).
- [PDO extension](https://www.php.net/manual/en/book.pdo.php) for relational database support.
- [GraphViz](https://graphviz.org/) for graph visualization.

## Key Commands

| Command | Action |
| --- | --- |
| `composer test` | Run PHPUnit tests |
| `composer analyze` | PHPStan static analysis |
| `composer check` | PHP-CS-Fixer dry-run (style check) |
| `composer fix` | PHP code style auto-fixer |
| `composer phplint` | PHP syntax lint |
| `composer benchmark` | PHPBench benchmarks |
| `composer coverage` | Analyze test coverage |
| `composer build` | Full pipeline: install → analyze → test → check |

## Architecture

Namespace `Rubix\ML` autoloaded from `src/`.

```text
src/          →  Rubix\ML\*          (PSR-4)
tests/        →  Rubix\ML\Tests\*    (PHPUnit, mirrors src/)
benchmarks/   →  Rubix\ML\Benchmarks\* (PHPBench, mirrors src/)
docs/         →  MkDocs documentation
```

### Core Interfaces

`Estimator`, `Learner`, `Online`, `Parallel`, `Probabilistic`, `Persistable`, `Verbose`, `RanksFeatures`, `Scoring`

### High-level data types

Rubix ML uses a high-level type system. Strings and integers are considered `categorical` and floats are considered `continuous`.

### Estimator Types

- **Classifiers** (15): AdaBoost, RandomForest, SVC, LogisticRegression, MLP, KNN, NaiveBayes, etc.
- **Regressors** (10): GradientBoost, Ridge, SVR, RegressionTree, Adaline, KNNRegressor, etc.
- **Clusterers** (5): KMeans, DBSCAN, GaussianMixture, MeanShift, FuzzyCMeans
- **Anomaly Detectors** (7): IsolationForest, LOF, OneClassSVM, GaussianMLE, Loda, RobustZScore

## Coding Conventions

- `declare(strict_types=1)` in every file
- PSR-2 with extended rules (enforced by PHP-CS-Fixer, see `.php-cs-fixer.dist.php`)
- DocBlock on every class, property, method, constant, and function
- No anonymous classes or functions (breaks serialization/persistence)
- Objects are *generally* immutable — state mutation only through a well-defined public API
- Domain-driven naming — names reflect the ML domain
- No inline comments — use expressive syntax and abstractions instead
- Single quotes for strings, short array syntax (`[]`)
- Prefer pre-increment (`++$i`) over post-increment where possible
- No superfluous `else`/`return` constructs
- Bugfixes should include a test that reproduces the bug before the fix
- Optimizations should include a before and after benchmark
- Include documentation updates for changes that effect the public API
- Class members annotated `internal` are not part of the public API
- Verify changes by running tests, static analysis, and code style fixer

## Workflows

### Adding a New Estimator

1. Create class in `src/Classifiers/`, `src/Regressors/`, `src/AnomalyDetectors/`, or `src/Clusterers/`
2. Implement the appropriate interface(s) — `Estimator` + `Learner` (or `Online`) at minimum
3. Create PHPUnit test in `tests/` with `#[CoversClass]` attribute
4. For learners: end-to-end test — generate synthetic data, train, validate against minimum score; seed the RNG for determinism
5. Create benchmark in `benchmarks/`
6. Run `composer analyze && composer test && composer fix`
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

GitHub Actions (`.github/workflows/ci.yml`) runs on push/PR: phpstan → phpunit → php-cs-fixer check.
