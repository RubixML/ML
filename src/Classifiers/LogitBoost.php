<?php

namespace Rubix\ML\Classifiers;

use Tensor\Vector;
use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\CPU;
use Rubix\ML\Probabilistic;
use Rubix\ML\RanksFeatures;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Regressors\ExtraTreeRegressor;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

use function count;
use function is_nan;
use function get_class;
use function in_array;
use function abs;
use function get_object_vars;

/**
 * Logit Boost
 *
 * Logit Boost is a stage-wise additive ensemble that uses regression trees to iteratively learn a logistic regression model
 * for binary classification problems.
 *
 * References:
 * [1] J. H. Friedman et al. (2000). Additive Logistic Regression: A Statistical View of Boosting.
 * [2] J. H. Friedman. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
 * [3] J. H. Friedman. (1999). Stochastic Gradient Boosting.
 * [4] Y. Wei. et al. (2017). Early stopping for kernel boosting algorithms: A general analysis with localized complexities.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LogitBoost implements Estimator, Learner, Probabilistic, RanksFeatures, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * The class names of the learners that can be used as boosters.
     *
     * @var class-string[]
     */
    public const COMPATIBLE_BOOSTERS = [
        RegressionTree::class,
        ExtraTreeRegressor::class,
    ];

    /**
     * The minimum size of each training subset.
     *
     * @var int
     */
    protected const MIN_SUBSAMPLE = 1;

    /**
     * The regressor used to fix up error residuals.
     *
     * @var \Rubix\ML\Learner
     */
    protected \Rubix\ML\Learner $booster;

    /**
     * The learning rate of the ensemble i.e. the *shrinkage* applied to each step.
     *
     * @var float
     */
    protected float $rate;

    /**
     * The ratio of samples to subsample from the training set for each booster.
     *
     * @var float
     */
    protected float $ratio;

    /**
     *  The max number of estimators to train in the ensemble.
     *
     * @var int
     */
    protected int $estimators;

    /**
     * The minimum change in the training loss necessary to continue training.
     *
     * @var float
     */
    protected float $minChange;

    /**
     * The number of epochs without improvement in the validation score to wait before considering an early stop.
     *
     * @var int
     */
    protected int $window;

    /**
     * The proportion of training samples to use for validation and progress monitoring.
     *
     * @var float
     */
    protected float $holdOut;

    /**
     * The metric used to score the generalization performance of the model during training.
     *
     * @var \Rubix\ML\CrossValidation\Metrics\Metric
     */
    protected \Rubix\ML\CrossValidation\Metrics\Metric $metric;

    /**
     * The ensemble of boosters.
     *
     * @var mixed[]|null
     */
    protected ?array $ensemble = null;

    /**
     * The validation scores at each epoch.
     *
     * @var float[]|null
     */
    protected ?array $scores = null;

    /**
     * The average training loss at each epoch.
     *
     * @var float[]|null
     */
    protected ?array $losses = null;

    /**
     * The unique class labels.
     *
     * @var mixed[]|null
     */
    protected ?array $classes = null;

    /**
     * The dimensionality of the training set.
     *
     * @var int|null
     */
    protected ?int $featureCount = null;

    /**
     * The logistic sigmoid function.
     *
     * @internal
     *
     * @param float $value
     * @return float
     */
    public static function sigmoid(float $value) : float
    {
        return 1.0 / (1.0 + exp(-$value));
    }

    /**
     * Weight a sample by its activation signal.
     *
     * @internal
     *
     * @param float $activation
     * @return float
     */
    public static function weightSample(float $activation) : float
    {
        return $activation * (1.0 - $activation);
    }

    /**
     * @param \Rubix\ML\Learner|null $booster
     * @param float $rate
     * @param float $ratio
     * @param int $estimators
     * @param float $minChange
     * @param int $window
     * @param float $holdOut
     * @param \Rubix\ML\CrossValidation\Metrics\Metric|null $metric
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(
        ?Learner $booster = null,
        float $rate = 0.1,
        float $ratio = 0.5,
        int $estimators = 1000,
        float $minChange = 1e-4,
        int $window = 10,
        float $holdOut = 0.1,
        ?Metric $metric = null
    ) {
        if ($booster and !in_array(get_class($booster), self::COMPATIBLE_BOOSTERS)) {
            throw new InvalidArgumentException('Booster is not compatible'
                . ' with the ensemble.');
        }

        if ($rate <= 0.0 or $rate > 1.0) {
            throw new InvalidArgumentException('Learning rate must be'
                . " between 0 and 1, $rate given.");
        }

        if ($ratio <= 0.0 or $ratio > 1.0) {
            throw new InvalidArgumentException('Ratio must be'
                . " between 0 and 1, $ratio given.");
        }

        if ($estimators < 1) {
            throw new InvalidArgumentException('Number of estimators'
                . " must be greater than 0, $estimators given.");
        }

        if ($minChange < 0.0) {
            throw new InvalidArgumentException('Minimum change must be'
                . " greater than 0, $minChange given.");
        }

        if ($window < 1) {
            throw new InvalidArgumentException('Window must be'
                . " greater than 0, $window given.");
        }

        if ($holdOut < 0.0 or $holdOut > 0.5) {
            throw new InvalidArgumentException('Hold out ratio must be'
                . " between 0 and 0.5, $holdOut given.");
        }

        if ($metric) {
            EstimatorIsCompatibleWithMetric::with($this, $metric)->check();
        }

        $this->booster = $booster ?? new RegressionTree(3);
        $this->rate = $rate;
        $this->ratio = $ratio;
        $this->estimators = $estimators;
        $this->minChange = $minChange;
        $this->window = $window;
        $this->holdOut = $holdOut;
        $this->metric = $metric ?? new FBeta();
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::classifier();
    }

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return $this->booster->compatibility();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @internal
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'booster' => $this->booster,
            'rate' => $this->rate,
            'ratio' => $this->ratio,
            'estimators' => $this->estimators,
            'min change' => $this->minChange,
            'window' => $this->window,
            'hold out' => $this->holdOut,
            'metric' => $this->metric,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->ensemble
            and $this->classes
            and $this->featureCount;
    }

    /**
     * Return an iterable progress table with the steps from the last training session.
     *
     * @return \Generator<mixed[]>
     */
    public function steps() : Generator
    {
        if (!$this->losses) {
            return;
        }

        foreach ($this->losses as $epoch => $loss) {
            yield [
                'epoch' => $epoch,
                'score' => $this->scores[$epoch] ?? null,
                'loss' => $loss,
            ];
        }
    }

    /**
     * Return the validation scores at each epoch from the last training session.
     *
     * @return float[]|null
     */
    public function scores() : ?array
    {
        return $this->scores;
    }

    /**
     * Return the loss for each epoch from the last training session.
     *
     * @return float[]|null
     */
    public function losses() : ?array
    {
        return $this->losses;
    }

    /**
     * Train the estimator with a dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new LabelsAreCompatibleWithLearner($dataset, $this),
        ])->check();

        if ($this->logger) {
            $this->logger->info("$this initialized");
        }

        $classes = $dataset->possibleOutcomes();

        if (count($classes) !== 2) {
            throw new InvalidArgumentException('Number of classes'
                . ' must be exactly 2, ' . count($classes) . ' given.');
        }

        [$testing, $training] = $dataset->stratifiedSplit($this->holdOut);

        [$min, $max] = $this->metric->range()->list();

        [$m, $n] = $training->shape();

        $classMap = array_flip($classes);

        $target = [];

        foreach ($training->labels() as $label) {
            $target[] = (float) $classMap[$label];
        }

        $target = Vector::quick($target);

        $z = $prevZ = Vector::zeros($m);
        $activation = Vector::fill(0.5, $m);

        if (!$testing->empty()) {
            $zTest = $prevZTest = Vector::zeros($testing->numSamples());
        }

        $p = max(self::MIN_SUBSAMPLE, (int) round($this->ratio * $m));

        $epsilon = 2.0 * CPU::epsilon();

        $weights = Vector::fill(max($epsilon, 1.0 / $m), $m)->asArray();

        $this->classes = $classes;
        $this->featureCount = $n;

        $this->ensemble = $this->scores = $this->losses = [];

        $bestScore = $min;
        $bestEpoch = $delta = 0;
        $score = null;
        $prevLoss = INF;

        for ($epoch = 1; $epoch <= $this->estimators; ++$epoch) {
            $booster = clone $this->booster;

            $gradient = $target->subtract($activation);

            $training = Labeled::quick($training->samples(), $gradient->asArray());

            $subset = $training->randomWeightedSubsetWithReplacement($p, $weights);

            $booster->train($subset);

            /** @var list<float> $predictions */
            $predictions = $booster->predict($training);

            $z = Vector::quick($predictions)
                ->multiply($this->rate)
                ->add($prevZ);

            $activation = $z->map([self::class, 'sigmoid']);

            $entropy = $activation->clipLower($epsilon)->log();

            $loss = $target->negate()->multiply($entropy)->mean();

            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->info('Numerical instability detected');
                }

                break;
            }

            $this->losses[$epoch] = $loss;

            $this->ensemble[] = $booster;

            if (isset($prevZTest)) {
                /** @var list<float> $predictions */
                $predictions = $booster->predict($testing);

                $zTest = Vector::quick($predictions)
                    ->multiply($this->rate)
                    ->add($prevZTest);

                $activationTest = $zTest->map([self::class, 'sigmoid']);

                $predictions = [];

                foreach ($activationTest as $probability) {
                    $predictions[] = $probability < 0.5 ? $classes[0] : $classes[1];
                }

                $score = $this->metric->score($predictions, $testing->labels());

                $this->scores[$epoch] = $score;
            }

            if ($this->logger) {
                $this->logger->info("Epoch $epoch - {$this->metric}: "
                    . ($score ?? 'n/a') . ", Cross Entropy: $loss");
            }

            if (isset($zTest)) {
                if ($score >= $max) {
                    break;
                }

                if ($score > $bestScore) {
                    $bestScore = $score;
                    $bestEpoch = $epoch;

                    $delta = 0;
                } else {
                    ++$delta;
                }

                if ($delta >= $this->window) {
                    break;
                }

                $prevZTest = $zTest;
            }

            if ($loss <= 0.0) {
                break;
            }

            if (abs($prevLoss - $loss) < $this->minChange) {
                break;
            }

            if ($epoch < $this->estimators) {
                $weights = $activation->map([self::class, 'weightSample'])
                    ->clipLower($epsilon)
                    ->asArray();

                $prevZ = $z;
                $prevLoss = $loss;
            }
        }

        if ($this->scores and end($this->scores) <= $bestScore) {
            if ($this->logger) {
                $this->logger->info("Restoring model state to epoch $bestEpoch");
            }

            $this->ensemble = array_slice($this->ensemble, 0, $bestEpoch);
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return list<string>
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map('Rubix\ML\argmax', $this->proba($dataset));
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return list<float[]>
     */
    public function proba(Dataset $dataset) : array
    {
        if (!$this->ensemble or !$this->classes or !$this->featureCount) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $z = array_fill(0, $dataset->numSamples(), 0.0);

        foreach ($this->ensemble as $estimator) {
            $predictions = $estimator->predict($dataset);

            foreach ($predictions as $j => $prediction) {
                $z[$j] += $this->rate * $prediction;
            }
        }

        [$classA, $classB] = $this->classes;

        $activations = array_map([self::class, 'sigmoid'], $z);

        $probabilities = [];

        foreach ($activations as $activation) {
            $probabilities[] = [
                $classA => 1.0 - $activation,
                $classB => $activation,
            ];
        }

        return $probabilities;
    }

    /**
     * Return the importance scores of each feature column of the training set.
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return float[]
     */
    public function featureImportances() : array
    {
        if (!$this->ensemble or !$this->featureCount) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $importances = array_fill(0, $this->featureCount, 0.0);

        foreach ($this->ensemble as $tree) {
            $importances = $tree->featureImportances();

            foreach ($importances as $column => $importance) {
                $importances[$column] += $importance;
            }
        }

        $numEstimators = count($this->ensemble);

        foreach ($importances as &$importance) {
            $importance /= $numEstimators;
        }

        return $importances;
    }

    /**
     * Return an associative array containing the data used to serialize the object.
     *
     * @return mixed[]
     */
    public function __serialize() : array
    {
        $properties = get_object_vars($this);

        unset($properties['losses'], $properties['scores']);

        return $properties;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Logit Boost (' . Params::stringify($this->params()) . ')';
    }
}
