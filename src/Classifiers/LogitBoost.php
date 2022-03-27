<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\RanksFeatures;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Regressors\RegressionTree;
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
use function array_map;
use function array_slice;
use function array_fill;
use function array_flip;
use function round;
use function max;
use function abs;
use function log;
use function get_object_vars;

/**
 * Logit Boost
 *
 * A stage-wise additive ensemble that uses regression trees to iteratively learn a Logistic Regression model for binary
 * classification problems. Unlike standard Logistic Regression, Logit Boost has the ability to learn a smooth non-linear
 * decision surface by training decision trees to follow the gradient of the cross entropy loss function. In addition,
 * Logit Boost concentrates more effort on classifying samples that it is less certain about.
 *
 * References:
 * [1] J. H. Friedman et al. (2000). Additive Logistic Regression: A Statistical View of Boosting.
 * [2] J. H. Friedman. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
 * [3] J. H. Friedman. (1999). Stochastic Gradient Boosting.
 * [4] Y. Wei. et al. (2017). Early stopping for kernel boosting algorithms: A general analysis with localized complexities.
 * [5] G. Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
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
    protected const MIN_SUBSAMPLE = 2;

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
     * The ratio of samples to subsample from the training set for each booster.d
     *
     * @var float
     */
    protected float $ratio;

    /**
     * The maximum number of training epochs. i.e. the number of times to iterate before terminating.
     *
     * @var int<0,max>
     */
    protected int $epochs;

    /**
     * The minimum change in the training loss necessary to continue training.
     *
     * @var float
     */
    protected float $minChange;

    /**
     * The number of epochs without improvement in the validation score to wait before considering an early stop.
     *
     * @var positive-int
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
    protected ?array $boosters = null;

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
     * @var list<string>|null
     */
    protected ?array $classes = null;

    /**
     * The dimensionality of the training set.
     *
     * @var int<0,max>|null
     */
    protected ?int $featureCount = null;

    /**
     * @param \Rubix\ML\Learner|null $booster
     * @param float $rate
     * @param float $ratio
     * @param int $epochs
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
        int $epochs = 1000,
        float $minChange = 1e-4,
        int $window = 5,
        float $holdOut = 0.1,
        ?Metric $metric = null
    ) {
        if ($booster and !in_array(get_class($booster), self::COMPATIBLE_BOOSTERS)) {
            throw new InvalidArgumentException('Booster is not compatible'
                . ' with the ensemble.');
        }

        if ($rate <= 0.0) {
            throw new InvalidArgumentException('Learning rate must be'
                . " greater than 0, $rate given.");
        }

        if ($ratio <= 0.0 or $ratio > 1.0) {
            throw new InvalidArgumentException('Ratio must be'
                . " between 0 and 1, $ratio given.");
        }

        if ($epochs < 0) {
            throw new InvalidArgumentException('Number of epochs'
                . " must be greater than 0, $epochs given.");
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
        $this->epochs = $epochs;
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
            'epochs' => $this->epochs,
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
        return $this->boosters
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

        /** @var list<string> $classes */
        $classes = $dataset->possibleOutcomes();

        if (count($classes) !== 2) {
            throw new InvalidArgumentException('Number of classes'
                . ' must be exactly 2, ' . count($classes) . ' given.');
        }

        if ($this->logger) {
            $this->logger->info("Training $this");
        }

        [$testing, $training] = $dataset->stratifiedSplit($this->holdOut);

        [$minScore, $maxScore] = $this->metric->range()->list();

        [$m, $n] = $training->shape();

        $classMap = array_flip($classes);

        $targets = [];

        foreach ($training->labels() as $label) {
            $targets[] = (float) $classMap[$label];
        }

        $z = array_fill(0, $m, 0.0);
        $out = array_fill(0, $m, 0.5);

        if (!$testing->empty()) {
            $zTest = array_fill(0, $testing->numSamples(), 0.0);
        } elseif ($this->logger) {
            $this->logger->notice('Insufficient validation data, '
                . 'some features are disabled');
        }

        $p = max(self::MIN_SUBSAMPLE, (int) round($this->ratio * $m));

        $weights = array_fill(0, $m, 1.0 / $m);

        $this->classes = $classes;
        $this->featureCount = $n;
        $this->boosters = $this->scores = $this->losses = [];

        $bestScore = $minScore;
        $bestEpoch = $numWorseEpochs = 0;
        $score = null;
        $prevLoss = INF;

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $gradient = array_map([$this, 'gradient'], $out, $targets);
            $losses = array_map([$this, 'crossEntropy'], $out, $targets);

            $loss = Stats::mean($losses);

            $lossChange = abs($prevLoss - $loss);

            $this->losses[$epoch] = $loss;

            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->warning('Numerical instability detected');
                }

                break;
            }

            if (isset($zTest)) {
                $predictions = [];

                foreach ($zTest as $value) {
                    $predictions[] = $value < 0.0 ? $classes[0] : $classes[1];
                }

                $score = $this->metric->score($predictions, $testing->labels());

                $this->scores[$epoch] = $score;
            }

            if ($this->logger) {
                $lossDirection = $loss < $prevLoss ? '↓' : '↑';

                $message = "Epoch: $epoch, "
                    . "Cross Entropy: $loss, "
                    . "Loss Change: {$lossDirection}{$lossChange}, "
                    . "{$this->metric}: " . ($score ?? 'N/A');

                $this->logger->info($message);
            }

            if (isset($score)) {
                if ($score >= $maxScore) {
                    break;
                }

                if ($score > $bestScore) {
                    $bestScore = $score;
                    $bestEpoch = $epoch;

                    $numWorseEpochs = 0;
                } else {
                    ++$numWorseEpochs;
                }

                if ($numWorseEpochs >= $this->window) {
                    break;
                }
            }

            if ($lossChange < $this->minChange) {
                break;
            }

            $training = Labeled::quick($training->samples(), $gradient);

            $subset = $training->randomWeightedSubsetWithReplacement($p, $weights);

            $booster = clone $this->booster;

            $booster->train($subset);

            $this->boosters[] = $booster;

            $predictions = $booster->predict($training);

            $z = array_map([$this, 'updateZ'], $predictions, $z);
            $out = array_map('Rubix\ML\sigmoid', $z);

            if (isset($zTest)) {
                $predictions = $booster->predict($testing);

                $zTest = array_map([$this, 'updateZ'], $predictions, $zTest);
            }

            $weights = array_map('abs', $gradient);

            $prevLoss = $loss;
        }

        if ($this->scores and end($this->scores) < $bestScore) {
            $this->boosters = array_slice($this->boosters, 0, $bestEpoch);

            if ($this->logger) {
                $this->logger->info("Model state restored to epoch $bestEpoch");
            }
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return list<string>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!isset($this->boosters, $this->classes, $this->featureCount)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $z = array_fill(0, $dataset->numSamples(), 0.0);

        foreach ($this->boosters as $estimator) {
            $zHat = $estimator->predict($dataset);

            $z = array_map([$this, 'updateZ'], $zHat, $z);
        }

        [$classA, $classB] = $this->classes;

        $predictions = [];

        foreach ($z as $value) {
            $predictions[] = $value < 0.0 ? $classA : $classB;
        }

        return $predictions;
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return list<array<string,float>>
     */
    public function proba(Dataset $dataset) : array
    {
        if (!isset($this->boosters, $this->classes, $this->featureCount)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $z = array_fill(0, $dataset->numSamples(), 0.0);

        foreach ($this->boosters as $estimator) {
            $zHat = $estimator->predict($dataset);

            $z = array_map([$this, 'updateZ'], $zHat, $z);
        }

        $out = array_map('Rubix\ML\sigmoid', $z);

        [$classA, $classB] = $this->classes;

        $probabilities = [];

        foreach ($out as $probability) {
            $probabilities[] = [
                $classA => 1.0 - $probability,
                $classB => $probability,
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
        if (!isset($this->boosters, $this->featureCount)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $importances = array_fill(0, $this->featureCount, 0.0);

        foreach ($this->boosters as $tree) {
            $scores = $tree->featureImportances();

            foreach ($scores as $column => $score) {
                $importances[$column] += $score;
            }
        }

        $numEstimators = count($this->boosters);

        foreach ($importances as &$importance) {
            $importance /= $numEstimators;
        }

        return $importances;
    }

    /**
     * Compute the gradient.
     *
     * @param float $out
     * @param float $target
     * @return float
     */
    protected function gradient(float $out, float $target) : float
    {
        return $target - $out;
    }

    /**
     * Compute the binary cross entropy loss function.
     *
     * @param float $out
     * @param float $target
     * @return float
     */
    protected function crossEntropy(float $out, float $target) : float
    {
        return $target >= 0.5 ? -log($out) : -log(1.0 - $out);
    }

    /**
     * Compute the z signal for an iteration.
     *
     * @param float $z
     * @param float $prevZ
     * @return float
     */
    protected function updateZ(float $z, float $prevZ) : float
    {
        return $this->rate * $z + $prevZ;
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
