<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\RanksFeatures;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\CrossValidation\Metrics\RMSE;
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
use function array_map;
use function array_reduce;
use function array_slice;
use function array_fill;
use function in_array;
use function round;
use function max;
use function abs;
use function get_object_vars;

/**
 * Gradient Boost
 *
 * Gradient Boost is a stage-wise additive ensemble that uses a Gradient Descent boosting
 * scheme for training  boosters (Decision Trees) to correct the error residuals of a
 * series of *weak* base learners. Stochastic gradient boosting is achieved by varying
 * the ratio of samples to subsample uniformly at random from the training set.
 *
 * References:
 * [1] J. H. Friedman. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
 * [2] J. H. Friedman. (1999). Stochastic Gradient Boosting.
 * [3] Y. Wei. et al. (2017). Early stopping for kernel boosting algorithms: A general analysis
 * with localized complexities.
 * [4] G. Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GradientBoost implements Estimator, Learner, RanksFeatures, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * The class names of the compatible learners to used as boosters.
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
     * The regressor that will fix up the error residuals of the *weak* base learner.
     *
     * @var Learner
     */
    protected Learner $booster;

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
     * The number of epochs without improvement in the validation score to wait before considering an
     * early stop.
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
     * @var Metric
     */
    protected Metric $metric;

    /**
     * An ensemble of weak regressors.
     *
     * @var mixed[]
     */
    protected array $ensemble = [
        //
    ];

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
     * The dimensionality of the training set.
     *
     * @var int<0,max>|null
     */
    protected ?int $featureCount = null;

    /**
     * The mean of the labels of the training set.
     *
     * @var float|null
     */
    protected ?float $mu = null;

    /**
     * @param Learner|null $booster
     * @param float $rate
     * @param float $ratio
     * @param int $epochs
     * @param float $minChange
     * @param int $window
     * @param float $holdOut
     * @param Metric|null $metric
     * @throws InvalidArgumentException
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
        $this->metric = $metric ?? new RMSE();
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::regressor();
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
        return !empty($this->ensemble);
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
     * @param Labeled $dataset
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
            $this->logger->info("Training $this");
        }

        [$testing, $training] = $dataset->randomize()->split($this->holdOut);

        [$minScore, $maxScore] = $this->metric->range()->list();

        [$m, $n] = $training->shape();

        $targets = $training->labels();

        $mu = Stats::mean($targets);

        $out = array_fill(0, $m, $mu);

        if (!$testing->empty()) {
            $outTest = array_fill(0, $testing->numSamples(), $mu);
        } elseif ($this->logger) {
            $this->logger->notice('Insufficient validation data, '
                . 'some features are disabled');
        }

        $p = max(self::MIN_SUBSAMPLE, (int) round($this->ratio * $m));

        $weights = array_fill(0, $m, 1.0 / $m);

        $this->featureCount = $n;
        $this->ensemble = $this->scores = $this->losses = [];
        $this->mu = $mu;

        $bestScore = $minScore;
        $bestEpoch = $numWorseEpochs = 0;
        $score = null;
        $prevLoss = INF;

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $gradient = array_map([$this, 'gradient'], $out, $targets);
            $loss = array_reduce($gradient, [$this, 'l2Loss'], 0.0);

            $loss /= $m;

            $lossChange = abs($prevLoss - $loss);

            $this->losses[$epoch] = $loss;

            if (isset($outTest)) {
                $score = $this->metric->score($outTest, $testing->labels());

                $this->scores[$epoch] = $score;
            }

            if ($this->logger) {
                $lossDirection = $loss < $prevLoss ? '↓' : '↑';

                $message = "Epoch: $epoch, "
                    . "L2 Loss: $loss, "
                    . "Loss Change: {$lossDirection}{$lossChange}, "
                    . "{$this->metric}: " . ($score ?? 'N/A');

                $this->logger->info($message);
            }

            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->warning('Numerical instability detected');
                }

                break;
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

            $this->ensemble[] = $booster;

            $predictions = $booster->predict($training);

            $out = array_map([$this, 'updateOut'], $predictions, $out);

            if (isset($outTest)) {
                $predictions = $booster->predict($testing);

                $outTest = array_map([$this, 'updateOut'], $predictions, $outTest);
            }

            $weights = array_map('abs', $gradient);

            $prevLoss = $loss;
        }

        if ($this->scores and end($this->scores) <= $bestScore) {
            $this->ensemble = array_slice($this->ensemble, 0, $bestEpoch);

            if ($this->logger) {
                $this->logger->info("Model state restored to epoch $bestEpoch");
            }
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make a prediction from a dataset.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<int|float>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!isset($this->ensemble, $this->featureCount, $this->mu)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $out = array_fill(0, $dataset->numSamples(), $this->mu);

        foreach ($this->ensemble as $estimator) {
            $predictions = $estimator->predict($dataset);

            $out = array_map([$this, 'updateOut'], $predictions, $out);
        }

        return $out;
    }

    /**
     * Return the importance scores of each feature column of the training set.
     *
     * @throws RuntimeException
     * @return float[]
     */
    public function featureImportances() : array
    {
        if (!isset($this->ensemble, $this->featureCount)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $importances = array_fill(0, $this->featureCount, 0.0);

        foreach ($this->ensemble as $tree) {
            $scores = $tree->featureImportances();

            foreach ($scores as $column => $score) {
                $importances[$column] += $score;
            }
        }

        $numEstimators = count($this->ensemble);

        foreach ($importances as &$importance) {
            $importance /= $numEstimators;
        }

        return $importances;
    }

    /**
     * Compute the output for an iteration.
     *
     * @param float $prediction
     * @param float $out
     * @return float
     */
    protected function updateOut(float $prediction, float $out) : float
    {
        return $this->rate * $prediction + $out;
    }

    /**
     * Compute the gradient for a single sample.
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
     * Compute the cross entropy loss function.
     *
     * @param float $loss
     * @param float $derivative
     * @return float
     */
    protected function l2Loss(float $loss, float $derivative) : float
    {
        return $loss + $derivative ** 2;
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
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Gradient Boost (' . Params::stringify($this->params()) . ')';
    }
}
