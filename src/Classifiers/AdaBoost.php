<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

use function count;
use function is_nan;
use function array_fill;
use function array_fill_keys;
use function array_sum;
use function get_object_vars;
use function round;
use function max;
use function abs;
use function log;
use function exp;

use const Rubix\ML\EPSILON;

/**
 * AdaBoost
 *
 * Short for *Adaptive Boosting*, this ensemble classifier can improve the performance
 * of an otherwise *weak* classifier by focusing more attention on samples that are
 * harder to classify. It builds an additive model where, at each stage, a new learner
 * is instantiated and trained.
 *
 * > **Note**: The default base classifier is a *Decision Stump* i.e a
 * Classification Tree with a max height of 1.
 *
 * References:
 * [1] Y. Freund et al. (1996). A Decision-theoretic Generalization of On-line
 * Learning and an Application to Boosting.
 * [2] J. Zhu et al. (2006). Multi-class AdaBoost.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class AdaBoost implements Estimator, Learner, Probabilistic, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * The minimum size of each training subset.
     *
     * @var int
     */
    protected const MIN_SUBSAMPLE = 2;

    /**
     * The base classifier to be boosted.
     *
     * @var Learner
     */
    protected Learner $base;

    /**
     * The learning rate of the ensemble i.e. the *shrinkage* applied to each step.
     *
     * @var float
     */
    protected float $rate;

    /**
     * The ratio of samples to train each weak learner on.
     *
     * @var float
     */
    protected float $ratio;

    /**
     * The maximum number of estimators to train in the ensemble.
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
     * The number of epochs without improvement in the training loss to wait before considering an early stop.
     *
     * @var positive-int
     */
    protected int $window;

    /**
     * The ensemble of *weak* classifiers.
     *
     * @var \Rubix\ML\Learner[]|null
     */
    protected ?array $ensemble = null;

    /**
     * The amount of influence a particular classifier has in the model.
     *
     * @var list<float>|null
     */
    protected ?array $influences = null;

    /**
     * The zero vector for the possible class outcomes.
     *
     * @var array<string,float>|null
     */
    protected ?array $classes = null;

    /**
     * The loss at each epoch from the last training session.
     *
     * @var list<float>]|null
     */
    protected ?array $losses = null;

    /**
     * The dimensionality of the training set.
     *
     * @var int<0,max>|null
     */
    protected ?int $featureCount = null;

    /**
     * @param Learner|null $base
     * @param float $rate
     * @param float $ratio
     * @param int $epochs
     * @param float $minChange
     * @param int $window
     * @throws InvalidArgumentException
     */
    public function __construct(
        ?Learner $base = null,
        float $rate = 1.0,
        float $ratio = 0.8,
        int $epochs = 100,
        float $minChange = 1e-4,
        int $window = 5
    ) {
        if ($base and !$base->type()->isClassifier()) {
            throw new InvalidArgumentException('Base Estimator must be'
                . " a classifier, {$base->type()} given.");
        }

        if ($rate < 0.0) {
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

        $this->base = $base ?? new ClassificationTree(1);
        $this->rate = $rate;
        $this->ratio = $ratio;
        $this->epochs = $epochs;
        $this->minChange = $minChange;
        $this->window = $window;
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
        return $this->base->compatibility();
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
            'base' => $this->base,
            'rate' => $this->rate,
            'ratio' => $this->ratio,
            'epochs' => $this->epochs,
            'min change' => $this->minChange,
            'window' => $this->window,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return isset($this->ensemble, $this->influences);
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
                'loss' => $loss,
            ];
        }
    }

    /**
     * Return the loss at each epoch of the last training session.
     *
     * @return float[]|null
     */
    public function losses() : ?array
    {
        return $this->losses;
    }

    /**
     * Train the learner with a dataset.
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

        [$m, $n] = $dataset->shape();

        $labels = $dataset->labels();

        $k = count($classes);
        $p = max(self::MIN_SUBSAMPLE, (int) round($this->ratio * $m));

        $weights = array_fill(0, $m, 1.0 / $m);

        $this->classes = array_fill_keys($classes, 0.0);
        $this->featureCount = $n;

        $this->ensemble = $this->influences = $this->losses = [];

        $prevLoss = $bestLoss = INF;
        $lossThreshold = 1.0 - (1.0 / $k);
        $numWorseEpochs = 0;

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $estimator = clone $this->base;

            $subset = $dataset->randomWeightedSubsetWithReplacement($p, $weights);

            $estimator->train($subset);

            $predictions = $estimator->predict($dataset);

            $loss = 0.0;

            foreach ($predictions as $i => $prediction) {
                if ($prediction != $labels[$i]) {
                    $loss += $weights[$i];
                }
            }

            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->warning('Numerical instability detected');
                }

                break;
            }

            $totalWeight = array_sum($weights) ?: EPSILON;

            $loss /= $totalWeight;

            $lossChange = abs($prevLoss - $loss);

            $this->losses[$epoch] = $loss;

            if ($this->logger) {
                $lossDirection = $loss < $prevLoss ? '↓' : '↑';

                $message = "Epoch: $epoch, "
                    . "Exponential Loss: $loss, "
                    . "Loss Change: {$lossDirection}{$lossChange}";

                $this->logger->info($message);
            }

            if ($loss > $lossThreshold) {
                if ($this->logger) {
                    $this->logger->notice('Learner dropped due to high training loss');
                }

                continue;
            }

            $influence = $this->rate
                * (log((1.0 - $loss) / ($loss ?: EPSILON))
                + log($k - 1));

            $this->ensemble[] = $estimator;
            $this->influences[] = $influence;

            if ($lossChange < $this->minChange) {
                break;
            }

            if ($loss > $bestLoss) {
                $bestLoss = $loss;

                $numWorseEpochs = 0;
            } else {
                ++$numWorseEpochs;
            }

            if ($numWorseEpochs >= $this->window) {
                break;
            }

            if ($epoch < $this->epochs) {
                $step = exp($influence);

                foreach ($predictions as $i => $prediction) {
                    if ($prediction != $labels[$i]) {
                        $weights[$i] *= $step;
                    }
                }

                $total = array_sum($weights) ?: EPSILON;

                foreach ($weights as &$weight) {
                    $weight /= $total;
                }
            }

            $prevLoss = $loss;
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param Dataset $dataset
     * @return list<int|string>
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map('Rubix\ML\argmax', $this->score($dataset));
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param Dataset $dataset
     * @return list<array<string,float>>
     */
    public function proba(Dataset $dataset) : array
    {
        $scores = $this->score($dataset);

        $probabilities = [];

        foreach ($scores as $influences) {
            $total = array_sum($influences) ?: EPSILON;

            $dist = [];

            foreach ($influences as $class => $influence) {
                $dist[$class] = $influence / $total;
            }

            $probabilities[] = $dist;
        }

        return $probabilities;
    }

    /**
     * Return the influence scores for each sample in the dataset.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<float[]>
     */
    protected function score(Dataset $dataset) : array
    {
        if (!isset($this->ensemble, $this->influences, $this->classes, $this->featureCount)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $scores = array_fill(0, $dataset->numSamples(), $this->classes);

        foreach ($this->ensemble as $i => $estimator) {
            $predictions = $estimator->predict($dataset);

            $influence = $this->influences[$i];

            foreach ($predictions as $j => $prediction) {
                $scores[$j][$prediction] += $influence;
            }
        }

        return $scores;
    }

    /**
     * Return an associative array containing the data used to serialize the object.
     *
     * @return mixed[]
     */
    public function __serialize() : array
    {
        $properties = get_object_vars($this);

        unset($properties['losses']);

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
        return 'AdaBoost (' . Params::stringify($this->params()) . ')';
    }
}
