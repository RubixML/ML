<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\Ranking;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Traits\RanksSingle;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\warn_deprecated;

use const Rubix\ML\EPSILON;

/**
 * Robust Z-Score
 *
 * A statistical anomaly detector that uses modified Z-Scores which are robust to preexisting
 * outliers in the training set. The modified Z-Score uses the median and median absolute
 * deviation (MAD) unlike the mean and standard deviation of a standard Z-Score - which are
 * more sensitive to outliers. Anomalies are flagged if their final weighted Z-Score exceeds a
 * user-defined threshold.
 *
 * > **Note:** An alpha value of 1 means the estimator only considers the maximum absolute Z-Score,
 * whereas a setting of 0 indicates that only the average Z-Score factors into the final score.
 *
 * References:
 * [1] B. Iglewicz et al. (1993). How to Detect and Handle Outliers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RobustZScore implements Estimator, Learner, Scoring, Ranking, Persistable
{
    use AutotrackRevisions, RanksSingle;

    /**
     * The expected value of the MAD as n goes to ∞.
     *
     * @var float
     */
    protected const ETA = 0.6745;

    /**
     * The minimum z score to be flagged as an anomaly.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The weight of the maximum per sample z score in the overall anomaly score.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The median of each feature column in the training set.
     *
     * @var float[]
     */
    protected $medians = [
        //
    ];

    /**
     * The median absolute deviation of each feature column.
     *
     * @var float[]
     */
    protected $mads = [
        //
    ];

    /**
     * @param float $threshold
     * @param float $alpha
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $threshold = 3.5, float $alpha = 0.5)
    {
        if ($threshold <= 0.0) {
            throw new InvalidArgumentException('Threshold must be'
                . " greater than 0, $threshold given.");
        }

        if ($alpha < 0.0 or $alpha > 1.0) {
            throw new InvalidArgumentException('Alpha must be'
                . " between 0 and 1, $alpha given.");
        }

        $this->threshold = $threshold;
        $this->alpha = $alpha;
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
        return EstimatorType::anomalyDetector();
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
        return [
            DataType::continuous(),
        ];
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
            'threshold' => $this->threshold,
            'alpha' => $this->alpha,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->medians and $this->mads;
    }

    /**
     * Return the array of computed feature column medians.
     *
     * @return float[]|null
     */
    public function medians() : ?array
    {
        return $this->medians;
    }

    /**
     * Return the array of computed feature column median absolute deviations.
     *
     * @return float[]|null
     */
    public function mads() : ?array
    {
        return $this->mads;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
        ])->check();

        $this->medians = $this->mads = [];

        foreach ($dataset->columns() as $column => $values) {
            [$median, $mad] = Stats::medianMad($values);

            $this->medians[$column] = $median;
            $this->mads[$column] = $mad ?: EPSILON;
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return list<int>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->medians or !$this->mads) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        return array_map([$this, 'predictSample'], $dataset->samples());
    }

    /**
     * Predict a single sample and return the result.
     *
     * @internal
     *
     * @param list<int|float> $sample
     * @return int
     */
    public function predictSample(array $sample) : int
    {
        return $this->z($sample) > $this->threshold ? 1 : 0;
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return list<float>
     */
    public function score(Dataset $dataset) : array
    {
        if (!$this->medians or !$this->mads) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count($this->medians))->check();

        return array_map([$this, 'z'], $dataset->samples());
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @deprecated
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return list<float>
     */
    public function rank(Dataset $dataset) : array
    {
        warn_deprecated('Rank() is deprecated, use score() instead.');

        return $this->score($dataset);
    }

    /**
     * Calculate the modified z score for a given sample.
     *
     * @param list<int|float> $sample
     * @return float
     */
    protected function z(array $sample) : float
    {
        $z = [];

        foreach ($sample as $column => $value) {
            $z[] = abs(
                (self::ETA * ($value - $this->medians[$column]))
                / $this->mads[$column]
            );
        }

        return (1.0 - $this->alpha) * Stats::mean($z)
            + $this->alpha * max($z);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Robust Z Score (' . Params::stringify($this->params()) . ')';
    }
}
