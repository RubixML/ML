<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use const Rubix\ML\EPSILON;

/**
 * Robust Z Score
 *
 * A statistical anomaly detector that uses modified Z scores that are robust to
 * preexisting outliers. The modified Z score takes the median and median absolute
 * deviation (MAD) unlike the mean and standard deviation of a *standard* Z score
 * - thus making the statistic more robust to training sets that may already contain
 * outliers. Anomalies are flagged if their maximum feature-specific Z score exceeds
 * some user-defined threshold parameter.
 *
 * References:
 * [1] P. J. Rousseeuw et al. (2017). Anomaly Detection by Robust Statistics.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RobustZScore implements Estimator, Learner, Ranking, Persistable
{
    protected const LAMBDA = 0.6745;

    /**
     * The minimum average z score to be considered an anomaly.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The median of each feature column in the training set.
     *
     * @var float[]|null
     */
    protected $medians;

    /**
     * The median absolute deviation of each feature column.
     *
     * @var float[]|null
     */
    protected $mads;

    /**
     * @param float $threshold
     * @throws \InvalidArgumentException
     */
    public function __construct(float $threshold = 3.5)
    {
        if ($threshold <= 0.) {
            throw new InvalidArgumentException('Threshold must be greater'
                . " than 0, $threshold given.");
        }

        $this->threshold = $threshold;
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return self::ANOMALY_DETECTOR;
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return [
            DataType::CONTINUOUS,
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
     * @return array|null
     */
    public function medians() : ?array
    {
        return $this->medians;
    }

    /**
     * Return the array of computed feature column median absolute deviations.
     *
     * @return array|null
     */
    public function mads() : ?array
    {
        return $this->mads;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        DatasetIsCompatibleWithEstimator::check($dataset, $this);

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
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map([self::class, 'decide'], $this->rank($dataset));
    }

    /**
     * Apply an arbitrary unnormalized scoring function over the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function rank(Dataset $dataset) : array
    {
        if (!$this->medians or !$this->mads) {
            throw new RuntimeException('The estimator has not'
                . ' been trained.');
        }

        DatasetIsCompatibleWithEstimator::check($dataset, $this);

        $scores = [];

        foreach ($dataset as $sample) {
            $zHat = [];

            foreach ($sample as $column => $value) {
                $zHat[] = abs(
                    (self::LAMBDA * ($value - $this->medians[$column]))
                    / $this->mads[$column]
                );
            }

            $scores[] = max($zHat);
        }

        return $scores;
    }

    /**
     * The decision function.
     *
     * @param float $score
     * @return int
     */
    protected function decide(float $score) : int
    {
        return $score > $this->threshold ? 1 : 0;
    }
}
