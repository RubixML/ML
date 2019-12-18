<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\Ranking;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Traits\RankSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use const Rubix\ML\EPSILON;

/**
 * Robust Z Score
 *
 * A statistical anomaly detector that uses modified Z scores that are robust to
 * preexisting outliers. The modified Z score uses the median and median absolute
 * deviation (MAD) unlike the mean and standard deviation of a *standard* Z score
 * which are sensitive to outliers. Anomalies are flagged if their final weighted
 * Z score exceeds a user-defined threshold.
 *
 * > **Note:** An alpha value of 1 means the estimator only considers the maximum
 * absolute z score whereas a setting of 0 indicates that only the average z score
 * factors into the final score.
 *
 * References:
 * [1] B. Iglewicz et al. (1993). How to Detect and Handle Outliers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RobustZScore implements Estimator, Learner, Ranking, Persistable
{
    use PredictsSingle, RankSingle;
    
    protected const ETA = 0.6745;

    /**
     * The minimum z score to be flagged as an anomaly.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The weight of the maximum per sample z score in the overall anomaly
     * score.
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
     * @throws \InvalidArgumentException
     */
    public function __construct(float $threshold = 3.5, float $alpha = 0.5)
    {
        if ($threshold <= 0.) {
            throw new InvalidArgumentException('Threshold must be greater'
                . " than 0, $threshold given.");
        }

        if ($alpha < 0. or $alpha > 1.) {
            throw new InvalidArgumentException('Alpha must be between'
                . " 0 and 1, $alpha given.");
        }

        $this->threshold = $threshold;
        $this->alpha = $alpha;
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
     * @param \Rubix\ML\Datasets\Dataset<array> $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithEstimator::check($dataset, $this);

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
     * @param \Rubix\ML\Datasets\Dataset<array> $dataset
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     * @return string[]
     */
    public function predict(Dataset $dataset) : array
    {
        return array_map([self::class, 'decide'], $this->rank($dataset));
    }

    /**
     * Apply an arbitrary unnormalized scoring function over the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset<array> $dataset
     * @throws \RuntimeException
     * @return float[]
     */
    public function rank(Dataset $dataset) : array
    {
        if (!$this->medians or !$this->mads) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        SamplesAreCompatibleWithEstimator::check($dataset, $this);

        return array_map([self::class, 'z'], $dataset->samples());
    }

    /**
     * Calculate the modified z score for a given sample.
     *
     * @param (int|float)[] $sample
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

        return (1. - $this->alpha) * Stats::mean($z)
            + $this->alpha * max($z);
    }

    /**
     * The decision function.
     *
     * @param float $score
     * @return string
     */
    protected function decide(float $score) : string
    {
        return $score > $this->threshold ? '1' : '0';
    }
}
