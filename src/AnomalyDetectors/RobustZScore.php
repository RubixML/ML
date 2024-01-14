<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Helpers\CPU;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function count;
use function abs;
use function max;
use function array_map;

/**
 * Robust Z-Score
 *
 * A statistical anomaly detector that uses modified Z-Scores which are robust to preexisting
 * outliers in the training set. The modified Z-Score uses the median and median absolute
 * deviation (MAD) unlike the mean and standard deviation of a standard Z-Score - which are
 * more sensitive to outliers. Anomalies are flagged if their final weighted Z-Score exceeds a
 * user-defined threshold.
 *
 * > **Note:** A beta value of 1 means the estimator only considers the maximum absolute Z-Score,
 * whereas a setting of 0 indicates that only the average Z-Score factors into the final score.
 *
 * References:
 * [1] B. Iglewicz et al. (1993). How to Detect and Handle Outliers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RobustZScore implements Estimator, Learner, Scoring, Persistable
{
    use AutotrackRevisions;

    /**
     * The expected value of the MAD as n asymptotes.
     *
     * @var float
     */
    protected const ETA = 0.6745;

    /**
     * The minimum z score to be flagged as an anomaly.
     *
     * @var float
     */
    protected float $threshold;

    /**
     * The weight of the maximum per sample z score in the overall anomaly score.
     *
     * @var float
     */
    protected float $beta;

    /**
     * The amount of epsilon smoothing added to the median absolute deviation (MAD) of each feature.
     *
     * @var float
     */
    protected float $smoothing;

    /**
     * The median of each feature column in the training set.
     *
     * @var float[]
     */
    protected array $medians = [
        //
    ];

    /**
     * The median absolute deviation of each feature column.
     *
     * @var float[]
     */
    protected array $mads = [
        //
    ];

    /**
     * @param float $threshold
     * @param float $beta
     * @param float $smoothing
     * @throws InvalidArgumentException
     */
    public function __construct(float $threshold = 3.5, float $beta = 0.5, float $smoothing = 1e-9)
    {
        if ($threshold <= 0.0) {
            throw new InvalidArgumentException('Threshold must be'
                . " greater than 0, $threshold given.");
        }

        if ($beta < 0.0 or $beta > 1.0) {
            throw new InvalidArgumentException('Beta must be'
                . " between 0 and 1, $beta given.");
        }

        if ($smoothing <= 0.0) {
            throw new InvalidArgumentException('Smoothing must be'
                . " greater than 0, $smoothing given.");
        }

        $this->threshold = $threshold;
        $this->beta = $beta;
        $this->smoothing = $smoothing;
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
            'beta' => $this->beta,
            'smoothing' => $this->smoothing,
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
     * @param Dataset $dataset
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
        ])->check();

        $this->medians = $this->mads = [];

        foreach ($dataset->features() as $column => $values) {
            [$median, $mad] = Stats::medianMad($values);

            $this->medians[$column] = $median;
            $this->mads[$column] = $mad;
        }

        $epsilon = max($this->smoothing * max($this->mads), CPU::epsilon());

        foreach ($this->mads as &$mad) {
            $mad += $epsilon;
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
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
        return $this->zHat($sample) > $this->threshold ? 1 : 0;
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<float>
     */
    public function score(Dataset $dataset) : array
    {
        if (!$this->medians or !$this->mads) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count($this->medians))->check();

        return array_map([$this, 'zHat'], $dataset->samples());
    }

    /**
     * Calculate the modified z score for a given sample.
     *
     * @param list<int|float> $sample
     * @return float
     */
    protected function zHat(array $sample) : float
    {
        $scores = [];

        foreach ($sample as $column => $value) {
            $scores[] = abs(
                (self::ETA
                * ($value - $this->medians[$column]))
                / $this->mads[$column]
            );
        }

        $zHat = (1.0 - $this->beta) * Stats::mean($scores)
            + $this->beta * max($scores);

        return $zHat;
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
        return 'Robust Z Score (' . Params::stringify($this->params()) . ')';
    }
}
