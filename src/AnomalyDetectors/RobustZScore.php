<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use MathPHP\Statistics\Average;
use InvalidArgumentException;

/**
 * Robust Z Score
 *
 * A quick global anomaly Detector, Robust Z Score uses a threshold to detect
 * outliers within a Dataset. The modified Z score consists of taking the median
 * and median absolute deviation (MAD) instead of the mean and standard
 * deviation thus making the statistic more robust to training sets that may
 * already contain outliers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RobustZScore implements Detector, Persistable
{
    const LAMBDA = 0.6745;

    /**
     * The threshold z score of a feature to consider the sample an outlier.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The median of each training feature column.
     *
     * @var array
     */
    protected $medians = [
        //
    ];

    /**
     * The median absolute deviation of each training feature column.
     *
     * @var array
     */
    protected $mads = [
        //
    ];

    /**
     * @param  float  $threshold
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $threshold = 3.5)
    {
        if ($threshold < 0) {
            throw new InvalidArgumentException('Z score threshold must be'
                . ' 0 or greater.');
        }

        $this->threshold = $threshold;
    }

    /**
     * Return the array of computed feature column medians.
     *
     * @return array
     */
    public function medians() : array
    {
        return $this->medians;
    }

    /**
     * Return the array of computed feature column median absolute deviations.
     *
     * @return array
     */
    public function mads() : array
    {
        return $this->mads;
    }

    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with'
                . ' continuous features.');
        }

        $this->medians = $this->mads = [];

        foreach ($dataset->rotate() as $column => $features) {
            $median = Average::median($features);

            $deviations = [];

            foreach ($features as $value) {
                $deviations[] = abs($value - $median);
            }

            $this->mads[$column] = Average::median($deviations)
                + self::EPSILON;

            $this->medians[$column] = $median;
        }
    }

    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $predictions = [];

        foreach ($dataset as $sample) {
            foreach ($sample as $column => $feature) {
                $zscore = (self::LAMBDA * ($feature - $this->medians[$column]))
                    / ($this->mads[$column] + self::EPSILON);

                if ($zscore > $this->threshold) {
                    $predictions[] = 1;

                    continue 2;
                }
            }

            $predictions[] = 0;
        }

        return $predictions;
    }
}
