<?php

namespace Rubix\ML\AnomalyDetection;

use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use MathPHP\Statistics\Average;
use InvalidArgumentException;

class RobustZScore implements Detector, Persistable
{
    const LAMBDA = 0.6745;

    /**
     * The threshold absolute z score to be considered an outlier.
     *
     * @var float
     */
    protected $tolerance;

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
     * @param  float  $tolerance
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $tolerance = 3.5)
    {
        if ($tolerance <= 0) {
            throw new InvalidArgumentException('Tolerance must be greater than'
                . ' 0.');
        }

        $this->tolerance = $tolerance;
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

            $this->mads[$column] = Average::median($deviations);

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
                $score = self::LAMBDA * ($feature - $this->medians[$column])
                    / $this->mads[$column];

                if ($score > $this->tolerance) {
                    $predictions[] = 1;

                    continue 2;
                }
            }

            $predictions[] = 0;
        }

        return $predictions;
    }
}
