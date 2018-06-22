<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use MathPHP\Statistics\Average;
use InvalidArgumentException;

class RobustStandardizer implements Transformer
{
    const LAMBDA = 0.6745;

    /**
     * The computed medians of the fitted data indexed by column.
     *
     * @var array
     */
    protected $medians = [
        //
    ];

    /**
     * The computed median absolute deviations of the fitted data indexed by
     * column.
     *
     * @var array
     */
    protected $mads = [
        //
    ];

    /**
     * Return the medians calculated by fitting the training set.
     *
     * @return  array
     */
    public function medians() : array
    {
        return $this->medians;
    }

    /**
     * Return the median absolute deviations calculated during fitting.
     *
     * @return  array
     */
    public function mads() : array
    {
        return $this->mads;
    }

    /**
     * Calculate the medians and median absolute deviations of the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $this->medians = $this->mads = [];

        foreach ($dataset->rotate() as $column => $values) {
            if ($dataset->type($column) === self::CONTINUOUS) {
                $median = Average::median($values);

                $deviations = [];

                foreach ($values as $value) {
                    $deviations[] = abs($value - $median);
                }

                $this->mads[$column] = Average::median($deviations)
                    + self::EPSILON;

                $this->medians[$column] = $median;
            }
        }
    }

    /**
     * Transform the features into a modified z score.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($sample as $column => &$feature) {
                $feature = (self::LAMBDA * ($feature - $this->medians[$column]))
                    / ($this->mads[$column] + self::EPSILON);
            }
        }
    }
}
