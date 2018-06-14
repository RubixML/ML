<?php

namespace Rubix\ML\Transformers;

use MathPHP\Statistics\Average;
use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

class ZScaleStandardizer implements Transformer
{
    /**
     * The computed means of the fitted data indexed by column.
     *
     * @var array
     */
    protected $means = [
        //
    ];

    /**
     * The computed standard deviations of the fitted data indexed by column.
     *
     * @var array
     */
    protected $stddevs = [
        //
    ];

    /**
     * Return the means calculated by fitting the training set.
     *
     * @return  array
     */
    public function means() : array
    {
        return $this->means;
    }

    /**
     * Return the standard deviations calculated during fitting.
     *
     * @return  array
     */
    public function stddevs() : array
    {
        return $this->stddevs;
    }

    /**
     * Calculate the means and standard deviations of the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $this->means = $this->stddevs = [];

        foreach ($dataset->columnTypes() as $column => $type) {
            if ($type === self::CONTINUOUS) {
                $values = $dataset->column($column);

                $mean = Average::mean($values);

                $stddev = sqrt(array_reduce($values,
                    function ($carry, $feature) use ($mean) {
                        return $carry += ($feature - $mean) ** 2;
                    }, 0) / count($values));

                $this->means[$column] = $mean;
                $this->stddevs[$column] = $stddev;
            }
        }
    }

    /**
     * Transform the features into a z score.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($this->means as $column => $mean) {
                $sample[$column] = ($sample[$column] - $mean)
                    / ($this->stddevs[$column] + self::EPSILON);
            }
        }
    }
}
