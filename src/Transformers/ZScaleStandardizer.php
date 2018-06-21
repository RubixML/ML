<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use MathPHP\Statistics\Average;
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

        foreach ($dataset->rotate() as $column => $values) {
            if ($dataset->type($column) === self::CONTINUOUS) {
                $mean = Average::mean($values);

                $deviations = [];

                foreach ($values as $value) {
                    $deviations[] = ($value - $mean) ** 2;
                }

                $this->stddevs[$column] = sqrt(Average::mean($deviations));

                $this->means[$column] = $mean;
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
