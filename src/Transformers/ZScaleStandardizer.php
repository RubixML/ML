<?php

namespace Rubix\Engine\Transformers;

use MathPHP\Statistics\Average;
use Rubix\Engine\Datasets\Dataset;
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
     * Calculate the means and standard deviations of the dataset.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This transformer only works on continuous features.');
        }

        $this->means = $this->stddevs = [];

        foreach ($dataset->rotate() as $column => $features) {
            $mean = Average::mean($features);

            $stddev = sqrt(array_reduce($features, function ($carry, $feature) use ($mean) {
                return $carry += ($feature - $mean) ** 2;
            }, 0) / count($features));

            $this->means[$column] = $mean;
            $this->stddevs[$column] = $stddev;
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
                $stddev = $this->stddevs[$column];

                $sample[$column] = ($sample[$column] - $mean) / ($stddev ? $stddev : self::EPSILON);
            }
        }
    }
}
