<?php

namespace Rubix\Engine\Preprocessors;

use Rubix\Engine\Dataset;
use MathPHP\Statistics\Average;

class ZScaleNormalizer implements Preprocessor
{
    /**
     * The computed means and standard deviations of the fitted data indexed by column.
     *
     * @var array
     */
    protected $stats = [
        //
    ];

    /**
     * @return array
     */
    public function stats() : array
    {
        return $this->stats;
    }

    /**
     * Calculate the means and standard deviations of training set.
     *
     * @param  \Rubix\Engine\Dataset  $data
     * @return void
     */
    public function fit(Dataset $data) : void
    {
        $this->stats = [];

        foreach ($data->columnTypes() as $column => $type) {
            if ($type === self::CONTINUOUS) {
                $features = array_column($data->samples(), $column);

                $mean = Average::mean($features);

                $stddev = sqrt(array_reduce($features, function ($carry, $feature) use ($mean) {
                    return $carry += ($feature - $mean) ** 2;
                }, 0) / count($features));

                $this->stats[$column] = [$mean, $stddev];
            }
        }
    }

    /**
     * Transform the continuous values into a z score between 0 and 1.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($this->stats as $column => $stats) {
                $sample[$column] = ($sample[$column] - $stats[0]) / ($stats[1] + self::EPSILON);
            }
        }
    }
}
