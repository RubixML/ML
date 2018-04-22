<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Dataset;
use MathPHP\Statistics\Average;

class ZScaleStandardizer implements Transformer
{
    /**
     * The type of each feature column. i.e. categorical or continuous.
     *
     * @var array
     */
    protected $columnTypes = [
        //
    ];

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
     * @return array
     */
    public function stats() : array
    {
        return $this->stats;
    }

    /**
     * Calculate the means and standard deviations of the dataset.
     *
     * @param  \Rubix\Engine\Dataset  $data
     * @return void
     */
    public function fit(Dataset $data) : void
    {
        $this->columnTypes = $data->columnTypes();
        $this->stats = [];

        foreach ($data->rotate() as $column => $features) {
            if ($this->columnTypes[$column] === self::CONTINUOUS) {
                $mean = Average::mean($features);

                $stddev = sqrt(array_reduce($features, function ($carry, $feature) use ($mean) {
                    return $carry += ($feature - $mean) ** 2;
                }, 0) / count($features));

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
                $stddev = $this->stddevs[$column];

                $sample[$column] = ($sample[$column] - $mean) / ($stddev ? $stddev : self::EPSILON);
            }
        }
    }
}
