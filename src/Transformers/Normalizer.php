<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Datasets\Dataset;

class Normalizer implements Transformer
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
     * The computed minimums of the fitted data indexed by column.
     *
     * @var array
     */
    protected $minimums = [
        //
    ];

    /**
     * The computed maximums of the fitted data indexed by column.
     *
     * @var array
     */
    protected $maximums = [
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
     * Calculate the minimums and maximums of each feature column in the dataset.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $this->columnTypes = $dataset->columnTypes();
        $this->minimums = $this->maximums = [];

        foreach ($dataset->rotate() as $column => $features) {
            if ($this->columnTypes[$column] === self::CONTINUOUS) {
                $this->minimums[$column] = min($features);
                $this->maximums[$column] = max($features);
            }
        }
    }

    /**
     * Transform the features into a value between 0 and 1.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($this->minimums as $column => $min) {
                $max = $this->maximums[$column];

                $sample[$column] = ($sample[$column] - $min) / (($max - $min) ? ($max - $min) : self::EPSILON);
            }
        }
    }
}
