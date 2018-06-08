<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Datasets\Dataset;
use InvalidArgumentException;

class MinMaxNormalizer implements Transformer
{
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
     * Calculate the minimums and maximums of each feature column in the dataset.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        $this->minimums = $this->maximums = [];

        foreach ($dataset->ColumnTypes() as $column => $type) {
            if ($type === self::CONTINUOUS) {
                $values = $dataset->column($column);

                $this->minimums[$column] = min($values);
                $this->maximums[$column] = max($values);
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
                $sample[$column] = ($sample[$column] - $min)
                    / (($this->maximums[$column] - $min) + self::EPSILON);
            }
        }
    }
}
