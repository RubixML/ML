<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use RuntimeException;

class MinMaxNormalizer implements Transformer
{
    /**
     * The computed minimums of the fitted data indexed by column.
     *
     * @var array|null
     */
    protected $minimums;

    /**
     * The computed maximums of the fitted data indexed by column.
     *
     * @var array|null
     */
    protected $maximums;

    /**
     * Calculate the minimums and maximums of each feature column in the dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
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
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples) : void
    {
        if (!isset($this->minimums) or !isset($this->maximums)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->minimums as $column => $min) {
                $sample[$column] = ($sample[$column] - $min)
                    / (($this->maximums[$column] - $min) + self::EPSILON);
            }
        }
    }
}
