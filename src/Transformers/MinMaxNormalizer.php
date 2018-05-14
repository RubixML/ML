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
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This transformer only works on continuous features.');
        }

        $this->minimums = $this->maximums = [];

        foreach ($dataset->rotate() as $column => $features) {
            $this->minimums[$column] = min($features);
            $this->maximums[$column] = max($features);
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

                $sample[$column] = ($sample[$column] - $min)
                    / (($max - $min) ? ($max - $min) : self::EPSILON);
            }
        }
    }
}
