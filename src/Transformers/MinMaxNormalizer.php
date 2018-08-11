<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use RuntimeException;

/**
 * Min Max Normalizer
 *
 * Often used as an alternative to Standard Scaling, the Min Max Normalization
 * scales the input features from a range of 0 to 1 by dividing the feature
 * value over the maximum value for that feature column.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
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
            if ($type === Dataset::CONTINUOUS) {
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
        if (is_null($this->minimums) or is_null($this->maximums)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->minimums as $column => $min) {
                $max = $this->maximums[$column];

                $denominator = $max - $min;

                $sample[$column] = $denominator !== 0.0
                    ? ($sample[$column] - $min) / $denominator : 1.0;
            }
        }
    }
}
