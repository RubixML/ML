<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;

class NumericStringConverter implements Transformer
{
    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        //
    }

    /**
     * Convert all numerial strings to floating point numbers.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($sample as &$feature) {
                if (is_string($feature) and is_numeric($feature)) {
                    $feature = (float) $feature;
                }
            }
        }
    }
}
