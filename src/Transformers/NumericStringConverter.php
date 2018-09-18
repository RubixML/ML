<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Other\Structures\DataFrame;

/**
 * Numeric String Converter
 *
 * This handy Transformer will convert all numeric strings into their floating
 * point counterparts. Useful for when extracting from a source that only
 * recognizes data as string types.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NumericStringConverter implements Transformer
{
    /**
     * Fit the transformer to the incoming data frame.
     *
     * @param  \Rubix\ML\Other\Structures\DataFrame  $dataframe
     * @return void
     */
    public function fit(DataFrame $dataframe) : void
    {
        //
    }

    /**
     * Apply the transformation to the samples in the data frame.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($sample as &$feature) {
                if (is_string($feature) and is_numeric($feature)) {
                    $feature = (int) $feature == $feature
                        ? (int) $feature
                        : (float) $feature;
                }
            }
        }
    }
}
