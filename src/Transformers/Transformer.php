<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\DataFrame;

interface Transformer
{
    const EPSILON = 1e-8;
    
    /**
     * Fit the transformer to the data.
     *
     * @param  \Rubix\ML\Datasets\DataFrame  $dataframe
     * @return void
     */
    public function fit(DataFrame $dataframe) : void;

    /**
     * Apply the transformation to the sample matrix.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void;
}
