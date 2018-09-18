<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Other\Structures\DataFrame;

interface Transformer
{
    /**
     * Fit the transformer to the incoming data frame.
     *
     * @param  \Rubix\ML\Other\Structures\DataFrame  $dataframe
     * @return void
     */
    public function fit(DataFrame $dataframe) : void;

    /**
     * Apply the transformation to the samples in the data frame.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void;
}
