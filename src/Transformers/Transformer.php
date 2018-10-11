<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\DataFrame;

interface Transformer
{
    const EPSILON = 1e-8;

    /**
     * Transform the sample matrix.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void;
}
