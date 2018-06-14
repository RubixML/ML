<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;

interface Transformer
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;
    
    const EPSILON = 1e-8;

    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void;

    /**
     * @param  array  $samples
     * @return array
     */
    public function transform(array &$samples) : void;
}
