<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Datasets\Dataset;

interface Transformer
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;
    const EPSILON = 1e-8;

    /**
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void;

    /**
     * @param  array  $samples
     * @return array
     */
    public function transform(array &$samples) : void;
}
