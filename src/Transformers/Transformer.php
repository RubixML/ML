<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Dataset;

interface Transformer
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;
    const EPSILON = 1e-8;

    /**
     * @param  \Rubix\Engine\Dataset  $data
     * @return void
     */
    public function fit(Dataset $data) : void;

    /**
     * @param  array  $samples
     * @return array
     */
    public function transform(array &$samples) : void;
}
