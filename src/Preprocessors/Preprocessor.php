<?php

namespace Rubix\Engine\Preprocessors;

use Rubix\Engine\Dataset;

interface Preprocessor
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;

    const EPSILON = 1e-10;

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
