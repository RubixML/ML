<?php

namespace Rubix\ML\Other\Generators;

use Rubix\ML\Datasets\Dataset;

interface Generator
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;

    const EPSILON = 1e-8;

    /**
     * Generate n data points.
     *
     * @param  int  $n
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function generate(int $n = 100) : Dataset;
}
