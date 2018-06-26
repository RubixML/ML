<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

interface Online extends Estimator
{
    /**
     * Perform a partial train.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function partial(Dataset $dataset) : void;
}
