<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Labeled;

interface Supervised
{
    /**
     * Train the estimator with a labeled dataset.
     *
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @return void
     */
    public function train(Labeled $dataset) : void;
}
