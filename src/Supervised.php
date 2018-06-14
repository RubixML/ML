<?php

namespace Rubix\Engine;

use Rubix\Engine\Datasets\Labeled;

interface Supervised
{
    /**
     * Train the estimator with a labeled dataset.
     *
     * @param  \Rubix\Engine\Datasets\Labeled  $dataset
     * @return void
     */
    public function train(Labeled $dataset) : void;
}
