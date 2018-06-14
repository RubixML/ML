<?php

namespace Rubix\Engine;

use Rubix\Engine\Datasets\Dataset;

interface Unsupervised
{
    /**
     * Train the estimator with an unlabeled dataset.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @return void
     */
    public function train(Dataset $dataset) : void;
}
