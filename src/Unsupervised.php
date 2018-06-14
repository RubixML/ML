<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

interface Unsupervised
{
    /**
     * Train the estimator with an unlabeled dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function train(Dataset $dataset) : void;
}
