<?php

namespace Rubix\Engine\Clusterers;

use Rubix\Engine\Datasets\Unsupervised;

interface Clusterer
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;
    const EPSILON = 1e-8;

    /**
     * Return an array of samples from a dataset organized by cluster.
     *
     * @param  \Rubix\Engine\Datasets\Unsupervised  $data
     * @return array
     */
    public function cluster(Unsupervised $data) : array;
}
