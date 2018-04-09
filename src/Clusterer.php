<?php

namespace Rubix\Engine;

interface Clusterer
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;
    
    /**
     * Return an array of samples from a dataset organized by cluster.
     *
     * @param  \Rubix\Engine\Dataset  $data
     * @return array
     */
    public function cluster(Dataset $data) : array;
}
