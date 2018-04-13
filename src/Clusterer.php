<?php

namespace Rubix\Engine;

interface Clusterer
{
    /**
     * Return an array of samples from a dataset organized by cluster.
     *
     * @param  \Rubix\Engine\Dataset  $data
     * @return array
     */
    public function cluster(Dataset $data) : array;
}
