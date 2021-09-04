<?php

namespace Rubix\ML\Datasets\Generators;

interface Generator
{
    /**
     * Return the dimensionality of the data this generates.
     *
     * @internal
     *
     * @return int<0,max>
     */
    public function dimensions() : int;

    /**
     * Generate n data points.
     *
     * @param int<0,max> $n
     * @return \Rubix\ML\Datasets\Dataset
     */
    public function generate(int $n);
}
