<?php

namespace Rubix\ML\Datasets\Generators;

interface Generator
{
    /**
     * Return the dimensionality of the data this generates.
     *
     * @return int
     */
    public function dimensions() : int;

    /**
     * Generate n data points.
     *
     * @param int $n
     * @return \Rubix\ML\Datasets\Dataset<array>
     */
    public function generate(int $n);
}
