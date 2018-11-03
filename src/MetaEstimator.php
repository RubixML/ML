<?php

namespace Rubix\ML;

interface MetaEstimator extends Estimator
{
    /**
     * Return the base estimator instance.
     *
     * @return \Rubix\ML\Estimator
     */
    public function estimator() : Estimator;
}
