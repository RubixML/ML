<?php

namespace Rubix\ML;

interface Wrapper extends Estimator
{
    /**
     * Return the base estimator instance.
     *
     * @return \Rubix\ML\Estimator
     */
    public function base() : Estimator;
}
