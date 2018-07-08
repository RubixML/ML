<?php

namespace Rubix\ML;

interface Ensemble extends Estimator
{
    /**
     * Return the ensemble of estimators.
     *
     * @return array
     */
    public function estimators() : array;
}
