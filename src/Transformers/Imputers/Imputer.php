<?php

namespace Rubix\Engine\Transformers\Imputers;

use Rubix\Engine\Datasets\Dataset;

interface Imputer
{
    const EPSILON = 1e-8;

    /**
     * Fit the imputer to the feature column of the training data.
     *
     * @param  array  $values
     * @return void
     */
    public function fit(array $values) : void;

    /**
     * Impute a missing value.
     *
     * @return mixed
     */
    public function impute();
}
