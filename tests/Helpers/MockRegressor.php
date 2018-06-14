<?php

namespace Rubix\Tests\Helpers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Regressors\Regressor;

class MockRegressor implements Regressor
{
    public $predictions;

    public function __construct(array $predictions)
    {
        $this->predictions = $predictions;
    }

    public function train(Dataset $dataset) : void
    {
        //
    }

    public function predict(Dataset $samples) : array
    {
        return $this->predictions;
    }
}
