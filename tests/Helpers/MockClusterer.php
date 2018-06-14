<?php

namespace Rubix\Tests\Helpers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Clusterers\Clusterer;

class MockClusterer implements Clusterer
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
