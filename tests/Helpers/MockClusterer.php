<?php

namespace Rubix\Tests\Helpers;

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Clusterers\Clusterer;

class MockClusterer implements Clusterer
{
    public $predictions;

    public function __construct(array $predictions)
    {
        $this->predictions = $predictions;
    }

    public function predict(Dataset $samples) : array
    {
        return $this->predictions;
    }
}
