<?php

namespace Rubix\Tests\Helpers;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\AnomalyDetectors\Detector;

class MockDetector implements Detector
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

    public function predict(Dataset $dataset) : array
    {
        return $this->predictions;
    }
}
