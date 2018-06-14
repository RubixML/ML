<?php

namespace Rubix\Tests\Helpers;

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Datasets\Labeled;
use Rubix\Engine\Classifiers\Classifier;

class MockClassifier implements Classifier
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
