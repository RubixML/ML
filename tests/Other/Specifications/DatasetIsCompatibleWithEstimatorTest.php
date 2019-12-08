<?php

namespace Rubix\ML\Tests\Other\Specifications;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\Other\Specifications\DatasetIsCompatibleWithEstimator;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class DatasetIsCompatibleWithEstimatorTest extends TestCase
{
    public function test_check() : void
    {
        $estimator = new NaiveBayes();

        $dataset = Unlabeled::quick([
            [6., -1.1, 5, 'college'],
        ]);

        $this->expectException(InvalidArgumentException::class);

        DatasetIsCompatibleWithEstimator::check($dataset, $estimator);
    }
}
