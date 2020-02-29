<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator
 */
class SamplesAreCompatibleWithEstimatorTest extends TestCase
{
    /**
     * @test
     */
    public function check() : void
    {
        $estimator = new NaiveBayes();

        $dataset = Unlabeled::quick([
            [6., -1.1, 5, 'college'],
        ]);

        $this->expectException(InvalidArgumentException::class);

        SamplesAreCompatibleWithEstimator::check($dataset, $estimator);
    }
}
