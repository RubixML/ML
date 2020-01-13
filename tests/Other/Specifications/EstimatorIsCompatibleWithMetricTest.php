<?php

namespace Rubix\ML\Tests\Other\Specifications;

use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\CrossValidation\Metrics\MeanSquaredError;
use Rubix\ML\Other\Specifications\EstimatorIsCompatibleWithMetric;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

/**
 * @group Specifications
 * @covers \Rubix\ML\Other\Specifications\EstimatorIsCompatibleWithMetric
 */
class EstimatorIsCompatibleWithMetricTest extends TestCase
{
    /**
     * @test
     */
    public function check() : void
    {
        $estimator = new NaiveBayes();

        $metric = new MeanSquaredError();

        $this->expectException(InvalidArgumentException::class);

        EstimatorIsCompatibleWithMetric::check($estimator, $metric);
    }
}
