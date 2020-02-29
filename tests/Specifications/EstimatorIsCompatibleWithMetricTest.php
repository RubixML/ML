<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\CrossValidation\Metrics\MeanSquaredError;
use Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric
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
