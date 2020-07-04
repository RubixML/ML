<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Estimator;
use Rubix\ML\Regressors\Ridge;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\AnomalyDetectors\RobustZScore;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\CrossValidation\Metrics\MeanSquaredError;
use Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use Generator;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric
 */
class EstimatorIsCompatibleWithMetricTest extends TestCase
{
    /**
     * @test
     * @dataProvider checkProvider
     *
     * @param \Rubix\ML\Estimator $estimator
     * @param \Rubix\ML\CrossValidation\Metrics\Metric $metric
     * @param bool $valid
     */
    public function check(Estimator $estimator, Metric $metric, bool $valid) : void
    {
        if (!$valid) {
            $this->expectException(InvalidArgumentException::class);
        }

        EstimatorIsCompatibleWithMetric::check($estimator, $metric);

        $this->assertTrue($valid);
    }

    /**
     * @return \Generator<array>
     */
    public function checkProvider() : Generator
    {
        yield [
            new SoftmaxClassifier(),
            new MeanSquaredError(),
            false,
        ];

        yield [
            new SoftmaxClassifier(),
            new Accuracy(),
            true,
        ];

        yield [
            new RobustZScore(),
            new Accuracy(),
            true,
        ];

        yield [
            new Ridge(),
            new VMeasure(),
            false,
        ];

        yield [
            new Ridge(),
            new MeanSquaredError(),
            true,
        ];
    }
}
