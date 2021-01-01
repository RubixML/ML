<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Regressors\Ridge;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\AnomalyDetectors\RobustZScore;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\CrossValidation\Metrics\MeanSquaredError;
use Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Specifications
 * @covers \Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric
 */
class EstimatorIsCompatibleWithMetricTest extends TestCase
{
    /**
     * @test
     * @dataProvider passesProvider
     *
     * @param \Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric $specification
     * @param bool $expected
     */
    public function passes(EstimatorIsCompatibleWithMetric $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }

    /**
     * @return \Generator<array>
     */
    public function passesProvider() : Generator
    {
        yield [
            EstimatorIsCompatibleWithMetric::with(
                new SoftmaxClassifier(),
                new MeanSquaredError()
            ),
            false,
        ];

        yield [
            EstimatorIsCompatibleWithMetric::with(
                new SoftmaxClassifier(),
                new Accuracy()
            ),
            true,
        ];

        yield [
            EstimatorIsCompatibleWithMetric::with(
                new RobustZScore(),
                new Accuracy()
            ),
            true,
        ];

        yield [
            EstimatorIsCompatibleWithMetric::with(
                new Ridge(),
                new VMeasure()
            ),
            false,
        ];

        yield [
            EstimatorIsCompatibleWithMetric::with(
                new Ridge(),
                new MeanSquaredError()
            ),
            true,
        ];
    }
}
