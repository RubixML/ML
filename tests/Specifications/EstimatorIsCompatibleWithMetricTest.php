<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Specifications;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Regressors\Ridge;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\AnomalyDetectors\RobustZScore;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\CrossValidation\Metrics\MeanSquaredError;
use Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Specifications')]
#[CoversClass(EstimatorIsCompatibleWithMetric::class)]
class EstimatorIsCompatibleWithMetricTest extends TestCase
{
    public static function passesProvider() : Generator
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

    /**
     * @param EstimatorIsCompatibleWithMetric $specification
     * @param bool $expected
     */
    #[DataProvider('passesProvider')]
    public function testPasses(EstimatorIsCompatibleWithMetric $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }
}
