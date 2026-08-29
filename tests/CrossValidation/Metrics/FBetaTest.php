<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Tuple;
use Rubix\ML\EstimatorType;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Metrics')]
#[CoversClass(FBeta::class)]
class FBetaTest extends TestCase
{
    protected FBeta $metric;

    /**
     * @return Generator<array>
     */
    public static function scoreProvider() : Generator
    {
        yield [
            ['wolf', 'lamb', 'wolf', 'lamb', 'wolf'],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            0.5833333333333333,
        ];

        yield [
            ['wolf', 'wolf', 'lamb', 'lamb', 'lamb'],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            0.0,
        ];

        yield [
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            ['lamb', 'lamb', 'wolf', 'wolf', 'wolf'],
            1.0,
        ];

        yield [
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            0.8076923076923077,
        ];

        yield [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            1.0,
        ];

        yield [
            [1, 1, 1, 0, 1],
            [0, 0, 0, 1, 0],
            0.0,
        ];
    }

    protected function setUp() : void
    {
        $this->metric = new FBeta(1.0);
    }

    public function testRange() : void
    {
        $tuple = $this->metric->range();

        $this->assertInstanceOf(Tuple::class, $tuple);
        $this->assertCount(2, $tuple);
        $this->assertGreaterThan($tuple[0], $tuple[1]);
    }

    public function testCompatibility() : void
    {
        $expected = [
            EstimatorType::classifier(),
            EstimatorType::anomalyDetector(),
        ];

        $this->assertEquals($expected, $this->metric->compatibility());
    }

    /**
     * @param (string|int)[] $predictions
     * @param (string|int)[] $labels
     * @param float $expected
     */
    #[DataProvider('scoreProvider')]
    public function testScore(array $predictions, array $labels, float $expected) : void
    {
        [$min, $max] = $this->metric->range()->list();

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        $this->assertThat(
            $score,
            $this->logicalAnd(
                $this->greaterThanOrEqual($min),
                $this->lessThanOrEqual($max)
            )
        );

        $this->assertEquals($expected, $score);
    }
}
