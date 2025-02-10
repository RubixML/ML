<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\CrossValidation\Metrics;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Tuple;
use Rubix\ML\EstimatorType;
use Rubix\ML\CrossValidation\Metrics\MeanAbsoluteError;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Metrics')]
#[CoversClass(MeanAbsoluteError::class)]
class MeanAbsoluteErrorTest extends TestCase
{
    protected MeanAbsoluteError $metric;

    /**
     * @return Generator<array>
     */
    public static function scoreProvider() : Generator
    {
        yield [
            [7, 9.5, -20, -500, .079],
            [10, 10.0, 6, -1400, .08],
            -185.90019999999998,
        ];

        yield [
            [0, 0, 0, 0, 0],
            [10, 10.0, 6, -1400, .08],
            -285.216,
        ];

        yield [
            [10, 10.0, 6, -1400, .08],
            [10, 10.0, 6, -1400, .08],
            0.0,
        ];
    }

    protected function setUp() : void
    {
        $this->metric = new MeanAbsoluteError();
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
            EstimatorType::regressor(),
        ];

        $this->assertEquals($expected, $this->metric->compatibility());
    }

    /**
     * @param (int|float)[] $predictions
     * @param (int|float)[] $labels
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
