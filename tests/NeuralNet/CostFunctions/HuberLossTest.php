<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('CostFunctions')]
#[CoversClass(HuberLoss::class)]
class HuberLossTest extends TestCase
{
    protected HuberLoss $costFn;

    public static function computeProvider() : Generator
    {
        yield [
            Matrix::quick([
                [0.99],
            ]),
            Matrix::quick([
                [1.0],
            ]),
            4.9998750062396624E-5,
        ];

        yield [
            Matrix::quick([
                [1000.0],
            ]),
            Matrix::quick([
                [1.0],
            ]),
            998.0005005003751,
        ];

        yield [
            Matrix::quick([
                [33.98],
                [20.0],
                [4.6],
                [44.2],
                [38.5],
            ]),
            Matrix::quick([
                [36.0],
                [22.0],
                [18.0],
                [41.5],
                [38.0],
            ]),
            3.384914773928223,
        ];
    }

    public static function differentiateProvider() : Generator
    {
        yield [
            Matrix::quick([
                [0.99],
            ]),
            Matrix::quick([
                [1.0],
            ]),
            [
                [-0.009999500037496884],
            ],
        ];

        yield [
            Matrix::quick([
                [1000.0],
            ]),
            Matrix::quick([
                [1.0],
            ]),
            [
                [0.999999498998874],
            ],
        ];

        yield [
            Matrix::quick([
                [33.98],
                [20.0],
                [4.6],
                [44.2],
                [38.5],
            ]),
            Matrix::quick([
                [36.0],
                [22.0],
                [18.0],
                [41.5],
                [38.0],
            ]),
            [
                [-0.8961947919452747],
                [-0.8944271909999159],
                [-0.9972269926097788],
                [0.9377487607237037],
                [0.4472135954999579],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->costFn = new HuberLoss(1.0);
    }

    /**
     * @param Matrix $output
     * @param Matrix $target
     * @param float $expected
     */
    #[DataProvider('computeProvider')]
    public function testCompute(Matrix $output, Matrix $target, float $expected) : void
    {
        $loss = $this->costFn->compute(output: $output, target: $target);

        $this->assertEqualsWithDelta($expected, $loss, 1e-8);
    }

    /**
     * @param Matrix $output
     * @param Matrix $target
     * @param list<list<float>> $expected
     */
    #[DataProvider('differentiateProvider')]
    public function testDifferentiate(Matrix $output, Matrix $target, array $expected) : void
    {
        $gradient = $this->costFn->differentiate($output, $target)->asArray();

        $this->assertEqualsWithDelta($expected, $gradient, 1e-8);
    }
}
