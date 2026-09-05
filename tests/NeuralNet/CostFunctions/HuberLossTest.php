<?php

namespace Rubix\ML\Tests\NeuralNet\CostFunctions;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('CostFunctions')]
#[CoversClass(HuberLoss::class)]
class HuberLossTest extends TestCase
{
    /**
     * @var HuberLoss
     */
    protected HuberLoss $costFn;

    /**
     * @return Generator<mixed[]>
     */
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

    /**
     * @return Generator<mixed[]>
     */
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

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(HuberLoss::class, $this->costFn);
        $this->assertInstanceOf(CostFunction::class, $this->costFn);
    }

    /**
     * @param Matrix $output
     * @param Matrix $target
     * @param float $expected
     */
    #[DataProvider('computeProvider')]
    #[Test]
    public function compute(Matrix $output, Matrix $target, float $expected) : void
    {
        $loss = $this->costFn->compute($output, $target);

        $this->assertEqualsWithDelta($expected, $loss, 1e-8);
    }

    /**
     * @param Matrix $output
     * @param Matrix $target
     * @param list<list<float>> $expected
     */
    #[DataProvider('differentiateProvider')]
    #[Test]
    public function differentiate(Matrix $output, Matrix $target, array $expected) : void
    {
        $gradient = $this->costFn->differentiate($output, $target)->asArray();

        $this->assertEqualsWithDelta($expected, $gradient, 1e-8);
    }

    #[Test]
    public function differentiateMatchesNumericGradient() : void
    {
        $costFn = new HuberLoss(0.5);

        $output = Matrix::quick([
            [0.1, 0.5, 1.0],
            [2.0, 5.0, 10.0],
        ]);

        $target = Matrix::quick([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]);

        $epsilon = 1e-6;

        $numeric = [];

        foreach ($output->asArray() as $i => $row) {
            foreach ($row as $j => $v) {
                $plus = $costFn->_compute($target[$i][$j] - ($v + $epsilon));
                $minus = $costFn->_compute($target[$i][$j] - ($v - $epsilon));

                $numeric[$i][$j] = ($plus - $minus) / (2.0 * $epsilon);
            }
        }

        $this->assertEqualsWithDelta($numeric, $costFn->differentiate($output, $target)->asArray(), 1e-8);
    }
}
