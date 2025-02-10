<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Tensor\Tensor;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\AdaMax;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Optimizers')]
#[CoversClass(AdaMax::class)]
class AdaMaxTest extends TestCase
{
    protected AdaMax $optimizer;

    public static function stepProvider() : Generator
    {
        yield [
            new Parameter(Matrix::quick([
                [0.1, 0.6, -0.4],
                [0.5, 0.6, -0.4],
                [0.1, 0.1, -0.7],
            ])),
            Matrix::quick([
                [0.01, 0.05, -0.02],
                [-0.01, 0.02, 0.03],
                [0.04, -0.01, -0.5],
            ]),
            [
                [0.0001, 0.0001, -0.0001],
                [-0.0001, 0.0001, 0.0001],
                [0.0001, -0.0001, -0.0001],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->optimizer = new AdaMax(
            rate: 0.001,
            momentumDecay: 0.1,
            normDecay: 0.001
        );
    }

    /**
     * @param Parameter $param
     * @param Tensor<int|float> $gradient
     * @param list<list<float>> $expected
     */
    #[DataProvider('stepProvider')]
    public function testStep(Parameter $param, Tensor $gradient, array $expected) : void
    {
        $this->optimizer->warm($param);

        $step = $this->optimizer->step(param: $param, gradient: $gradient);

        $this->assertEqualsWithDelta($expected, $step->asArray(), 1e-8);
    }
}
