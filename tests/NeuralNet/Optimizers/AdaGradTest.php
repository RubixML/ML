<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Tensor\Tensor;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\AdaGrad;
use Rubix\ML\NeuralNet\Optimizers\Adaptive;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Optimizers')]
#[CoversClass(AdaGrad::class)]
class AdaGradTest extends TestCase
{
    /**
     * @var AdaGrad
     */
    protected AdaGrad $optimizer;

    /**
     * @return Generator<mixed[]>
     */
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
                [0.001, 0.001, -0.001],
                [-0.001, 0.001, 0.001],
                [0.001, -0.001, -0.001],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->optimizer = new AdaGrad(0.001);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(AdaGrad::class, $this->optimizer);
        $this->assertInstanceOf(Adaptive::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    /**
     * @param Parameter $param
     * @param Tensor<int|float> $gradient
     * @param list<list<float>> $expected
     */
    #[DataProvider('stepProvider')]
    #[Test]
    public function step(Parameter $param, Tensor $gradient, array $expected) : void
    {
        $this->optimizer->warm($param);

        $step = $this->optimizer->step($param, $gradient);

        $this->assertEquals($expected, $step->asArray());
    }
}
