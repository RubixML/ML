<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Tensor\Tensor;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\AdaMax;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Scheduler;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Optimizers')]
#[CoversClass(AdaMax::class)]
class AdaMaxTest extends TestCase
{
    /**
     * @var AdaMax
     */
    protected AdaMax $optimizer;

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
                [0.0001, 0.0001, -0.0001],
                [-0.0001, 0.0001, 0.0001],
                [0.0001, -0.0001, -0.0001],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->optimizer = new AdaMax(new Constant(0.001), 0.1, 0.001);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(AdaMax::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    #[Test]
    public function scheduler() : void
    {
        $this->assertInstanceOf(Scheduler::class, $this->optimizer->scheduler());
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

        $this->assertEqualsWithDelta($expected, $step->asArray(), 1e-8);
    }
}
