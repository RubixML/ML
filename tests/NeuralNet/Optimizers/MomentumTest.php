<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Tensor\Tensor;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Momentum;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\StepDecay;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Optimizers')]
#[CoversClass(Momentum::class)]
class MomentumTest extends TestCase
{
    /**
     * @var Momentum
     */
    protected Momentum $optimizer;

    /**
     * @return Generator<mixed[]>
     */
    public static function updateProvider() : Generator
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
                [1e-5, 5e-5, -2e-5],
                [-1e-5, 2e-5, 3e-5],
                [4e-5, -1e-5, -0.0005],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->optimizer = new Momentum(new Constant(0.001), 0.1, false);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(Momentum::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    #[Test]
    public function step() : void
    {
        $scheduler = new StepDecay(0.1, 2, 0.5);

        $optimizer = new Momentum($scheduler);

        $initialRate = $scheduler->rate();

        $optimizer->step();
        $optimizer->step();

        $decreasedRate = $scheduler->rate();

        $this->assertLessThan($initialRate, $decreasedRate);
    }

    /**
     * @param Parameter $param
     * @param Tensor<int|float> $gradient
     * @param list<list<float>> $expected
     */
    #[DataProvider('updateProvider')]
    #[Test]
    public function update(Parameter $param, Tensor $gradient, array $expected) : void
    {
        $this->optimizer->warm($param);

        $step = $this->optimizer->update($param, $gradient);

        $this->assertEquals($expected, $step->asArray());
    }
}
