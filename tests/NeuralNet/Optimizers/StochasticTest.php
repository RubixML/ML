<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Tensor\Tensor;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\StepDecay;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Cyclical;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Optimizers')]
#[CoversClass(Stochastic::class)]
class StochasticTest extends TestCase
{
    /**
     * @var Stochastic
     */
    protected Stochastic $optimizer;

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
        $this->optimizer = new Stochastic(new Constant(0.001));
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(Stochastic::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    #[Test]
    public function step() : void
    {
        $scheduler = new StepDecay(0.1, 2, 0.5);

        $optimizer = new Stochastic($scheduler);

        $initialRate = $scheduler->rate();

        $optimizer->scheduler()->tick();
        $optimizer->scheduler()->tick();

        $decreasedRate = $scheduler->rate();

        $this->assertLessThan($initialRate, $decreasedRate);
    }

    #[Test]
    public function stepWithCyclical() : void
    {
        $scheduler = new Cyclical(0.001, 0.006, 1, 0.5);

        $optimizer = new Stochastic($scheduler);

        $initialRate = $scheduler->rate();

        $optimizer->scheduler()->tick();

        $this->assertGreaterThan($initialRate, $scheduler->rate());
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

        $param->accumulate($gradient);

        $step = $this->optimizer->update($param);

        $this->assertEquals($expected, $step->asArray());
    }

    #[Test]
    public function flush() : void
    {
        $param = new Parameter(Matrix::quick([[0.1, 0.2]]));

        $gradient = Matrix::quick([[0.01, -0.03]]);

        $this->optimizer->warm($param);

        $param->accumulate($gradient);

        $this->optimizer->update($param);

        $param->resetGradient();

        $this->optimizer->flush();

        $this->optimizer->warm($param);

        $param->accumulate($gradient);

        $step = $this->optimizer->update($param);

        $this->assertEquals([[0.01 * 0.001, -0.03 * 0.001]], $step->asArray());
    }

    #[Test]
    public function stringRepresentation() : void
    {
        $this->assertEquals('Stochastic (scheduler: Constant (rate: 0.001))', (string) $this->optimizer);
    }
}
