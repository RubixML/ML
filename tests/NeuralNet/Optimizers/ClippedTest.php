<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Tensor\Tensor;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Clipped;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\StepDecay;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Optimizers')]
#[CoversClass(Clipped::class)]
class ClippedTest extends TestCase
{
    /**
     * @var Clipped
     */
    protected Clipped $optimizer;

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
                [0.5, 0.02, -0.4],
                [-0.8, 0.03, 0.7],
                [0.004, -0.6, -1.2],
            ]),
            [
                [2e-4, 2e-5, -2e-4],
                [-2e-4, 3e-5, 2e-4],
                [4e-6, -2e-4, -2e-4],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->optimizer = new Clipped(new Stochastic(new Constant(0.001)), 0.2);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(Clipped::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    #[Test]
    public function badMax() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->expectExceptionMessage('Max must be greater than 0, 0 given.');

        new Clipped(new Stochastic(new Constant(0.001)), 0.0);
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

    #[Test]
    public function step() : void
    {
        $param = new Parameter(Matrix::quick([[0.1, 0.2]]));

        $gradient = Matrix::quick([[5.0, -5.0]]);

        $this->optimizer->step([[$param, $gradient]]);

        $expected = [
            [0.0998, 0.2002],
        ];

        $this->assertEqualsWithDelta($expected, $param->param()->asArray(), 1e-8);

        $scheduler = new StepDecay(0.1, 2, 0.5);

        $optimizer = new Clipped(new Stochastic($scheduler), 1.0);

        $param = new Parameter(Matrix::quick([[0.1, 0.2]]));

        $initialRate = $scheduler->rate();

        $optimizer->step([[$param, $gradient]]);

        $this->assertEquals($initialRate, $scheduler->rate());

        $optimizer->step([[$param, $gradient]]);

        $decreasedRate = $scheduler->rate();

        $this->assertLessThan($initialRate, $decreasedRate);
    }

    #[Test]
    public function flush() : void
    {
        $param = new Parameter(Matrix::quick([[0.1, 0.2]]));

        $gradient = Matrix::quick([[0.01, -0.03]]);

        $this->optimizer->warm($param);

        $this->optimizer->update($param, $gradient);

        $this->optimizer->flush();

        $this->optimizer->warm($param);

        $step = $this->optimizer->update($param, $gradient);

        $this->assertIsArray($step->asArray());
    }

    #[Test]
    public function stringRepresentation() : void
    {
        $this->assertEquals('Clipped (optimizer: Stochastic (scheduler: Constant (rate: 0.001)), max: 0.2)', (string) $this->optimizer);
    }
}
