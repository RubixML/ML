<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Tensor\Tensor;
use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\NormClipped;
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

use function sqrt;

#[Group('Optimizers')]
#[CoversClass(NormClipped::class)]
class NormClippedTest extends TestCase
{
    /**
     * @var NormClipped
     */
    protected NormClipped $optimizer;

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

        $scale = 1.5 / 5.0;

        yield [
            new Parameter(Matrix::quick([
                [0.1, 0.6, -0.4],
                [0.5, 0.6, -0.4],
                [0.1, 0.1, -0.7],
            ])),
            Matrix::quick([
                [3.0, 0.0],
                [0.0, 4.0],
                [0.0, 0.0],
            ]),
            [
                [3.0 * $scale * 0.001, 0.0],
                [0.0, 4.0 * $scale * 0.001],
                [0.0, 0.0],
            ],
        ];
    }

    protected function setUp() : void
    {
        $this->optimizer = new NormClipped(new Stochastic(new Constant(0.001)), 1.5);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(NormClipped::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    #[Test]
    public function badMax() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->expectExceptionMessage('Max must be greater than 0, 0 given.');

        new NormClipped(new Stochastic(new Constant(0.001)), 0.0);
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

        $gradient = Matrix::quick([[3.0, 4.0]]);

        $norm = sqrt(3.0 ** 2 + 4.0 ** 2);

        $scale = 1.5 / $norm;

        $expected = [
            [0.1 - 3.0 * $scale * 0.001, 0.2 - 4.0 * $scale * 0.001],
        ];

        $this->optimizer->step([[$param, $gradient]]);

        $this->assertEqualsWithDelta($expected, $param->param()->asArray(), 1e-8);

        $scheduler = new StepDecay(0.1, 2, 0.5);

        $optimizer = new NormClipped(new Stochastic($scheduler), 1.0);

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
        $this->assertEquals('Norm Clipped (optimizer: Stochastic (scheduler: Constant (rate: 0.001)), max: 1.5)', (string) $this->optimizer);
    }
}
