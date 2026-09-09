<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Optimizers\GlobalNormClipped;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\StepDecay;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('Optimizers')]
#[CoversClass(GlobalNormClipped::class)]
class GlobalNormClippedTest extends TestCase
{
    /**
     * @var GlobalNormClipped
     */
    protected GlobalNormClipped $optimizer;

    protected function setUp() : void
    {
        $this->optimizer = new GlobalNormClipped(new Stochastic(new Constant(0.01)), 2.0);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(GlobalNormClipped::class, $this->optimizer);
        $this->assertInstanceOf(Optimizer::class, $this->optimizer);
    }

    #[Test]
    public function badMax() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->expectExceptionMessage('Max must be greater than 0, 0 given.');

        new GlobalNormClipped(new Stochastic(new Constant(0.01)), 0.0);
    }

    #[Test]
    public function step() : void
    {
        $first = new Parameter(Matrix::quick([[1.0]]));

        $second = new Parameter(Matrix::quick([[1.0]]));

        $firstGradient = Matrix::quick([[3.0]]);

        $secondGradient = Matrix::quick([[4.0]]);

        $this->optimizer->step([[$first, $firstGradient], [$second, $secondGradient]]);

        $expected = [
            [1.0 - 3.0 * 0.4 * 0.01],
        ];

        $this->assertEqualsWithDelta($expected, $first->param()->asArray(), 1e-8);

        $expected = [
            [1.0 - 4.0 * 0.4 * 0.01],
        ];

        $this->assertEqualsWithDelta($expected, $second->param()->asArray(), 1e-8);

        $first = new Parameter(Matrix::quick([[1.0]]));

        $second = new Parameter(Matrix::quick([[1.0]]));

        $firstGradient = Matrix::quick([[0.5]]);

        $secondGradient = Matrix::quick([[0.5]]);

        $this->optimizer->step([[$first, $firstGradient], [$second, $secondGradient]]);

        $expected = [
            [0.995],
        ];

        $this->assertEqualsWithDelta($expected, $first->param()->asArray(), 1e-8);

        $expected = [
            [0.995],
        ];

        $this->assertEqualsWithDelta($expected, $second->param()->asArray(), 1e-8);

        $scheduler = new StepDecay(0.1, 2, 0.5);

        $optimizer = new GlobalNormClipped(new Stochastic($scheduler), 1.0);

        $param = new Parameter(Matrix::quick([[0.1, 0.2]]));

        $gradient = Matrix::quick([[0.5, 0.5]]);

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
        $this->assertEquals('Global Norm Clipped (optimizer: Stochastic (scheduler: Constant (rate: 0.01)), max: 2)', (string) $this->optimizer);
    }
}
