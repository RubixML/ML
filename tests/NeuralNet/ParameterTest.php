<?php

namespace Rubix\ML\Tests\NeuralNet;

use Tensor\Matrix;
use Tensor\ColumnVector;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

use function sqrt;

#[Group('NeuralNet')]
#[CoversClass(Parameter::class)]
class ParameterTest extends TestCase
{
    /**
     * @var Parameter
     */
    protected Parameter $param;

    /**
     * @var \Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer;

    protected function setUp() : void
    {
        $this->param = new Parameter(Matrix::quick([
            [5, 4],
            [-2, 6],
        ]));

        $this->optimizer = new Stochastic(new Constant());
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(Parameter::class, $this->param);
    }

    #[Test]
    public function id() : void
    {
        $this->assertIsInt($this->param->id());
    }

    #[Test]
    public function accumulateGradient() : void
    {
        $gradient = Matrix::quick([
            [2, 1],
            [1, -2],
        ]);

        $this->assertNull($this->param->gradient());

        $this->param->accumulate($gradient);

        $this->assertInstanceOf(Matrix::class, $this->param->gradient());
        $this->assertEquals($gradient->asArray(), $this->param->gradient()->asArray());

        $this->param->accumulate($gradient);

        $expected = [
            [4, 2],
            [2, -4],
        ];

        $this->assertEquals($expected, $this->param->gradient()->asArray());
    }

    #[Test]
    public function resetGradient() : void
    {
        $this->param->accumulate(Matrix::quick([
            [2, 1],
            [1, -2],
        ]));

        $this->param->resetGradient();

        $this->assertNull($this->param->gradient());
    }

    #[Test]
    public function scaleGradient() : void
    {
        $gradient = Matrix::quick([
            [2, 1],
            [1, -2],
        ]);

        $this->param->accumulate($gradient);
        $this->param->accumulate($gradient);

        $this->param->scaleGradient(0.5);

        $expected = [
            [2, 1],
            [1, -2],
        ];

        $this->assertEquals($expected, $this->param->gradient()->asArray());

        $this->param->resetGradient();

        $this->expectException(RuntimeException::class);

        $this->param->scaleGradient(0.5);
    }

    #[Test]
    public function gradientNorm() : void
    {
        $gradient = Matrix::quick([
            [2, 1],
            [1, -2],
        ]);

        $this->param->accumulate($gradient);

        $this->assertEqualsWithDelta(sqrt(10.0), $this->param->gradientNorm(), 1e-8);
    }

    #[Test]
    public function gradientNormOfColumnVector() : void
    {
        $param = new Parameter(ColumnVector::quick([1.0, -2.0, 3.0, 4.0]));

        $param->accumulate(ColumnVector::quick([2.0, 0.0, -4.0, 0.0]));

        $this->assertEqualsWithDelta(sqrt(20.0), $param->gradientNorm(), 1e-8);
    }

    #[Test]
    public function gradientNormWithoutGradient() : void
    {
        $this->expectException(RuntimeException::class);

        $this->param->gradientNorm();
    }

    #[Test]
    public function update() : void
    {
        $gradient = Matrix::quick([
            [2, 1],
            [1, -2],
        ]);

        $expected = [
            [4.98, 3.99],
            [-2.01, 6.02],
        ];

        $this->param->accumulate($gradient);

        $this->param->update($this->optimizer);

        $this->assertEquals($expected, $this->param->param()->asArray());
    }
}
