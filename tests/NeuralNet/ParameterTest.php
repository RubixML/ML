<?php

namespace Rubix\ML\Tests\NeuralNet;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use PHPUnit\Framework\TestCase;

/**
 * @group NeuralNet
 * @covers \Rubix\ML\NeuralNet\Parameter
 */
class ParameterTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\Parameter
     */
    protected $param;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->param = new Parameter(Matrix::quick([
            [5, 4],
            [-2, 6],
        ]));
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Parameter::class, $this->param);
    }

    /**
     * @test
     */
    public function id() : void
    {
        $this->assertIsInt($this->param->id());
    }

    /**
     * @test
     */
    public function update() : void
    {
        $step = Matrix::quick([
            [2, 1],
            [1, -2],
        ]);

        $expected = [
            [3, 3],
            [-3, 8],
        ];

        $this->param->update($step);

        $this->assertEquals($expected, $this->param->param()->asArray());
    }
}
