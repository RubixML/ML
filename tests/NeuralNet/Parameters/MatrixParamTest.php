<?php

namespace Rubix\ML\Tests\NeuralNet\Parameters;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameters\Parameter;
use Rubix\ML\NeuralNet\Parameters\MatrixParam;
use PHPUnit\Framework\TestCase;

class MatrixParamTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\Parameters\MatrixParam
     */
    protected $param;

    public function setUp() : void
    {
        $this->param = new MatrixParam(Matrix::quick([[5]]));
    }

    public function test_build_parameter() : void
    {
        $this->assertInstanceOf(MatrixParam::class, $this->param);
        $this->assertInstanceOf(Parameter::class, $this->param);
    }

    public function test_id() : void
    {
        $this->assertIsInt($this->param->id());
    }

    public function test_w() : void
    {
        $w = $this->param->w();

        $this->assertInstanceOf(Matrix::class, $w);
        $this->assertEquals(5, $w[0][0]);
    }

    public function test_update() : void
    {
        $step = Matrix::quick([[2]]);

        $this->param->update($step);

        $w = $this->param->w();

        $this->assertEquals(3, $w[0][0]);
    }
}
