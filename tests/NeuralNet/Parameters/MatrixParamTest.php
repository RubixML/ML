<?php

namespace Rubix\ML\Tests\NeuralNet\Parameters;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameters\Parameter;
use Rubix\ML\NeuralNet\Parameters\MatrixParam;
use PHPUnit\Framework\TestCase;

/**
 * @group Parameters
 * @covers \Rubix\ML\NeuralNet\Parameters\MatrixParam
 */
class MatrixParamTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\Parameters\MatrixParam
     */
    protected $param;
    
    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->param = new MatrixParam(Matrix::quick([[5]]));
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(MatrixParam::class, $this->param);
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
    public function w() : void
    {
        $w = $this->param->w();

        $this->assertInstanceOf(Matrix::class, $w);
        $this->assertEquals(5, $w[0][0]);
    }
    
    /**
     * @test
     */
    public function update() : void
    {
        $step = Matrix::quick([[2]]);

        $this->param->update($step);

        $w = $this->param->w();

        $this->assertEquals(3, $w[0][0]);
    }
}
