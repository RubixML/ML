<?php

namespace Rubix\ML\Tests\NeuralNet\Parameters;

use Tensor\Vector;
use Rubix\ML\NeuralNet\Parameters\Parameter;
use Rubix\ML\NeuralNet\Parameters\VectorParam;
use PHPUnit\Framework\TestCase;

/**
 * @group Parameters
 * @covers \Rubix\ML\NeuralNet\Parameters\VectorParam
 */
class VectorParamTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\Parameters\VectorParam
     */
    protected $param;
    
    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->param = new VectorParam(Vector::quick([5]));
    }
    
    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(VectorParam::class, $this->param);
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

        $this->assertInstanceOf(Vector::class, $w);
        $this->assertEquals(5, $w[0]);
    }
    
    /**
     * @test
     */
    public function update() : void
    {
        $step = Vector::quick([2]);

        $this->param->update($step);

        $w = $this->param->w();

        $this->assertEquals(3, $w[0]);
    }
}
