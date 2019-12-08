<?php

namespace Rubix\ML\Tests\NeuralNet\Parameters;

use Tensor\Vector;
use Rubix\ML\NeuralNet\Parameters\Parameter;
use Rubix\ML\NeuralNet\Parameters\VectorParam;
use PHPUnit\Framework\TestCase;

class VectorParamTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\Parameters\VectorParam
     */
    protected $param;

    public function setUp() : void
    {
        $this->param = new VectorParam(Vector::quick([5]));
    }

    public function test_build_parameter() : void
    {
        $this->assertInstanceOf(VectorParam::class, $this->param);
        $this->assertInstanceOf(Parameter::class, $this->param);
    }

    public function test_id() : void
    {
        $this->assertIsInt($this->param->id());
    }

    public function test_w() : void
    {
        $w = $this->param->w();

        $this->assertInstanceOf(Vector::class, $w);
        $this->assertEquals(5, $w[0]);
    }

    public function test_update() : void
    {
        $step = Vector::quick([2]);

        $this->param->update($step);

        $w = $this->param->w();

        $this->assertEquals(3, $w[0]);
    }
}
