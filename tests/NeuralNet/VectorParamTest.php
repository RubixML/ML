<?php

namespace Rubix\ML\Tests\NeuralNet;

use Rubix\Tensor\Vector;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\VectorParam;
use PHPUnit\Framework\TestCase;

class VectorParamTest extends TestCase
{
    protected $param;

    public function setUp()
    {
        $this->param = new VectorParam(Vector::quick([5]));
    }

    public function test_build_parameter()
    {
        $this->assertInstanceOf(VectorParam::class, $this->param);
        $this->assertInstanceOf(Parameter::class, $this->param);
    }

    public function test_id()
    {
        $this->assertInternalType('integer', $this->param->id());
    }

    public function test_w()
    {
        $w = $this->param->w();

        $this->assertInstanceOf(Vector::class, $w);
        $this->assertEquals(5, $w[0]);
    }

    public function test_update()
    {
        $step = Vector::quick([2]);

        $this->param->update($step);

        $w = $this->param->w();

        $this->assertEquals(3, $w[0]);
    }
}
