<?php

namespace Rubix\Tests\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\Layers\Parametric;
use PHPUnit\Framework\TestCase;

class PReLUTest extends TestCase
{
    protected $fanIn;

    protected $fanOut;

    protected $input;

    protected $layer;

    public function setUp()
    {
        $this->fanIn = 3;

        $this->fanOut = 3;

        $this->input = new Matrix([
            [1., 2.5, -4.],
            [0.1, 0., 2.2],
            [0.002, -6., 1.2],
        ], false);

        $this->layer = new PReLU(0.25);

        $this->layer->init($this->fanIn);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(PReLU::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Hidden::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);
    }

    public function test_width()
    {
        $this->assertEquals($this->fanOut, $this->layer->width());
    }

    public function test_forward()
    {
        $out = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $out);
        $this->assertEquals([3, 3], $out->shape());
    }

    public function test_infer()
    {
        $out = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $out);
        $this->assertEquals([3, 3], $out->shape());
    }
}
