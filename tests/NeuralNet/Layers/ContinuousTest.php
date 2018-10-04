<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Continuous;
use Rubix\ML\NeuralNet\Layers\Parametric;
use PHPUnit\Framework\TestCase;

class ContinuousTest extends TestCase
{
    protected $fanIn;

    protected $fanOut;

    protected $input;

    protected $layer;

    public function setUp()
    {
        $this->fanIn = 3;

        $this->fanOut = 1;

        $this->input = new Matrix([
            [1., 2.5, -4.],
            [0.1, 0., 2.2],
            [0.002, -6., 1.2],
        ], false);

        $this->layer = new Continuous();

        $this->layer->init($this->fanIn);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(Continuous::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Output::class, $this->layer);
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
        $this->assertEquals([1, 3], $out->shape());
    }

    public function test_infer()
    {
        $out = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $out);
        $this->assertEquals([1, 3], $out->shape());
    }
}
