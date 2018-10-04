<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Layers\Parametric;
use PHPUnit\Framework\TestCase;

class DenseTest extends TestCase
{
    protected $fanIn;

    protected $fanOut;

    protected $input;

    protected $layer;

    public function setUp()
    {
        $this->fanIn = 3;

        $this->fanOut = 5;

        $this->input = new Matrix([
            [1., 2.5, -4.],
            [0.1, 0., 2.2],
            [0.002, -6., 1.2],
        ], false);

        $this->layer = new Dense($this->fanOut, new He());

        $this->layer->init($this->fanIn);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(Dense::class, $this->layer);
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
        $this->assertEquals([5, 3], $out->shape());
    }

    public function test_infer()
    {
        $out = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $out);
        $this->assertEquals([5, 3], $out->shape());
    }
}
