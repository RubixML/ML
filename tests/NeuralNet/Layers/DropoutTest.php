<?php

namespace Rubix\Tests\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\NeuralNet\Layers\Nonparametric;
use PHPUnit\Framework\TestCase;

class DropoutTest extends TestCase
{
    protected $fanIn;

    protected $fanOut;

    protected $input;

    protected $output;

    protected $layer;

    public function setUp()
    {
        $this->fanIn = 5;

        $this->fanOut = 5;

        $this->input = new Matrix([
            [1., 2.5,],
            [0.1, 0.],
            [0.002, -6.],
        ], false);

        $this->output = [
            [1., 2.5,],
            [0.1, 0.],
            [0.002, -6.],
        ];

        $this->layer = new Dropout(0.5);

        $this->layer->init($this->fanIn);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(Dropout::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Hidden::class, $this->layer);
        $this->assertInstanceOf(Nonparametric::class, $this->layer);
    }

    public function test_width()
    {
        $this->assertEquals($this->fanOut, $this->layer->width());
    }

    public function test_forward()
    {
        $out = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $out);
        $this->assertEquals([3, 2], $out->shape());
    }

    public function test_infer()
    {
        $out = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $out);
        $this->assertEquals($this->output, $out->asArray());
    }
}
