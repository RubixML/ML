<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Multiclass;
use Rubix\ML\NeuralNet\Layers\Parametric;
use PHPUnit\Framework\TestCase;

class MulticlassTest extends TestCase
{
    protected $fanIn;

    protected $fanOut;

    protected $input;

    protected $layer;

    public function setUp()
    {
        $this->fanIn = 4;

        $this->fanOut = 3;

        $this->input = new Matrix([
            [1., 2.5, -4.,],
            [0.1, 0., 2.2],
            [0.002, -6., 1.2],
            [0.5, -0.05, 0.1],
        ], false);

        $this->layer = new Multiclass(['hot', 'cold', 'ice cold']);

        $this->layer->init($this->fanIn);
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
