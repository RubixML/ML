<?php

namespace Rubix\Tests\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\Layers\Placeholder;
use PHPUnit\Framework\TestCase;

class PlaceholderTest extends TestCase
{
    protected $fanIn;

    protected $fanOut;

    protected $input;

    protected $output;

    protected $layer;

    public function setUp()
    {
        $this->fanIn = 0;

        $this->fanOut = 3;

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

        $this->layer = new Placeholder($this->fanOut);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(Placeholder::class, $this->layer);
        $this->assertInstanceOf(Input::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
    }

    public function test_forward()
    {
        $out = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $out);
        $this->assertEquals($this->output, $out->asArray());
    }

    public function test_infer()
    {
        $out = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $out);
        $this->assertEquals($this->output, $out->asArray());
    }
}
