<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Placeholder;
use PHPUnit\Framework\TestCase;

class PlaceholderTest extends TestCase
{
    protected $fanIn;

    protected $input;

    protected $layer;

    public function setUp()
    {
        $this->fanIn = 0;

        $this->input = Matrix::quick([
            [1., 2.5,],
            [0.1, 0.],
            [0.002, -6.],
        ]);

        $this->layer = new Placeholder(3);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(Placeholder::class, $this->layer);
        $this->assertInstanceOf(Input::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
    }

    public function test_width()
    {
        $this->assertEquals(3, $this->layer->width());
    }

    public function test_forward_infer()
    {
        $out = $this->layer->forward($this->input);

        $output = [
            [1., 2.5,],
            [0.1, 0.],
            [0.002, -6.],
        ];

        $this->assertInstanceOf(Matrix::class, $out);
        $this->assertEquals([3, 2], $out->shape());
        $this->assertEquals($output, $out->asArray());

        $out = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $out);
        $this->assertEquals([3, 2], $out->shape());
        $this->assertEquals($output, $out->asArray());
    }
}
