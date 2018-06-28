<?php

namespace Rubix\Tests\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Input;
use PHPUnit\Framework\TestCase;

class InputTest extends TestCase
{
    protected $layer;

    public function setUp()
    {
        $this->layer = new Input(50);
    }

    public function test_build_input_layer()
    {
        $this->assertInstanceOf(Input::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
    }

    public function test_width()
    {
        $this->assertEquals(51, $this->layer->width());
    }
}
