<?php

namespace Rubix\Tests\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Placeholder;
use PHPUnit\Framework\TestCase;

class PlaceholderTest extends TestCase
{
    protected $layer;

    public function setUp()
    {
        $this->layer = new Placeholder(50);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(Placeholder::class, $this->layer);
        $this->assertInstanceOf(Input::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
    }

    public function test_width()
    {
        $this->assertEquals(50, $this->layer->width());
    }
}
