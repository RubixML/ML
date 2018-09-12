<?php

namespace Rubix\Tests\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Parametric;
use PHPUnit\Framework\TestCase;

class PReLUTest extends TestCase
{
    protected $layer;

    public function setUp()
    {
        $this->layer = new PReLU(10, 0.25);
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
        $this->assertEquals(10, $this->layer->width());
    }
}
