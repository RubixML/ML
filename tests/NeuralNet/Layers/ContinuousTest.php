<?php

namespace Rubix\Tests\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Continuous;
use Rubix\ML\NeuralNet\Layers\Parametric;
use PHPUnit\Framework\TestCase;

class ContinuousTest extends TestCase
{
    protected $layer;

    public function setUp()
    {
        $this->layer = new Continuous();
    }

    public function test_build_continuous_layer()
    {
        $this->assertInstanceOf(Continuous::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Output::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);
    }

    public function test_width()
    {
        $this->assertEquals(1, $this->layer->width());
    }
}
