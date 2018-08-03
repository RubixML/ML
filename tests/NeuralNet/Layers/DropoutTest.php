<?php

namespace Rubix\Tests\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use PHPUnit\Framework\TestCase;

class DropoutTest extends TestCase
{
    protected $layer;

    public function setUp()
    {
        $this->layer = new Dropout(10, new ELU(), 0.5);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(Dropout::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Hidden::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);
    }

    public function test_width()
    {
        $this->assertEquals(11, $this->layer->width());
    }
}
