<?php

namespace Rubix\Tests\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\NeuralNet\Layers\Nonparametric;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use PHPUnit\Framework\TestCase;

class DropoutTest extends TestCase
{
    protected $layer;

    public function setUp()
    {
        $this->layer = new Dropout(0.5);
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
        $this->layer->init(10);

        $this->assertEquals(10, $this->layer->width());
    }
}
