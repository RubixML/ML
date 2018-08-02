<?php

namespace Rubix\Tests\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Multinomial;
use Rubix\ML\NeuralNet\Layers\Parametric;
use PHPUnit\Framework\TestCase;

class MultinomialTest extends TestCase
{
    protected $layer;

    public function setUp()
    {
        $this->layer = new Multinomial(['hot', 'cold', 'ice cold']);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(Multinomial::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Output::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);
    }

    public function test_width()
    {
        $this->assertEquals(3, $this->layer->width());
    }
}
