<?php

namespace Rubix\Tests\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Layers\AlphaDropout;
use Rubix\ML\NeuralNet\ActivationFunctions\SELU;
use PHPUnit\Framework\TestCase;

class AlphaDropoutTest extends TestCase
{
    protected $layer;

    public function setUp()
    {
        $this->layer = new AlphaDropout(10, new SELU(), 0.1);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(AlphaDropout::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Hidden::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);
    }

    public function test_width()
    {
        $this->assertEquals(11, $this->layer->width());
    }
}
