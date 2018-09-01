<?php

namespace Rubix\Tests\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\Layers\Parametric;
use PHPUnit\Framework\TestCase;

class BatchNormTest extends TestCase
{
    protected $layer;

    public function setUp()
    {
        $this->layer = new BatchNorm(0.1);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(BatchNorm::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Hidden::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);
    }

    public function test_width()
    {
        $this->layer->init(10);

        $this->assertEquals(10, $this->layer->width());
    }
}
