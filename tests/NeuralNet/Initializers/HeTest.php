<?php

namespace Rubix\Tests\NeuralNet\Initializers;

use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\TestCase;

class HeTest extends TestCase
{
    protected $initializer;

    public function setUp()
    {
        $this->initializer = new He();
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(He::class, $this->initializer);
        $this->assertInstanceOf(Initializer::class, $this->initializer);
    }

    public function test_initialize()
    {
        $w = $this->initializer->initialize(4, 3);

        $this->assertInstanceOf(Matrix::class, $w);
        $this->assertEquals([3, 4], $w->shape());
    }
}
