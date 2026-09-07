<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Initializers\Uniform;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('Initializers')]
#[CoversClass(Uniform::class)]
class UniformTest extends TestCase
{
    /**
     * @var Uniform
     */
    protected Uniform $initializer;

    protected function setUp() : void
    {
        $this->initializer = new Uniform(0.05);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(Uniform::class, $this->initializer);
        $this->assertInstanceOf(Initializer::class, $this->initializer);
    }

    #[Test]
    public function initialize() : void
    {
        $w = $this->initializer->initialize(4, 3);

        $this->assertInstanceOf(Matrix::class, $w);
        $this->assertEquals([3, 4], $w->shape());
    }
}
