<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Initializers\Normal;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('Initializers')]
#[CoversClass(Normal::class)]
class NormalTest extends TestCase
{
    /**
     * @var Normal
     */
    protected Normal $initializer;

    protected function setUp() : void
    {
        $this->initializer = new Normal(0.05);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(Normal::class, $this->initializer);
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
