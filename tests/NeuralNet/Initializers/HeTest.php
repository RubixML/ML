<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

use function max;
use function sqrt;

#[Group('Initializers')]
#[CoversClass(He::class)]
class HeTest extends TestCase
{
    /**
     * @var He
     */
    protected He $initializer;

    protected function setUp() : void
    {
        $this->initializer = new He();
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(He::class, $this->initializer);
        $this->assertInstanceOf(Initializer::class, $this->initializer);
    }

    #[Test]
    public function initialize() : void
    {
        $w = $this->initializer->initialize(4, 3);

        $this->assertInstanceOf(Matrix::class, $w);
        $this->assertEquals([3, 4], $w->shape());
    }

    #[Test]
    public function initializeHasCorrectScale() : void
    {
        $fanIn = 1000;
        $fanOut = 100;
        $limit = sqrt(6.0 / $fanIn);

        $w = $this->initializer->initialize($fanIn, $fanOut);
        $maxAbs = max($w->abs()->max()->asArray());

        $this->assertLessThanOrEqual($limit * 1.0001, $maxAbs);
        $this->assertGreaterThanOrEqual(0.98 * $limit, $maxAbs);
    }
}
