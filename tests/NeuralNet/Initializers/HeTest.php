<?php

namespace Rubix\ML\Tests\NeuralNet\Initializers;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\NeuralNet\Parameter;
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
        $parameter = $this->initializer->initialize([3, 4]);

        $this->assertInstanceOf(Parameter::class, $parameter);
        $this->assertInstanceOf(Matrix::class, $parameter->param());
        $this->assertEquals([3, 4], $parameter->param()->shape());
    }

    #[Test]
    public function initializeVector() : void
    {
        $parameter = $this->initializer->initialize([4]);

        $this->assertInstanceOf(Parameter::class, $parameter);
        $this->assertInstanceOf(Vector::class, $parameter->param());
        $this->assertEquals([4], $parameter->param()->shape());
    }

    #[Test]
    public function initializeHasCorrectScale() : void
    {
        $fanIn = 1000;
        $fanOut = 100;
        $limit = sqrt(6.0 / $fanIn);

        $parameter = $this->initializer->initialize([$fanOut, $fanIn]);
        $maxAbs = max($parameter->param()->abs()->max()->asArray());

        $this->assertLessThanOrEqual($limit * 1.0001, $maxAbs);
        $this->assertGreaterThanOrEqual(0.98 * $limit, $maxAbs);
    }
}
