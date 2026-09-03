<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Traits;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\InvalidFanInException;
use Rubix\ML\Exceptions\InvalidFanOutException;
use Rubix\ML\Traits\AssertsShapes;
use PHPUnit\Framework\TestCase;
use Tensor\Matrix;

#[Group('Traits')]
#[CoversClass(AssertsShapes::class)]
class AssertsShapesTest extends TestCase
{
    protected ShapeAssertingFixture $fixture;

    protected function setUp() : void
    {
        $this->fixture = new ShapeAssertingFixture();
    }

    #[Test]
    public function assertSameShapePassesWithMatchingShapes() : void
    {
        $output = Matrix::quick([[1.0, 2.0], [3.0, 4.0]]);
        $target = Matrix::quick([[5.0, 6.0], [7.0, 8.0]]);

        $this->fixture->checkSameShape($output, $target);

        $this->assertTrue(true);
    }

    #[Test]
    public function assertSameShapeThrowsOnMismatchedShapes() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $output = Matrix::quick([[1.0, 2.0], [3.0, 4.0]]);
        $target = Matrix::quick([[5.0]]);

        $this->fixture->checkSameShape($output, $target);
    }

    #[Test]
    public function validateFanInFanOutPassesWithValidValues() : void
    {
        $this->fixture->checkFanInFanOut(1, 1);

        $this->assertTrue(true);
    }

    #[Test]
    public function validateFanInFanOutThrowsOnInvalidFanIn() : void
    {
        $this->expectException(InvalidFanInException::class);

        $this->fixture->checkFanInFanOut(0, 3);
    }

    #[Test]
    public function validateFanInFanOutThrowsOnInvalidFanOut() : void
    {
        $this->expectException(InvalidFanOutException::class);

        $this->fixture->checkFanInFanOut(3, 0);
    }
}

/**
 * A fixture that exposes the protected trait methods for testing.
 *
 * @internal
 */
class ShapeAssertingFixture
{
    use AssertsShapes;

    /**
     * @param Matrix $output
     * @param Matrix $target
     */
    public function checkSameShape(Matrix $output, Matrix $target) : void
    {
        $this->assertSameShape(output: $output, target: $target);
    }

    public function checkFanInFanOut(int $fanIn, int $fanOut) : void
    {
        $this->validateFanInFanOut(fanIn: $fanIn, fanOut: $fanOut);
    }
}
