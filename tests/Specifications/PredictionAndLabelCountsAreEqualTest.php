<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Specifications;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Specifications\PredictionAndLabelCountsAreEqual;
use PHPUnit\Framework\TestCase;

#[Group('Specifications')]
#[CoversClass(PredictionAndLabelCountsAreEqual::class)]
class PredictionAndLabelCountsAreEqualTest extends TestCase
{
    #[Test]
    public function checkPassesWithEqualCounts() : void
    {
        $specification = PredictionAndLabelCountsAreEqual::with(
            predictions: ['red', 'green', 'red'],
            labels: [0, 1, 0]
        );

        $this->assertNull($specification->check());
    }

    #[Test]
    public function checkPassesWithEmptyArrays() : void
    {
        $specification = PredictionAndLabelCountsAreEqual::with(
            predictions: [],
            labels: []
        );

        $this->assertNull($specification->check());
    }

    #[Test]
    public function checkThrowsWithUnequalCounts() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $specification = PredictionAndLabelCountsAreEqual::with(
            predictions: ['red', 'green'],
            labels: [0, 1, 0]
        );

        $specification->check();
    }
}
