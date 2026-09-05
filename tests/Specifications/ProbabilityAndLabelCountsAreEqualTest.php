<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Specifications;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Specifications\ProbabilityAndLabelCountsAreEqual;
use PHPUnit\Framework\TestCase;

#[Group('Specifications')]
#[CoversClass(ProbabilityAndLabelCountsAreEqual::class)]
class ProbabilityAndLabelCountsAreEqualTest extends TestCase
{
    #[Test]
    public function checkPassesWithEqualCounts() : void
    {
        $specification = ProbabilityAndLabelCountsAreEqual::with(
            probabilities: [
                ['red' => 0.9, 'green' => 0.1],
                ['red' => 0.2, 'green' => 0.8],
            ],
            labels: [0, 1]
        );

        $this->assertNull($specification->check());
    }

    #[Test]
    public function checkPassesWithEmptyArrays() : void
    {
        $specification = ProbabilityAndLabelCountsAreEqual::with(
            probabilities: [],
            labels: []
        );

        $this->assertNull($specification->check());
    }

    #[Test]
    public function checkThrowsWithUnequalCounts() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $specification = ProbabilityAndLabelCountsAreEqual::with(
            probabilities: [
                ['red' => 0.9, 'green' => 0.1],
            ],
            labels: [0, 1, 0]
        );

        $specification->check();
    }
}
