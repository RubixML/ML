<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Graph\Nodes;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Graph\Nodes\Average;
use PHPUnit\Framework\TestCase;

#[Group('Nodes')]
#[CoversClass(Average::class)]
class AverageTest extends TestCase
{
    protected const float OUTCOME = 44.21;

    protected const float IMPURITY = 6.0;

    protected const int N = 3;

    protected Average $node;

    protected function setUp() : void
    {
        $this->node = new Average(
            outcome: self::OUTCOME,
            impurity: self::IMPURITY,
            n: self::N
        );
    }

    public function testOutcome() : void
    {
        $this->assertSame(self::OUTCOME, $this->node->outcome());
    }

    public function testImpurity() : void
    {
        $this->assertSame(self::IMPURITY, $this->node->impurity());
    }

    public function testN() : void
    {
        $this->assertSame(self::N, $this->node->n());
    }
}
