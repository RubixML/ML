<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Graph\Nodes;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Graph\Nodes\Best;
use PHPUnit\Framework\TestCase;

#[Group('Nodes')]
#[CoversClass(Best::class)]
class BestTest extends TestCase
{
    protected const string OUTCOME = 'cat';

    protected const array PROBABILITIES = [
        'cat' => 0.7,
        'pencil' => 0.3,
    ];

    protected const float IMPURITY = 14.1;

    protected const int N = 6;

    protected Best $node;

    protected function setUp() : void
    {
        $this->node = new Best(
            outcome: self::OUTCOME,
            probabilities: self::PROBABILITIES,
            impurity: self::IMPURITY,
            n: self::N
        );
    }

    public function testOutcome() : void
    {
        $this->assertEquals(self::OUTCOME, $this->node->outcome());
    }

    public function testProbabilities() : void
    {
        $this->assertEquals(self::PROBABILITIES, $this->node->probabilities());
    }

    public function testImpurity() : void
    {
        $this->assertEquals(self::IMPURITY, $this->node->impurity());
    }

    public function testN() : void
    {
        $this->assertEquals(self::N, $this->node->n());
    }
}
