<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Leaf;
use Rubix\ML\Graph\Nodes\Purity;
use Rubix\ML\Graph\Nodes\Outcome;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class OutcomeTest extends TestCase
{
    protected const OUTCOME = 'cat';
    
    protected const PROBABILITIES = [
        'cat' => 0.7,
        'pencil' => 0.3,
    ];

    protected const IMPURITY = 14.1;
    protected const N = 6;

    public function test_build_node()
    {
        $node = new Outcome(self::OUTCOME, self::PROBABILITIES, self::IMPURITY, self::N);

        $this->assertInstanceOf(Outcome::class, $node);
        $this->assertInstanceOf(Purity::class, $node);
        $this->assertInstanceOf(BinaryNode::class, $node);
        $this->assertInstanceOf(Leaf::class, $node);
        $this->assertInstanceOf(Node::class, $node);

        $this->assertEquals(self::OUTCOME, $node->class());
        $this->assertEquals(self::PROBABILITIES, $node->probabilities());
        $this->assertEquals(self::IMPURITY, $node->impurity());
        $this->assertEquals(self::N, $node->n());
    }
}
