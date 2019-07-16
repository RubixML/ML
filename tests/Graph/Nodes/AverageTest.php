<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Leaf;
use Rubix\ML\Graph\Nodes\Average;
use Rubix\ML\Graph\Nodes\Outcome;
use Rubix\ML\Graph\Nodes\Decision;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class AverageTest extends TestCase
{
    protected const OUTCOME = 44.21;
    protected const IMPURITY = 6.;
    protected const N = 3;

    public function test_build_node()
    {
        $node = new Average(self::OUTCOME, self::IMPURITY, self::N);

        $this->assertInstanceOf(Average::class, $node);
        $this->assertInstanceOf(Outcome::class, $node);
        $this->assertInstanceOf(Decision::class, $node);
        $this->assertInstanceOf(BinaryNode::class, $node);
        $this->assertInstanceOf(Leaf::class, $node);
        $this->assertInstanceOf(Node::class, $node);

        $this->assertEquals(self::OUTCOME, $node->outcome());
        $this->assertEquals(self::IMPURITY, $node->impurity());
        $this->assertEquals(self::N, $node->n());
    }
}
