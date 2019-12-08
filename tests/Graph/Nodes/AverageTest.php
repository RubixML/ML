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

    /**
     * @var \Rubix\ML\Graph\Nodes\Average
     */
    protected $node;

    public function setUp() : void
    {
        $this->node = new Average(self::OUTCOME, self::IMPURITY, self::N);
    }

    public function test_build_node() : void
    {
        $this->assertInstanceOf(Average::class, $this->node);
        $this->assertInstanceOf(Outcome::class, $this->node);
        $this->assertInstanceOf(Decision::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Leaf::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    public function test_outcome() : void
    {
        $this->assertSame(self::OUTCOME, $this->node->outcome());
    }

    public function test_impurity() : void
    {
        $this->assertSame(self::IMPURITY, $this->node->impurity());
    }

    public function test_n() : void
    {
        $this->assertSame(self::N, $this->node->n());
    }
}
