<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Best;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Leaf;
use Rubix\ML\Graph\Nodes\Outcome;
use Rubix\ML\Graph\Nodes\Decision;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class BestTest extends TestCase
{
    protected const OUTCOME = 'cat';
    
    protected const PROBABILITIES = [
        'cat' => 0.7,
        'pencil' => 0.3,
    ];

    protected const IMPURITY = 14.1;

    protected const N = 6;

    /**
     * @var \Rubix\ML\Graph\Nodes\Best
     */
    protected $node;

    public function setUp() : void
    {
        $this->node = new Best(self::OUTCOME, self::PROBABILITIES, self::IMPURITY, self::N);
    }

    public function test_build_node() : void
    {
        $this->assertInstanceOf(Best::class, $this->node);
        $this->assertInstanceOf(Outcome::class, $this->node);
        $this->assertInstanceOf(Decision::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Leaf::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    public function test_outcome() : void
    {
        $this->assertEquals(self::OUTCOME, $this->node->outcome());
    }

    public function test_probabilities() : void
    {
        $this->assertEquals(self::PROBABILITIES, $this->node->probabilities());
    }

    public function test_impurity() : void
    {
        $this->assertEquals(self::IMPURITY, $this->node->impurity());
    }

    public function test_n() : void
    {
        $this->assertEquals(self::N, $this->node->n());
    }
}
