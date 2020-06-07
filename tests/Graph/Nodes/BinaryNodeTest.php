<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Graph\Nodes\Traits\HasBinaryChildren;
use PHPUnit\Framework\TestCase;

/**
 * @group Nodes
 * @covers \Rubix\ML\Graph\Nodes\BinaryNode
 */
class BinaryNodeTest extends TestCase
{
    /**
     * @var \Rubix\ML\Graph\Nodes\BinaryNode
     */
    protected $node;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->node = new class() implements BinaryNode {
            use HasBinaryChildren;
        };
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    /**
     * @test
     */
    public function attachDetachLeft() : void
    {
        $this->node->attachLeft(new class() implements BinaryNode {
            use HasBinaryChildren;
        });

        $node = $this->node->left();

        $this->assertNotNull($node);
        $this->assertInstanceOf(BinaryNode::class, $node);
        $this->assertEquals(2, $this->node->height());
        $this->assertEquals(-1, $this->node->balance());
        $this->assertEquals(1, $node->height());
        $this->assertFalse($this->node->leaf());
        $this->assertTrue($node->leaf());

        $this->node->detachLeft();

        $this->assertNull($this->node->left());
    }

    /**
     * @test
     */
    public function attachDetachRight() : void
    {
        $this->node->attachRight(new class() implements BinaryNode {
            use HasBinaryChildren;
        });

        $node = $this->node->right();

        $this->assertNotNull($node);
        $this->assertInstanceOf(BinaryNode::class, $node);
        $this->assertEquals(2, $this->node->height());
        $this->assertEquals(1, $this->node->balance());
        $this->assertEquals(1, $node->height());
        $this->assertFalse($this->node->leaf());
        $this->assertTrue($node->leaf());

        $this->node->detachRight();

        $this->assertNull($this->node->right());
    }

    protected function assertPreConditions() : void
    {
        $this->assertEquals(1, $this->node->height());
        $this->assertEquals(0, $this->node->balance());
        $this->assertNull($this->node->left());
        $this->assertNull($this->node->right());
        $this->assertTrue($this->node->leaf());
    }
}
