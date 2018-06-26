<?php

use Rubix\ML\Graph\Node;
use Rubix\ML\Graph\BinaryNode;
use Rubix\ML\Graph\GraphObject;
use PHPUnit\Framework\TestCase;

class BinaryNodeTest extends TestCase
{
    protected $node;

    public function setUp()
    {
        $this->node = new BinaryNode(['coolness_factor' => 'medium']);
    }

    public function test_create_binary_node()
    {
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
        $this->assertInstanceOf(GraphObject::class, $this->node);

        $this->assertEquals('medium', $this->node->coolness_factor);

        $this->assertEquals(1, $this->node->height());
        $this->assertNull($this->node->left());
        $this->assertNull($this->node->right());
        $this->assertTrue($this->node->isLeaf());
    }

    public function test_attach_left_child()
    {
        $this->assertNull($this->node->left());
        $this->assertTrue($this->node->isLeaf());

        $this->node->attachLeft(new BinaryNode(['coolness_factor' => 'low']));

        $this->assertNotNull($this->node->left());
        $this->assertInstanceOf(BinaryNode::class, $this->node->left());
        $this->assertEquals(2, $this->node->height());
        $this->assertEquals(1, $this->node->left()->height());
        $this->assertFalse($this->node->isLeaf());
    }

    public function test_attach_right_child()
    {
        $this->assertNull($this->node->right());
        $this->assertTrue($this->node->isLeaf());

        $this->node->attachRight(new BinaryNode(['coolness_factor' => 'high']));

        $this->assertNotNull($this->node->right());
        $this->assertInstanceOf(BinaryNode::class, $this->node->right());
        $this->assertEquals(2, $this->node->height());
        $this->assertEquals(1, $this->node->right()->height());
        $this->assertFalse($this->node->isLeaf());
    }

    public function test_detach_left_child()
    {
        $this->node->attachLeft(new BinaryNode(['coolness_factor' => 'high']));

        $this->assertNotNull($this->node->left());
        $this->assertInstanceOf(BinaryNode::class, $this->node->left());
        $this->assertEquals('high', $this->node->left()->coolness_factor);

        $this->node->detachLeft();

        $this->assertNull($this->node->left());
    }

    public function test_detach_right_child()
    {
        $this->node->attachRight(new BinaryNode(['coolness_factor' => 'low']));

        $this->assertNotNull($this->node->right());
        $this->assertInstanceOf(BinaryNode::class, $this->node->right());
        $this->assertEquals('low', $this->node->right()->coolness_factor);

        $this->node->detachRight();

        $this->assertNull($this->node->right());
    }
}
