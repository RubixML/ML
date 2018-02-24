<?php

use Rubix\Engine\BinaryNode;
use PHPUnit\Framework\TestCase;

class BinaryNodeTest extends TestCase
{
    protected $node;

    public function setUp()
    {
        $this->node = new BinaryNode(1, ['coolness_factor' => 'medium']);
    }

    public function test_create_binary_node()
    {
        $this->assertTrue($this->node instanceof BinaryNode);
        $this->assertEquals(1, $this->node->value());
        $this->assertEquals('medium', $this->node->coolness_factor);
        $this->assertEquals(1, $this->node->height());
        $this->assertNull($this->node->left());
        $this->assertNull($this->node->right());
        $this->assertTrue($this->node->isBalanced());
        $this->assertTrue($this->node->isLeaf());
    }

    public function test_attach_left_child()
    {
        $this->assertNull($this->node->left());
        $this->assertTrue($this->node->isBalanced());
        $this->assertTrue($this->node->isLeaf());

        $node = $this->node->attachLeft(new BinaryNode(2));

        $this->assertNotNull($node->left());
        $this->assertTrue($node->left() instanceof BinaryNode);
        $this->assertEquals(2, $node->height());
        $this->assertEquals(1, $node->left()->height());
        $this->assertTrue($node->isBalanced());
        $this->assertFalse($node->isLeaf());
    }

    public function test_attach_right_child()
    {
        $this->assertNull($this->node->right());
        $this->assertTrue($this->node->isBalanced());
        $this->assertTrue($this->node->isLeaf());

        $node = $this->node->attachRight(new BinaryNode(2));

        $this->assertNotNull($node->right());
        $this->assertTrue($node->right() instanceof BinaryNode);
        $this->assertEquals(2, $node->height());
        $this->assertEquals(1, $node->right()->height());
        $this->assertTrue($node->isBalanced());
        $this->assertFalse($node->isLeaf());
    }
}
