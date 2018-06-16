<?php

use Rubix\ML\Graph\Node;
use Rubix\ML\Graph\Tree;
use Rubix\ML\Graph\BinaryTree;
use Rubix\ML\Graph\BinaryNode;
use PHPUnit\Framework\TestCase;

class BinaryTreeTest extends TestCase
{
    protected $tree;

    public function setUp()
    {
        $this->tree = new BinaryTree();
    }

    public function test_build_tree()
    {
        $this->assertInstanceOf(BinaryTree::class, $this->tree);
        $this->assertInstanceOf(Tree::class, $this->tree);

        $this->assertEquals(null, $this->tree->root());
    }

    public function test_set_root_node()
    {
        $this->tree->setRoot(new BinaryNode());

        $this->assertInstanceOf(BinaryNode::class, $this->tree->root());
    }
}
