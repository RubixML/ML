<?php

namespace Rubix\Tests\Graph\Trees;

use Rubix\ML\Graph\Trees\Tree;
use Rubix\ML\Graph\Trees\ITree;
use PHPUnit\Framework\TestCase;

class ITreeTest extends TestCase
{
    protected $tree;

    public function setUp()
    {
        $this->tree = new ITree();
    }

    public function test_build_tree()
    {
        $this->assertInstanceOf(ITree::class, $this->tree);
        $this->assertInstanceOf(Tree::class, $this->tree);
    }
}
