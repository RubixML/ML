<?php

namespace Rubix\ML\Tests\Graph;

use Rubix\ML\Graph\Tree;
use Rubix\ML\Graph\ITree;
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
