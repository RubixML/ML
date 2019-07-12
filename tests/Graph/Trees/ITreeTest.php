<?php

namespace Rubix\ML\Tests\Graph\Trees;

use Rubix\ML\Graph\Nodes\Cell;
use Rubix\ML\Graph\Trees\Tree;
use Rubix\ML\Graph\Trees\ITree;
use Rubix\ML\Graph\Nodes\Isolator;
use Rubix\ML\Graph\Trees\BinaryTree;
use Rubix\ML\Graph\Nodes\BinaryNode;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

class ITreeTest extends TestCase
{
    protected const RANDOM_SEED = 0;

    protected $generator;

    protected $tree;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'east' => new Blob([5, -2, -2]),
            'west' => new Blob([0, 5, -3]),
        ], [0.5, 0.5]);

        $this->tree = new ITree();

        srand(self::RANDOM_SEED);
    }

    public function test_build_tree()
    {
        $this->assertInstanceOf(ITree::class, $this->tree);
        $this->assertInstanceOf(BinaryTree::class, $this->tree);
        $this->assertInstanceOf(Tree::class, $this->tree);

        $this->assertNull($this->tree->root());
        $this->assertEquals(0, $this->tree->height());
    }

    public function test_c()
    {
        $this->assertEquals(3.748880484475505, ITree::c(10));
        $this->assertEquals(8.364671030072245, ITree::c(100));
        $this->assertEquals(12.969940887100174, ITree::c(1000));
        $this->assertEquals(17.575112063754766, ITree::c(10000));
        $this->assertEquals(22.180282259643523, ITree::c(100000));
    }

    public function test_grow_range()
    {
        $this->tree->grow($this->generator->generate(50));

        $this->assertInstanceOf(Isolator::class, $this->tree->root());
        $this->assertInstanceOf(BinaryNode::class, $this->tree->root());

        $this->assertGreaterThan(5, $this->tree->height());

        $sample = $this->generator->generate(1)->row(0);

        $node = $this->tree->search($sample);

        $this->assertInstanceOf(Cell::class, $node);
        $this->assertInstanceOf(BinaryNode::class, $node);
    }
}
