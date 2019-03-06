<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Cell;
use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Leaf;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class CellTest extends TestCase
{
    protected const SAMPLES = [
        [5., 2., -3],
        [6., 4., -5],
    ];

    protected const DEPTH = 8;
    protected const C = 7.1544313298030655;

    public function test_build_node()
    {
        $node = new Cell(self::C);

        $this->assertInstanceOf(Cell::class, $node);
        $this->assertInstanceOf(BinaryNode::class, $node);
        $this->assertInstanceOf(Leaf::class, $node);
        $this->assertInstanceOf(Node::class, $node);

        $this->assertEquals(self::C, $node->depth());
    }

    public function test_terminate()
    {
        $dataset = Unlabeled::quick(self::SAMPLES);

        $node = Cell::terminate($dataset, self::DEPTH);

        $this->assertEquals(self::C, $node->depth());
    }
}
