<?php

namespace Rubix\ML\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Graph\Nodes\Isolator;
use Rubix\ML\Graph\Nodes\BinaryNode;
use PHPUnit\Framework\TestCase;

class IsolatorTest extends TestCase
{
    protected const COLUMN = 1;
    protected const VALUE = 3.;

    protected const SAMPLES = [
        [5., 2., -3],
        [6., 4., -5],
    ];

    /**
     * @var \Rubix\ML\Graph\Nodes\Isolator
     */
    protected $node;

    public function setUp() : void
    {
        $groups = [
            Unlabeled::quick([self::SAMPLES[0]]),
            Unlabeled::quick([self::SAMPLES[1]]),
        ];
        
        $this->node = new Isolator(self::COLUMN, self::VALUE, $groups);
    }

    public function test_build_node() : void
    {
        $this->assertInstanceOf(Isolator::class, $this->node);
        $this->assertInstanceOf(BinaryNode::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    public function test_split() : void
    {
        $dataset = Unlabeled::quick(self::SAMPLES);

        $node = Isolator::split($dataset);

        $this->assertInstanceOf(Isolator::class, $node);
    }

    public function test_column() : void
    {
        $this->assertSame(self::COLUMN, $this->node->column());
    }

    public function test_value() : void
    {
        $this->assertSame(self::VALUE, $this->node->value());
    }
}
