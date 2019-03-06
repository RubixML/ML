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

    public function test_build_node()
    {
        $groups = [
            Unlabeled::quick([self::SAMPLES[0]]),
            Unlabeled::quick([self::SAMPLES[1]]),
        ];
        
        $node = new Isolator(self::COLUMN, self::VALUE, $groups);

        $this->assertInstanceOf(Isolator::class, $node);
        $this->assertInstanceOf(BinaryNode::class, $node);
        $this->assertInstanceOf(Node::class, $node);
    }

    public function test_split()
    {
        $dataset = Unlabeled::quick(self::SAMPLES);

        $node = Isolator::split($dataset);

        $this->assertInstanceOf(Isolator::class, $node);
    }
}
