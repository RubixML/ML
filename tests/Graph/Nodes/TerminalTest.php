<?php

namespace Rubix\Tests\Graph\Nodes;

use Rubix\ML\Graph\Nodes\Node;
use Rubix\ML\Graph\Nodes\Terminal;
use PHPUnit\Framework\TestCase;

class TerminalTest extends TestCase
{
    protected $node;

    protected $params;

    public function setUp()
    {
        $this->params = [20, ['meta' => true]];

        $this->node = new Terminal(...$this->params);
    }

    public function test_build_node()
    {
        $this->assertInstanceOf(Terminal::class, $this->node);
        $this->assertInstanceOf(Node::class, $this->node);
    }

    public function test_get_outcome()
    {
        $this->assertEquals($this->params[0], $this->node->outcome());
    }

    public function test_get_meta()
    {
        $this->assertEquals($this->params[1]['meta'], $this->node->meta('meta'));
    }
}
