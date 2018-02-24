<?php

use Rubix\Engine\GADDAG;
use Rubix\Engine\Node;
use PHPUnit\Framework\TestCase;

class GADDAGTest extends TestCase
{
    protected $graph;

    public function setUp()
    {
        $this->graph = new GADDAG([
            'its', 'just', 'literally', 'a', 'normal', 'car', 'in', 'space', 'i', 'like', 'the',
            'absurdity', 'of', 'that', 'its', 'silly', 'and', 'fun', 'but', 'i', 'think', 'that',
            'fun', 'silly', 'things', 'are', 'important', 'stay', 'calm']);
    }

    public function test_build_gaddag()
    {
        $this->assertTrue($this->graph instanceof GADDAG);
        $this->assertTrue($this->graph->root() instanceof Node);
    }

    public function test_has_word()
    {
        $this->assertTrue($this->graph->has('just'));
        $this->assertTrue($this->graph->has('space'));
        $this->assertTrue($this->graph->has('i'));
        $this->assertTrue($this->graph->has('things'));
        $this->assertFalse($this->graph->has('cant'));
        $this->assertFalse($this->graph->has('carbon'));
        $this->assertFalse($this->graph->has('thin'));
    }

    public function test_insert_word()
    {
        $this->assertFalse($this->graph->has('pig'));

        $graph = $this->graph->insert('pig');

        $this->assertTrue($graph->has('pig'));
    }
}
