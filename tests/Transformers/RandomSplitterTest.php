<?php

use Rubix\Engine\Transformers\RandomSplitter;
use PHPUnit\Framework\TestCase;

class RandomSplitterTest extends TestCase
{
    protected $transformer;

    public function setUp()
    {
        $samples = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'man', 'sitting',
            'at', 'a', 'bus', 'stop', 'drinking', 'a', 'can', 'of', 'coke'];

        $this->transformer = new RandomSplitter($samples, 0.3);
    }

    public function test_build_splitter()
    {
        $this->assertTrue($this->transformer instanceof RandomSplitter);
    }
}
