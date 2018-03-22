<?php

use Rubix\Engine\Transformers\Splitter;
use PHPUnit\Framework\TestCase;

class SplitterTest extends TestCase
{
    protected $transformer;

    public function setUp()
    {
        $samples = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'man', 'sitting',
            'at', 'a', 'bus', 'stop', 'drinking', 'a', 'can', 'of', 'coke'];

        $this->transformer = new Splitter($samples, 0.5);
    }

    public function test_build_splitter()
    {
        $this->assertTrue($this->transformer instanceof Splitter);
    }

    public function test_get_training_data()
    {
        $this->assertEquals(['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'man'], $this->transformer->training());
    }

    public function test_get_testing_data()
    {
        $this->assertEquals(['sitting', 'at', 'a', 'bus', 'stop', 'drinking', 'a', 'can', 'of', 'coke'], $this->transformer->testing());
    }
}
