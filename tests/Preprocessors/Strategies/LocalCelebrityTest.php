<?php

use Rubix\Engine\Preprocessors\Strategies\LocalCelebrity;
use PHPUnit\Framework\TestCase;

class LocalCelebrityTest extends TestCase
{
    protected $strategy;

    public function setUp()
    {
        $this->strategy = new LocalCelebrity();
    }

    public function test_build_local_celebrity_strategy()
    {
        $this->assertInstanceOf(LocalCelebrity::class, $this->strategy);
    }

    public function test_guess_value()
    {
        $data = ['a', 'a', 'b'];

        $this->assertTrue(in_array($this->strategy->guess($data), ['a', 'b']));
    }
}
