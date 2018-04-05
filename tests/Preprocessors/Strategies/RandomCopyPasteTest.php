<?php

use Rubix\Engine\Preprocessors\Strategies\RandomCopyPaste;
use PHPUnit\Framework\TestCase;

class RandomCopyPasteTest extends TestCase
{
    protected $strategy;

    public function setUp()
    {
        $this->strategy = new RandomCopyPaste();
    }

    public function test_build_random_copy_paste_strategy()
    {
        $this->assertInstanceOf(RandomCopyPaste::class, $this->strategy);
    }

    public function test_guess_value()
    {
        $data = [1, 2, 'a'];

        $this->assertTrue(in_array($this->strategy->guess($data), [1, 2, 'a']));
    }
}
