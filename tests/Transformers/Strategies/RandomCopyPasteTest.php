<?php

use Rubix\Engine\Transformers\Strategies\Strategy;
use Rubix\Engine\Transformers\Strategies\RandomCopyPaste;
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
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_guess_value()
    {
        $data = [1, 2, 'a'];

        $value = $this->strategy->guess($data);

        $this->assertContains($value, $data);
    }
}
