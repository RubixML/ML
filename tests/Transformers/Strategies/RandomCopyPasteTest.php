<?php

use Rubix\ML\Transformers\Strategies\Strategy;
use Rubix\ML\Transformers\Strategies\RandomCopyPaste;
use PHPUnit\Framework\TestCase;

class RandomCopyPasteTest extends TestCase
{
    protected $values;

    protected $strategy;

    public function setUp()
    {
        $this->values = [1, 2, 3, 4, 5];

        $this->strategy = new RandomCopyPaste();
    }

    public function test_build_random_copy_paste_strategy()
    {
        $this->assertInstanceOf(RandomCopyPaste::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_guess_value()
    {
        $this->strategy->fit($this->values);

        $value = $this->strategy->guess();

        $this->assertContains($value, $this->values);
    }
}
