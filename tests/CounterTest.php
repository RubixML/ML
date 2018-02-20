<?php

use Rubix\Engine\Counter;
use PHPUnit\Framework\TestCase;

class CounterTest extends TestCase
{
    protected $counter;

    public function setUp()
    {
        $this->counter = new Counter();
    }

    public function test_build_counter()
    {
        $this->assertTrue($this->counter instanceof Counter);
    }

    public function test_current_id()
    {
        $this->assertEquals(0, $this->counter->current());
    }

    public function test_increment_id()
    {
        $this->assertEquals(1, $this->counter->next());
        $this->assertEquals(2, $this->counter->next());
        $this->assertEquals(3, $this->counter->next());
    }

    public function test_build_with_offset()
    {
        $counter = new Counter(11);

        $this->assertEquals(11, $counter->current());
    }
}
