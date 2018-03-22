<?php

use Rubix\Engine\Math\Random;
use PHPUnit\Framework\TestCase;

class RandomTest extends TestCase
{
    public function setUp()
    {
        //
    }

    public function test_random_item()
    {
        $item = Random::item(['good', 'bad', 'ugly']);

        $this->assertTrue(is_string($item));
        $this->assertTrue(in_array($item, ['good', 'bad', 'ugly']));
    }

    public function test_random_float()
    {
        $float = Random::float(1.0, 99.99);

        $this->assertTrue(is_float($float));
        $this->assertTrue($float >= 1.0);
        $this->assertTrue($float <= 99.99);
    }

    public function test_random_int()
    {
        $float = Random::int(1, 99);

        $this->assertTrue(is_int($float));
        $this->assertTrue($float >= 1);
        $this->assertTrue($float <= 99);
    }

    public function test_random_even_int()
    {
        $even = Random::even(1, 100);

        $this->assertTrue(is_int($even));
        $this->assertTrue($even >= 1);
        $this->assertTrue($even <= 100);
        $this->assertTrue($even % 2 === 0);
    }

    public function test_random_odd_int()
    {
        $odd = Random::odd(1, 100);

        $this->assertTrue(is_int($odd));
        $this->assertTrue($odd >= 1);
        $this->assertTrue($odd <= 100);
        $this->assertTrue($odd % 2 === 1);
    }
}
