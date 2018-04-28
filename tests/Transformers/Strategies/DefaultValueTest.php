<?php

use Rubix\Engine\Transformers\Strategies\Strategy;
use Rubix\Engine\Transformers\Strategies\DefaultValue;
use PHPUnit\Framework\TestCase;

class DefaultValueTest extends TestCase
{
    protected $strategy;

    public function setUp()
    {
        $this->strategy = new DefaultValue('z', 0);
    }

    public function test_build_blurry_mean_strategy()
    {
        $this->assertInstanceOf(DefaultValue::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_guess_categorical_value()
    {
        $data = ['a', 'b', 'c', 'd'];

        $value = $this->strategy->guess($data);

        $this->assertEquals('z', $value);
    }

    public function test_guess_continuous_value()
    {
        $data = [1, 2, 3, 4];

        $value = $this->strategy->guess($data);

        $this->assertEquals(0, $value);
    }
}
