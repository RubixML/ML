<?php

use Rubix\ML\Transformers\Strategies\Strategy;
use Rubix\ML\Transformers\Strategies\DefaultValue;
use PHPUnit\Framework\TestCase;

class DefaultValueTest extends TestCase
{
    protected $strategy;

    public function setUp()
    {
        $this->strategy = new DefaultValue('z');
    }

    public function test_build_blurry_mean_strategy()
    {
        $this->assertInstanceOf(DefaultValue::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_guess_categorical_value()
    {
        $values = ['a', 'b', 'c', 'd'];

        $this->strategy->fit($values);

        $value = $this->strategy->guess();

        $this->assertEquals('z', $value);
    }

    public function test_guess_continuous_value()
    {
        $strategy = new DefaultValue(6);

        $values = [1, 2, 3, 4, 5];

        $strategy->fit($values);

        $value = $strategy->guess();

        $this->assertEquals(6, $value);
    }
}
