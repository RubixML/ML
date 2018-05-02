<?php

use Rubix\Engine\Transformers\Imputers\Imputer;
use Rubix\Engine\Transformers\Imputers\DefaultValue;
use PHPUnit\Framework\TestCase;

class DefaultValueTest extends TestCase
{
    protected $imputer;

    public function setUp()
    {
        $this->imputer = new DefaultValue('z');
    }

    public function test_build_blurry_mean_imputer()
    {
        $this->assertInstanceOf(DefaultValue::class, $this->imputer);
        $this->assertInstanceOf(Imputer::class, $this->imputer);
    }

    public function test_guess_categorical_value()
    {
        $values = ['a', 'b', 'c', 'd'];

        $this->imputer->fit($values);

        $value = $this->imputer->impute();

        $this->assertEquals('z', $value);
    }

    public function test_guess_continuous_value()
    {
        $imputer = new DefaultValue(6);

        $values = [1, 2, 3, 4, 5];

        $imputer->fit($values);

        $value = $imputer->impute();

        $this->assertEquals(6, $value);
    }
}
